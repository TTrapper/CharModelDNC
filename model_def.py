import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import numpy as np
import tensorflow as tf
import data_pipe

def make_model(batchsize, seqlen, numlayers, layersize, numheads):
    numclasses = 256 # assume utf-8 bytes
    char_embed_size = 32
    chars_per_slot = layersize // char_embed_size
    assert seqlen % chars_per_slot == 0
    num_in_slots = seqlen // chars_per_slot
    # Embed Characters
    char_ids_2 = tf.keras.Input(shape=(seqlen), batch_size=batchsize)
    char_embeds_3 = tf.keras.layers.Embedding(numclasses, char_embed_size)(char_ids_2)
    # Convolution
    """
    char_embeds_3 = tf.keras.layers.LayerNormalization()(char_embeds_3)
    char_embeds_3 = tf.keras.layers.Conv1D(filters=char_embed_size, kernel_size=3, strides=1,
            padding='same', data_format='channels_last', dilation_rate=1,
            activation=tf.nn.relu)(char_embeds_3)

    """
    char_embeds_3 = tf.keras.layers.LayerNormalization()(char_embeds_3)
    # Attach some extra values to the inputs to act as a extra memory slots for the DNC
    num_extra_slots = 1
    extra_slots = tf.zeros([batchsize, num_extra_slots * chars_per_slot, char_embed_size])
    char_embeds_3 = tf.concat([extra_slots, char_embeds_3], axis=1)
    num_in_slots += num_extra_slots
    # DNC layers
    # TODO: no longer processed as a sequence, move layers out of an RNN cell
    char_embeds_3 = tf.reshape(char_embeds_3, [batchsize, 1, num_in_slots * layersize])
    cell = DNCCell(layersize, num_in_slots, numlayers, numheads)
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True, stateful=False)
    out = rnn(char_embeds_3)
    forward = out[0] # to predict next char
    reconstruct = out[1] # to predicts the masked inputs

    # Reconstruct the masked input characters
    reconstruct = tf.reshape(reconstruct, [batchsize, num_in_slots, numheads * char_embed_size])
    reconstruct = reconstruct[:, num_extra_slots:, :] # Remove the extra memory slots
    reconstruct = tf.keras.layers.Dense(numheads * char_embed_size, tf.nn.relu)(reconstruct)
    reconstruct = tf.keras.layers.Dropout(rate=0.1)(reconstruct)
    reconstruct = tf.reshape(reconstruct, [batchsize, seqlen, char_embed_size])
    reconstruct = tf.keras.layers.Dense(numclasses, None)(reconstruct)
    # Mask the logits that don't correspond to masked inputs
    mask = char_ids_2 == 0
    mask = tf.tile(tf.expand_dims(mask, axis=2), [1, 1, numclasses])
    reconstruct = tf.where(mask, reconstruct, tf.zeros_like(reconstruct))
    reconstruct = tf.keras.layers.Masking()(reconstruct)

    # Predict ahead one character
    forward = tf.keras.layers.Dense(layersize, tf.nn.relu)(forward) # batch, 1, layersize
    forward = tf.keras.layers.Dropout(rate=0.1)(forward)
    logits_3 = tf.keras.layers.Dense(numclasses, None)(forward)

    # Model
    model = tf.keras.Model(inputs=char_ids_2, outputs=(logits_3, reconstruct))
    return model


class DNCCell(tf.keras.layers.Layer):
    """
    A cross between a transformer and differentiable neural computer. Conceptually, it arranges a
    sequence of inputs as a 2D grid of memory slots. Each layer uses attention to selectively
    read from and write to the memory slots.

    Arguments:
        units: Positive integer, dimensionality of the output space.
        numheads: Positive integer, number of read/write attention heads per memory state
        memsize: Positive integer, mumber of memory states
        depth: Positive integer, number of layers that will share this cell's memory.
    """

    def __init__(self, units, memsize, depth, numheads, **kwargs):
        if units % numheads != 0:
            raise ValueError('The number of attention heads must evenly divide the number of units'
                    '(got units:{}, numheads:{})'.format(units, numheads))
        self.units = units
        self.numheads = numheads
        self.headsize = units // numheads
        self.state_size = (units * memsize,)
        self.output_size = (units, units)
        self.memsize = memsize
        self.depth = depth
        super(DNCCell, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'numheads': self.numheads,
            'units': self.units,
            'memsize': self.memsize,
            'depth': self.depth
        })
        return config

    def build(self, input_shape):
        if input_shape[-1] % self.units != 0:
            raise ValueError('input size must evenly divide the output size, got: {}, {}'.format(
                input_shape[-1], self.units))
        if input_shape[-1] // self.units != self.memsize:
            raise ValueError('The specified number of input slots does not match the input size: '\
                    + 'got: {}, {}'.format(self.memsize, input_shape[-1] // self.units))
        self.layers = []
        self.normlayer = tf.keras.layers.LayerNormalization()
        for layernum in range(self.depth):
            layer = {}
            layer['readlayer'] = tf.keras.layers.Dense(self.numheads, None)
            layer['writelayer'] = tf.keras.layers.Dense(self.numheads, None)
            layer['kernel'] = tf.keras.layers.Dense(self.units, tf.nn.relu)
            layer['project'] = tf.keras.layers.Dense(self.units, None)
            self.layers.append(layer)
        self.readout = tf.keras.layers.Dense(self.numheads, None)
        self.built = True

    def call(self, inputs, states, training=False):
        memory_4 = tf.reshape(states[0], [-1, self.memsize, self.numheads, self.headsize])
        inputs = tf.reshape(inputs, [-1, self.memsize, self.numheads, self.headsize])
        memory_4 += inputs
        # Reshape and transform so that each head contains a contiguous subsequence
        memory_4 = tf.reshape(memory_4, [-1, self.numheads, self.memsize, self.headsize])
        memory_4 = tf.transpose(memory_4, [0, 2, 1, 3]) # batch, memsize, heads, headsize
        for layernum, layer in enumerate(self.layers):
            # Return inputs to their original sequence for the second half of the layers
            if layernum == len(self.layers) // 2:
                memory_4 = tf.transpose(memory_4, [0, 2, 1, 3]) # batch, heads, memsize, headsize
                memory_4 = tf.reshape(memory_4, [-1, self.memsize, self.numheads, self.headsize])
            outputs, memory_4 = self._runlayer(memory_4, layer)
        out = self._read_memory(memory_4, readlayer=self.readout)
        memory_2 = tf.reshape(memory_4, [-1, self.memsize * self.units])
        return out, memory_2

    def _runlayer(self, memory_4, layer):
        """
        Process one of this cell's potentially many layers.
        memory_4: The 4D memory tensor extracted from the the cell state
        kernels: A dict containing this layer's 'readlayer', 'writelayer', and 'kernel'
        """
        attended_mem_2 = self._read_memory(memory_4, layer['readlayer'])
        # Compute a new value from the attended memory
        new_memval_2 = layer['kernel'](attended_mem_2)
        new_memval_2 = tf.keras.layers.Dropout(rate=0.1)(new_memval_2)
        new_memval_2 = layer['project'](new_memval_2)
        new_memval_2 = self.normlayer(new_memval_2)
        # Write the new value to memory
        keys = self._make_memory_keys(memory_4) + tf.expand_dims(new_memval_2, axis=1)
        write_weights_4 = self._process_heads(layer['writelayer'](keys))
        new_memval_4 = tf.reshape(new_memval_2, [-1, 1, self.numheads, self.headsize])
        memory_4 = ((1 - write_weights_4) * memory_4) + (write_weights_4 * new_memval_4)
        return new_memval_2, memory_4

    def _make_memory_keys(self, memory_4):
        """
        Creates attention keys from memory slots by combining each slot with the mean of all slots
        """
        memory = tf.reshape(memory_4, [-1, self.memsize, self.units])
        mean_memory = tf.reduce_mean(memory, axis=1, keepdims=True)
        return memory + mean_memory

    def _read_memory(self, memory_4, readlayer):
        keys = self._make_memory_keys(memory_4) # batch, memsize, numheads * headsize
        weights = self._process_heads(readlayer(keys)) # batch, memsize, numheads, 1
        attended_mem_3 = tf.reshape(weights * memory_4, [-1, self.memsize, self.units])
        return tf.reduce_sum(attended_mem_3, axis=1)

    def _process_heads(self, weights_3):
        """
        Takes read/write logits and seperately applies softmax to each read/write head
        """
        weights_3 = tf.nn.softmax(weights_3, axis=1)
        return tf.expand_dims(weights_3, axis=3)


@tf.function
def sample_logits(logits, temperature):
    prediction = tf.random.categorical(logits/temperature, num_samples=1)
    return prediction

def _string_to_inputs(input_string, batchsize):
    contextlen = len(input_string)
    input_ids = data_pipe.string_to_ids(tf.constant(input_string))
    # Stateful layers need their batchsize to be fixed, so just passing the same input in for now.
    input_ids = tf.expand_dims(input_ids, axis=0)
    input_ids = tf.tile(input_ids, [batchsize, 1])
    return input_ids

def run_inference(model, input_string, numpredict, temperature=1e-16):
    print('\n******************************************')
    print('softmax temperature: {}'.format(temperature))
    print('******************************************\n')
    temperature = tf.constant(temperature)
    # Convert string to integers. Prepend the start-of-text byte.
    input_string = bytes( input_string, 'utf-8')
    batchsize = model.input_shape[0]
    seqlen = model.input_shape[1]
    input_ids = _string_to_inputs(input_string, batchsize)
    result = [input_ids]
    pad = tf.ones([batchsize, seqlen], input_ids.dtype)
    input_ids = tf.concat([pad, input_ids], axis=1)
    for _ in range(numpredict):
        input_ids = input_ids[:, -seqlen:]
        outputs = model.predict_on_batch(input_ids)
        outputs = outputs[0]
        latest_logits = outputs[:, -1, :]
        prediction = sample_logits(latest_logits, temperature)
        prediction = tf.cast(prediction, input_ids.dtype)
        input_ids = tf.concat([input_ids, prediction], axis=1)
        result.append(prediction)
    # Convert to strings
    outstring = data_pipe.ids_to_python_string(tf.concat(result, axis=1))
    # Print the results for each sequence in the batch
    max_numlines = 8
    for line in outstring[:max_numlines]:
        print(line.replace('\\n', '\n'), '\n')
        print('--------------------------------------------')
    return outstring

if __name__ == '__main__':
    # Run inference
    config = json.load(open('./config.json'))
    model = make_model(16, config['maxseqlen'], config['numlayers'], config['layersize'],
            config['numheads'])
    model.load_weights('./model.h5')
    numpredict = 512
    lines = ['This sentence is an example']
    context = ' '
    _ = run_inference(model, context, numpredict, 1e-16)
    lines = run_inference(model, context, numpredict, 0.5)
    _ = run_inference(model, context, numpredict, 0.75)

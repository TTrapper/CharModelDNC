import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import data_pipe

def make_model(batchsize, numlayers, layersize, memsize, numheads, return_state=False):
    numclasses = 256 # assume utf-8 bytes
    inputs = tf.keras.Input((None,), batch_size=batchsize)
    # Embed Characters
    char_embeds_3 = tf.keras.layers.Embedding(numclasses, layersize)(inputs)
    # Sequence layers
    cell = DNCCell(layersize, memsize, numlayers, numheads)
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=return_state,
            stateful=True)
    outputs = rnn(char_embeds_3)
    if return_state:
        char_embeds_3 = outputs[0]
        memory_states = tuple(outputs[1:])
    else:
        char_embeds_3 = outputs
    # Output layer
    logits_3 = tf.keras.layers.Dense(numclasses,  None)(char_embeds_3)
    # Model
    if return_state:
        model = tf.keras.Model(inputs=inputs, outputs=(logits_3, (memory_states)))
        return model
    else:
        model = tf.keras.Model(inputs=inputs, outputs=logits_3)
        return model

class DNCCell(tf.keras.layers.Layer):
    """
    A dumbed down version of a differentiable neural computer. At each timestep it reads from
    memory, computes a new outputvalue, and writes to memory.

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
        self.state_size = (units * memsize,) +\
                (depth * (numheads * (memsize + 1), numheads * memsize))
        self.nested_state_size = (units * memsize,
                depth * ((numheads * (memsize + 1), numheads * memsize),))
        self.output_size = units
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
        if input_shape[-1] != self.units:
            raise ValueError('input size must match output size, got: {}, {}'.format(
                input_shape[-1], self.units))
        self.layers = []
        for layernum in range(self.depth):
            layer = {}
            layer['readlayer'] = tf.keras.layers.Dense(self.numheads * (self.memsize + 1), None)
            layer['writelayer'] = tf.keras.layers.Dense(self.numheads * self.memsize, None)
            layer['kernel'] = tf.keras.layers.Dense(self.units, tf.nn.relu)
            self.layers.append(layer)
        self.built = True

    def call(self, inputs, states, training=False):
        memory_4 = tf.reshape(states[0], [-1, self.memsize, self.numheads, self.headsize])
        readwrites_by_layer = []
        for layer in self.layers:
            outputs, memory_4, readwrites = self._runlayer(inputs,
                    memory_4, layer)
            readwrites_by_layer.append(readwrites)
        memory_2 = tf.reshape(memory_4, [-1, self.memsize * self.units])
        return outputs, [memory_2, readwrites_by_layer]

    def _runlayer(self, inputs_2, memory_4, layer):
        """
        Process one of this cell's potentially many layers.
        inputs_2: 2D tensor input to call or output from a previous layer
        memory_4: The 4D memory tensor extracted from the the cell state
        kernels: A dict containing this layer's 'readlayer', 'writelayer', and 'kernel'
        """
        # Take the memory state from the previous step or layer and concat the current input to it
        inputs_4 = tf.reshape(inputs_2, [-1, 1, self.numheads, self.headsize])
        attended_mem_4 = tf.concat([memory_4, inputs_4], axis=1)
        # Compute attention over memory
        memstate_2 = tf.reshape(attended_mem_4, [-1, self.memsize + 1, self.units])
        memstate_2 = tf.reduce_mean(memstate_2, axis=1)
        read_weights_2 = layer['readlayer'](memstate_2)
        read_weights_4, read_weights_2 = self._process_heads(read_weights_2)
        attended_mem_4 *= read_weights_4
        attended_mem_3 = tf.reshape(attended_mem_4, [-1, self.memsize + 1, self.units])
        # Compute a new value from the attended memory
        new_memval_2 = layer['kernel'](tf.reduce_sum(attended_mem_3, axis=1))
        # Write the new value to memory
        write_weights_2 = layer['writelayer'](new_memval_2)
        write_weights_4, write_weights_2 = self._process_heads(write_weights_2, False)
        new_memval_4 = tf.reshape(new_memval_2, [-1, 1, self.numheads, self.units//self.numheads])
        memory_4 = ((1 - write_weights_4) * memory_4) + (write_weights_4 * new_memval_4)
        return new_memval_2, memory_4, (read_weights_2, write_weights_2)

    def _process_heads(self, weights_2, read=True):
        """
        Takes flattened read/write logits and seperately applies softmax to each read/write head
        """
        numitems = self.memsize + 1 if read else self.memsize
        weights_3 = tf.reshape(weights_2, [-1, numitems, self.numheads])
        weights_3 = tf.nn.softmax(weights_3, axis=1)
        weights_2 = tf.reshape(weights_3, [-1, numitems * self.numheads])
        return tf.expand_dims(weights_3, axis=3), weights_2

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
    input_string = bytes(chr(2) + input_string, 'utf-8')
    input_ids = _string_to_inputs(input_string, model.input_shape[0])
    result = [input_ids]
    for _ in range(numpredict):
        outputs = model.predict_on_batch(input_ids)
        latest_logits = outputs[:, -1, :]
        prediction = sample_logits(latest_logits, temperature)
        prediction = tf.cast(prediction, input_ids.dtype)
        input_ids = prediction
        result.append(prediction)
    # Remove the GO byte and convert to strings
    outstring = data_pipe.ids_to_python_string(tf.concat(result, axis=1)[:, 1:])
    # Print the results for each sequence in the batch
    for line in outstring:
        print(line.replace('\\n', '\n'), '\n')
        print('--------------------------------------------')
    return outstring

def inspect_memory(model, input_string):
    """
    Plots the read and write weights used by DNC layers as they process their memory
    model: a keras model compiled with the DNC states appended to its outputs.
    input_string: the string to process
    """
    assert model.input_shape[0] == 1 # TODO support other batch sizes
    # Find out how many DNC layers the model has
    numlayers = len(model.output_shape[1][1:])//2 # FIXME this is a hard coded solution
    reads = [[] for _ in range(numlayers)]
    writes = [[] for _ in range(numlayers)]
    # Convert string to integers.
    input_string = bytes(chr(2) + input_string, 'utf-8')
    input_ids = _string_to_inputs(input_string, model.input_shape[0])
    # Pass each character in and collect the read/write weights
    for idx in range(input_ids.shape[1]):
        current_input = input_ids[:, idx:idx+1]
        results = model.predict_on_batch(current_input)
        # Manually extract the states and pack them into layers
        readwrites = results[2:]
        readwrites_by_layer = [(readwrites[i: i+2]) for i in range(0, len(readwrites), 2)]
        for layernum, layer in enumerate(readwrites_by_layer):
            reads[layernum].append(layer[0])
            writes[layernum].append(layer[1])
    # Plot the weights
    layered_reads = [np.concatenate(layer, axis=0) for layer in reads]
    layered_writes = [np.concatenate(layer, axis=0) for layer in writes]
    maxseqlen = 256
    ticks = range(len(input_string[:maxseqlen]))
    ticklabels = [chr(c) for c in input_string[-maxseqlen:]]
    for layernum, (reads, writes) in enumerate(zip(layered_reads, layered_writes)):
        fig, axs = plt.subplots(nrows=2)
        fig.suptitle('Layer {}'.format(layernum))
        yticks = range(reads.shape[1])
        axs[0].imshow(np.transpose(reads[-maxseqlen:]), cmap='gray', aspect='auto')
        axs[0].set_yticks(yticks, minor=False)
        axs[0].set_xticks(ticks, minor=False)
        axs[0].set_xticklabels(ticklabels, minor=False)
        axs[0].set_title('Read Weights')
        axs[1].imshow(np.transpose(writes[-maxseqlen:]), cmap='gray', aspect='auto')
        yticks = range(writes.shape[1])
        axs[1].set_yticks(yticks, minor=False)
        axs[1].set_xticks(ticks, minor=False)
        axs[1].set_xticklabels(ticklabels, minor=False)
        axs[1].set_title('Write Weights')
    plt.show()


if __name__ == '__main__':
    # Run inference
    config = json.load(open('./config.json'))
    model = tf.keras.models.load_model('./model.h5', custom_objects={'DNCCell':DNCCell},
        compile=True)
    numpredict = 512
    lines = ['This sentence is an example']
    context = 'she'
    _ = run_inference(model, context, numpredict, 1e-16)
    lines = run_inference(model, context, numpredict, 0.5)
    _ = run_inference(model, context, numpredict, 0.75)
    # Plot the attention weights
    config = json.load(open('./config.json'))
    model = make_model(1, config['numlayers'], config['layersize'], config['memsize'],
            config['numheads'], return_state=True)
    model.load_weights('./model.h5')
    inspect_memory(model, lines[0])

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import process_config
import numpy as np
import tensorflow as tf
import data_pipe
import random

def make_model(config):
    inputs_collector = []
    outputs_collector = []
    predict_ahead_layers = make_predict_ahead_layers(config['numclasses'], 0.02)
    char_embed_layer = tf.keras.layers.Embedding(config['numclasses'], config['char_embed_size'],
            trainable=config['train_char_embeds'])
    char_embeds = embed_characters(char_embed_layer, config, inputs_collector)
    for blocknum, block_config in enumerate(config['blocks']):
        namescope = 'block_{}'.format(blocknum)
        char_embeds = build_block(char_embeds, block_config, config, inputs_collector,
                outputs_collector, namescope)
        build_predict_ahead(char_embeds, block_config['predict_ahead'], config['batchsize'],
                char_embed_layer, predict_ahead_layers, block_config['trainable'],
                config['dropout'], namescope, inputs_collector, outputs_collector)
    context = char_embeds
    next_char = tf.keras.layers.Dense(config['numclasses'], None, name='out_logits')(context)
    outputs_collector.append(next_char)
    model = tf.keras.Model(inputs=tuple(inputs_collector), outputs=tuple(outputs_collector))
    return model

def make_predict_ahead_layers(numclasses, dropout):
    predict_ahead_layers = []
    predict_ahead_layers.append(tf.keras.layers.GRU(512, return_sequences=True, unroll=False,
        recurrent_dropout=dropout, dropout=dropout))
    predict_ahead_layers.append(tf.keras.layers.Dense(numclasses, None))
    return predict_ahead_layers

def embed_characters(char_embed_layer, config, inputs_collector):
    seqlen = config['seqlen']
    batchsize = config['batchsize']
    char_ids = tf.keras.Input(shape=(seqlen), batch_size=batchsize, name='char_ids')
    char_ids_normalized = tf.keras.Input(shape=(seqlen), batch_size=batchsize,
            name='char_ids_normalized')
    inputs_collector.append(char_ids)
    inputs_collector.append(char_ids_normalized)
    char_embeds = char_embed_layer(char_ids)
    char_embeds_normalized = char_embed_layer(char_ids_normalized)
    char_embeds = tf.reduce_mean(tf.stack([char_embeds, char_embeds_normalized]), axis=0)
    char_embeds = tf.keras.layers.LayerNormalization(trainable=config['train_char_embeds'])\
            (char_embeds)
    char_embeds = tf.keras.layers.Dropout(rate=config['dropout'])(char_embeds)
    return char_embeds # batch, seqlen, embedsize

def build_block(char_embeds, block_config, config, inputs_collector, outputs_collector, namescope):
    char_embed_size = config['char_embed_size']
    char_embeds = tf.reshape(char_embeds, [-1, block_config['numheads'], block_config['memsize'],
        char_embed_size])
    char_embeds = tf.reshape(char_embeds,
            [-1, 1, block_config['memsize'] * block_config['wordsize']])
    cell = DNCCell(block_config['wordsize'], block_config['memsize'], block_config['numlayers'],
            block_config['numheads'], kernel_size=block_config['kernelsize'],
            fixed_write_slots=block_config['writeheads'] // block_config['numheads'],
            num_position_embeds=128, dropout=config['dropout'])
    rnn = tf.keras.layers.RNN(cell, return_sequences=False, return_state=True, stateful=False,
            trainable=block_config['trainable'], name=namescope + '_rnn')
    char_embeds = rnn(char_embeds)[1]
    context_size = (block_config['subseqlen'] * char_embed_size)
    if block_config['compress']:
        window_size = config['char_embed_size'] * block_config['writeheads']
        if window_size * 2 <= context_size:
            window_size *= 2
        char_embeds = tf.reshape(char_embeds, [-1, window_size])
        char_embeds = linear_project(char_embeds, window_size // 2, block_config['trainable'],
                config['dropout'], namescope + '_compress')
        context_size = context_size // 2
    char_embeds = tf.reshape(char_embeds, [-1, context_size])
    return char_embeds

def create_predict_ahead_inputs(context, ahead, batchsize, embedlayer, inputs_collector):
    # Define inputs and embed chars
    char_ids = tf.keras.Input(shape=(None, ahead), batch_size=batchsize)
    inputs_collector.append(char_ids)
    char_embeds = embedlayer(char_ids) # batchsize, subseq-batchsize, ahead, embedsize
    char_embed_size = char_embeds.shape[-1]
    char_embeds = tf.reshape(char_embeds, [-1, ahead, char_embed_size])
    # Attach context vector to each char embed
    context = tf.expand_dims(context, axis=1) # batch, 1, context_size
    context = tf.tile(context, [1, ahead, 1])
    context = tf.concat([context, char_embeds], axis=2)
    return context

def build_predict_ahead(context, num_ahead, batchsize, char_embed_layer, predict_ahead_layers,
        trainable, dropout, namescope, inputs_collector, outputs_collector):
    if num_ahead == 0:
        return
    if len(predict_ahead_layers) == 0:
        raise ValueError('build_predict_ahead was called with empty list: predict_ahead_layers')
    context = linear_project(context, 1024, trainable, dropout, namescope + '_context_ahead')
    ahead = create_predict_ahead_inputs(context, num_ahead, batchsize, char_embed_layer,
            inputs_collector)
    for layer in predict_ahead_layers:
         ahead = layer(ahead)
    outputs_collector.append(ahead)
    return context

def linear_project(values, size, trainable, dropout, namescope):
    values = tf.keras.layers.Dense(size, None, trainable=trainable,
            name=namescope + '_dense')(values)
    values = tf.keras.layers.LayerNormalization(trainable=trainable,
            name=namescope + '_layernorm')(values)
    return tf.keras.layers.Dropout(rate=dropout)(values)


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

    def __init__(self, units, memsize, depth, numheads, kernel_size=None, fixed_write_slots=None,
            num_position_embeds=None, dropout=0.0, **kwargs):

        if units % numheads != 0:
            raise ValueError('The number of attention heads must evenly divide the number of units'
                    '(got units:{}, numheads:{})'.format(units, numheads))
        self.fixed_write_slots = fixed_write_slots
        if self.fixed_write_slots:
            assert self.fixed_write_slots > 0 and self.fixed_write_slots <= memsize
            assert memsize % self.fixed_write_slots == 0
        self.num_position_embeds = num_position_embeds if num_position_embeds else memsize
        assert self.num_position_embeds >= memsize
        self.kernel_size = kernel_size if kernel_size else units
        self.units = units
        self.numheads = numheads
        self.headsize = units // numheads
        self.state_size = (units * memsize,)
        self.output_size = (2 * units, units)
        self.memsize = memsize
        self.depth = depth
        self.dropout = dropout
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
        self.normlayer = tf.keras.layers.LayerNormalization(scale=True)
        self.pos_size = 32
        self.pos = tf.keras.layers.Embedding(self.num_position_embeds, self.pos_size)
        num_write_heads = self.numheads * self.fixed_write_slots if self.fixed_write_slots else\
                self.numheads
        project_size = num_write_heads * self.headsize if\
                self.fixed_write_slots else self.units + self.pos_size
        self.project_size = project_size
        for layernum in range(self.depth):
            layer = {}
            layer['readlayer'] = tf.keras.layers.Dense(self.numheads, None)
            layer['writelayer'] = tf.keras.layers.Dense(num_write_heads, None)
            layer['kernel'] = tf.keras.layers.Dense(self.kernel_size, tf.nn.relu)
            layer['project'] = tf.keras.layers.Dense(project_size, None)
            self.layers.append(layer)
        self.first_readout = tf.keras.layers.Dense(self.numheads, None)
        self.second_readout = tf.keras.layers.Dense(self.numheads, None)
        self.built = True

    def call(self, inputs, states, training=False):
        memory_4 = tf.reshape(states[0], [-1, self.memsize, self.numheads, self.headsize])
        inputs = tf.reshape(inputs, [-1, self.memsize, self.numheads, self.headsize])
        memory_4 += inputs
        for layernum, layer in enumerate(self.layers):
            outputs, memory_4 = self._runlayer(memory_4, layer)
        out = self._read_memory(memory_4, readlayer=self.first_readout)
        second_out = self._read_memory(memory_4, readlayer=self.second_readout)
        out = tf.concat([out, second_out], axis=1)
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
        new_memval_2 = tf.keras.layers.Dropout(rate=self.dropout)(new_memval_2)
        new_memval_2 = layer['project'](new_memval_2)
        new_memval_2 = self.normlayer(new_memval_2)
        new_memval_2 = tf.keras.layers.Dropout(rate=self.dropout)(new_memval_2)
        # Write the new value to memory
        if self.fixed_write_slots:

            write_slots = tf.reshape(memory_4,
                    [-1, self.memsize // self.fixed_write_slots, self.project_size])
            new_memval = tf.reshape(new_memval_2, [-1, 1, self.project_size])
            write_slots += new_memval
            write_slots = self._attach_pos(write_slots)
            write_slots = tf.reshape(write_slots, [-1, self.project_size + self.pos_size])
            weights = layer['writelayer'](write_slots)
            weights = tf.nn.sigmoid(weights) # batch, memsize * heads
            weights = tf.reshape(weights, [-1, self.memsize, self.numheads])
            weights = tf.expand_dims(weights, axis=3) # batch, memsize, heads, 1
            new_memval = tf.reshape(new_memval_2,
                    [-1, self.fixed_write_slots, self.numheads, self.headsize])
            new_memval = tf.tile(new_memval, [1, self.memsize // self.fixed_write_slots, 1, 1])
            memory_4 = ((1 - weights) * memory_4) + (weights * new_memval)

        else: # write to one memory slot per head (via attention along heads)
            keys = self._make_memory_keys(memory_4) + tf.expand_dims(new_memval_2, axis=1)
            write_weights_4 = self._process_heads(layer['writelayer'](keys))
            new_memval_2 = new_memval_2[:, :-self.pos_size] # strip off extra from position embeds
            new_memval_4 = tf.reshape(new_memval_2, [-1, 1, self.numheads, self.headsize])
            memory_4 = ((1 - write_weights_4) * memory_4) + (write_weights_4 * new_memval_4)
        return new_memval_2, memory_4

    def _make_memory_keys(self, memory_4):
        """
        Creates attention keys from memory slots by combining each slot with the mean of all slots,
        and attaching position embeddings to each slot.
        """
        memory = tf.reshape(memory_4, [-1, self.memsize, self.units]) # batch, memsize, units
        mean_memory = tf.reduce_mean(memory, axis=1, keepdims=True) # batch, 1, units
        memory += mean_memory # batch, memsize, units
        return self._attach_pos(memory)

    def _attach_pos(self, tensor_3D):
        batchsize, num_items, itemsize = tensor_3D.shape
        # position embeds taken from the end of the range so that if memsize is increased, the value
        # to the right of memory (which is the value at the end of a sequence), keeps the same pos.
        pos = self.pos(tf.range(self.num_position_embeds - num_items, self.num_position_embeds))
        pos = tf.tile(tf.expand_dims(pos, axis=0), [batchsize, 1, 1])# batch, items, pos_size
        return tf.concat([tensor_3D, pos], axis=2) # batch, items, itemsize + pos_size

    def _read_memory(self, memory_4, readlayer):
        keys = self._make_memory_keys(memory_4) # batch, memsize, numheads * headsize
        weights = self._process_heads(readlayer(keys)) # batch, memsize, numheads, 1
        read_dropout = 0.0
        weights = tf.keras.layers.Dropout(rate=read_dropout)(weights)
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

def run_inference(model, config, input_string, numpredict, temperature=1e-16):
    print('\n******************************************')
    print('softmax temperature: {}'.format(temperature))
    print('******************************************\n')
    temperature = tf.constant(temperature)
    # Convert string to integers. Prepend the start-of-text byte.
    input_string = bytes( input_string, 'utf-8')
    batchsize = model.input_shape[0][0]
    seqlen = model.input_shape[0][1]
    input_ids = _string_to_inputs(input_string, batchsize)
    result = [input_ids]
    # FIXME not actually used should build a version of the model that doesn't have these
    subseqlens, n_ahead_for_each_subseq = data_pipe.collect_subsequences_to_num_ahead(config)
    ahead = max(n_ahead_for_each_subseq)
    pad = tf.ones([batchsize, seqlen + ahead], input_ids.dtype)
    input_ids = tf.concat([pad, input_ids], axis=1)[:, -(seqlen + ahead):]
    future = data_pipe.select_targets(input_ids, config['seqlen'], subseqlens,
            n_ahead_for_each_subseq)
    future = tuple([data_pipe.add_go_byte(x) for x in future])
    for _ in range(numpredict):
        input_ids = input_ids[:, -seqlen:]
        outputs = model.predict_on_batch((input_ids, data_pipe.normalize(input_ids)) + future )

        first_logits = outputs[-1][:, :]
        prediction = sample_logits(first_logits, temperature)
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
    config = process_config.load_config()
    config['batchsize'] = 4
    model = make_model(config)
    model.load_weights('./model.h5', by_name=True, skip_mismatch=True)
    model.summary()
    numpredict = 512
    lines = ['This sentence is an example']
    context = '. '
    _ = run_inference(model, config, context, numpredict, 1e-16)
    lines = run_inference(model, config, context, numpredict, 0.5)
    _ = run_inference(model, config, context, numpredict, 0.75)
    _ = run_inference(model, config, context, numpredict, 1.0)

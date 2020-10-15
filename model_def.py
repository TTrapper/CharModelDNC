import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import process_config
import numpy as np
import tensorflow as tf
import data_pipe
import random

class LinearProject(tf.keras.layers.Layer):
    """
    Wraps a linear projection followed by layer normalization and dropout
    """
    def __init__(self, size, dropout, **kwargs):
        super(LinearProject, self).__init__(kwargs)
        self.dropout = dropout
        self.size = size
        self.dense = tf.keras.layers.Dense(size, None)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.trainable = True
        if 'trainable' in kwargs:
            self.trainable = kwargs['trainable']

    def get_config(self):
        config = super(LinearProject, self).get_config()
        config.update({'size':self.size, 'dropout':self.dropout})
        return config

    def call(self, values):
        values = self.layernorm(self.dense(values))
        return tf.keras.layers.Dropout(rate=self.dropout)(values)

class Positioner(tf.keras.layers.Layer):
    """
    Takes a sequence of shape: [batchsize, numitems, itemsize] and adds position information by
    performing a non-linear combination of each item with a one-hot representation of its position
    """
    def __init__(self, dropout, **kwargs):
        super(Positioner, self).__init__(kwargs)
        self.dropout = dropout
        self.trainable = True
        if 'trainable' in kwargs:
            self.trainable = kwargs['trainable']

    def get_config(self):
        config = super(Positioner, self).get_config()
        config.update({'dropout':self.dropout})
        return config

    def build(self, input_shape):
        batchsize, nslots, slotsize = input_shape
        self.dense = tf.keras.layers.Dense(slotsize, tf.nn.relu)

    def call(self, values):
        batchsize, nslots, slotsize = values.shape
        pos = tf.one_hot(tf.range(nslots), nslots, dtype=values.dtype)
        pos = tf.tile(tf.expand_dims(pos, axis=0), [batchsize, 1, 1]) # batchsize, nslots, nslots
        values = tf.concat([values, pos], axis=2) # batch, nslots, slot_size + nslots
        values = self.dense(values)
        return tf.keras.layers.Dropout(rate=self.dropout)(values)

class Reader(tf.keras.layers.Layer):
    """
    Uses multi-head attention to read from an array of shape [batchsize, nslots, nheads, headsize].
    Additive attention is applied over the slots.
    """
    def __init__(self, kernelsize, dropout, **kwargs):
        super(Reader, self).__init__(kwargs)
        self.kernelsize = kernelsize
        self.dropout = dropout
        self.trainable = False
        if 'trainable' in kwargs:
            self.trainable = kwargs['trainable']

    def get_config(self):
        config = super(Reader, self).get_config()
        config.update({'kernelsize':self.kernelsize, 'dropout':self.dropout})
        return config

    def build(self, input_shape):
        batchsize, nslots, nheads, headsize = input_shape
        self.attend_layer = tf.keras.layers.Dense(nheads, None)
        self.kernel = tf.keras.layers.Dense(self.kernelsize, tf.nn.relu)
        self.projection_layer = LinearProject(nheads * headsize, self.dropout,
                trainable=self.trainable) # TODO trainable has to be set explicitly. why?

    @staticmethod
    def get_keys(values):
        """ Create attention keys by combining each slot with the mean of all slots """
        batchsize, nslots, nheads, headsize = values.shape
        slots = tf.reshape(values, [batchsize, nslots, nheads * headsize])
        mean_slots = tf.tile(tf.reduce_mean(slots, axis=1, keepdims=True), [1, nslots, 1])
        return tf.concat([slots, mean_slots], axis=2)

    def call(self, values, keys=None):
        batchsize, nslots, nheads, headsize = values.shape
        # Compute attention weights
        keys = self.get_keys(values) if keys is None else keys
        weights = tf.nn.softmax(self.attend_layer(keys), axis=1)
        weights = tf.expand_dims(weights, axis=3) # batch, nslots, nheads, 1
        # Apply attention and process result
        attended = tf.reshape(weights * values, [batchsize, nslots, nheads * headsize])
        attended = tf.reduce_sum(attended, axis=1)
        attended = self.kernel(attended)
        attended = tf.keras.layers.Dropout(rate=self.dropout)(attended)
        # Project to original size
        attended = self.projection_layer(attended)
        return tf.reshape(attended, [batchsize, 1, nheads,  headsize])

class MultiReader(tf.keras.layers.Layer):
    """
    Performs parallel reads on the input sequence and concatenates the results.
    """
    def __init__(self, kernelsize, dropout, numreads, **kwargs):
        super(MultiReader, self).__init__(kwargs)
        self.kernelsize = kernelsize
        self.dropout = dropout
        self.numreads = numreads
        self.trainable = False
        if 'trainable' in kwargs:
            self.trainable = kwargs['trainable']
        self.readers = [Reader(kernelsize, dropout, **kwargs) for _ in range(numreads)]

    def get_config(self):
        config = super(MultiReader, self).get_config()
        config.update({'kernelsize':self.kernelsize,
                       'dropout':self.dropout,
                       'numreads': self.numreads})
        return config

    def call(self, values):
        keys = Reader.get_keys(values) # compute keys once and reuse for each read
        reads = [reader(values, keys=keys) for reader in self.readers]
        return tf.concat(reads, axis=1)

def write(new_values, keys, old_values, trainable, namescope):
    """
    Writes new values into the old values using attention
    """
    batchsize, nslots, nheads, headsize = old_values.shape
    # Add new values to the write keys and compute write weights
    new_values = tf.reshape(new_values, [batchsize, 1, nheads * headsize])
    weights = tf.keras.layers.Dense(nheads, None, trainable=trainable,
            name=namescope + '_write_weights')(keys)
    weights = tf.nn.sigmoid(weights)
    weights = tf.reshape(weights, [-1, nslots, nheads, 1])
    # Apply write weights and update values
    new_values = tf.reshape(new_values, [batchsize, 1, nheads, headsize])
    return ((1 - weights) * old_values) + (weights * new_values)


def compress(values, trainable, dropout, namescope):
    """
    Compresses the sequence by shaping the values to put neighbouring slots together, effectively
    halving the sequence length while doubling the feature size
    """
    batchsize, nslots, nheads, headsize = values.shape
    slotsize = nheads * headsize
    values = tf.reshape(values, [batchsize, nslots // 2, 2 * slotsize])
    values = tf.keras.layers.Dense(2 * slotsize, tf.nn.relu, trainable=trainable,
            name='{}_compress_dense'.format(namescope))(values)
    values = tf.keras.layers.Dropout(rate=dropout)(values)
    return tf.reshape(values, [batchsize, nslots // 2, 2* nheads, headsize])


def make_model(config):
    inputs_collector = []
    outputs_collector = []
    char_embed_layer = tf.keras.layers.Embedding(config['numclasses'], config['char_embed_size'],
            trainable=config['train_char_embeds'])
    char_embeds = embed_characters(char_embed_layer, config, inputs_collector)
    previous_slotsize = char_embeds.shape[-1]
    dropout = config['dropout']
    for blocknum, block_config in enumerate(config['blocks']):
        namescope = 'block_{}'.format(blocknum)
        trainable = block_config['trainable']
        nslots = block_config['memsize']
        slotsize = block_config['wordsize']
        # Position information for each slot
        char_embeds = tf.reshape(char_embeds, [-1, nslots, previous_slotsize])
        char_embeds = Positioner(dropout=dropout, trainable=trainable)(char_embeds)
        # Project and reshape to this block's size
        char_embeds = LinearProject(slotsize, dropout, trainable=trainable)(char_embeds)
        char_embeds = tf.reshape(char_embeds, [-1, nslots, block_config['numheads'],
            config['headsize']])
        # Attention and rewrite layers
        char_embeds = MultiReader(block_config['kernelsize'], dropout, nslots,
                trainable=trainable)(char_embeds)
        if block_config['compress']:
            nslots //= 2
        char_embeds = MultiReader(block_config['kernelsize'], dropout, nslots,
                trainable=trainable)(char_embeds)
        # Logits from this block
        batchsize, nslots, nheads, headsize = char_embeds.shape
        previous_slotsize = nheads * headsize
        subseqlen = block_config['subseqlen_compressed']
        assert batchsize == config['batchsize'] * config['seqlen'] // subseqlen
        context = tf.reshape(char_embeds, [batchsize, nslots * nheads * headsize])
        prev_char = tf.keras.layers.Dense(config['numclasses'], None,
                name='{}b{}'.format(blocknum, subseqlen), trainable=trainable)(context)
        next_char = tf.keras.layers.Dense(config['numclasses'], None,
                name='{}f{}'.format(blocknum, subseqlen), trainable=trainable)(context)
        outputs_collector.append((prev_char, next_char))
    model = tf.keras.Model(inputs=tuple(inputs_collector), outputs=tuple(outputs_collector))
    return model

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
    return char_embeds # batch, seqlen, embedsize


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
    pad = tf.ones([batchsize, seqlen], input_ids.dtype)
    input_ids = tf.concat([pad, input_ids], axis=1)
    for _ in range(numpredict):
        input_ids = input_ids[:, -seqlen:]
        outputs = model.predict_on_batch((input_ids, data_pipe.normalize(input_ids)))
        first_logits = outputs[-1][-1][:, :]
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

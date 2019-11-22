import numpy as np
import tensorflow as tf

import data_pipe

def make_model(batchsize, seqlen):
    numclasses = 256 # assume utf-8 bytes
    layersize = 1024
    numlayers = 1
    inputs = tf.keras.Input((seqlen))
    # Embed Characters
    char_embeds_3 = tf.keras.layers.Embedding(numclasses, layersize)(inputs)
    # Sequence layers
    with tf.name_scope('block1'):
        char_embeds_3 = dnc_block(char_embeds_3, layersize, 4)
    with tf.name_scope('block2'):
        char_embeds_3 = dnc_block(char_embeds_3, layersize, 8)
    with tf.name_scope('block3'):
        char_embeds_3 = dnc_block(char_embeds_3, layersize, 16)
    with tf.name_scope('block4'):
        char_embeds_3 = dnc_block(char_embeds_3, layersize, 32)
    # Output layer
    logits_3 = tf.keras.layers.Dense(numclasses,  None)(char_embeds_3)
    outputs = logits_3
    # Model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def gatedrnn_block(char_embeds_3, layersize):
    """
    Scans the the sequence of char_embeds_3 and applies a simplified gated rnn
    """
    char_embeds_3 = tf.transpose(char_embeds_3, [1, 0, 2])
    char_embeds_3 = tf.scan(make_gatedrnn_fn(layersize), char_embeds_3, parallel_iterations=1)
    char_embeds_3 = tf.transpose(char_embeds_3, [1, 0, 2])
    char_embeds_3 = tf.keras.layers.LayerNormalization(axis=-1)(char_embeds_3)
    return char_embeds_3

def dnc_block(char_embeds_3, layersize, memsize):
    """
    Scans the the char embedding sequence and applies a simplified differentiable neural computeri.
    char_embeds_3: the input sequence with shape: [batchsize, sequence, layersize]
    layersize: int that matches the the input size
    memsize: the size of DNC memory (number of words) 
    """
    shape = []
    for shape_int, shape_tensor in zip(char_embeds_3.shape, tf.unstack(tf.shape(char_embeds_3))):
        shape.append(shape_int if shape_int is not None else shape_tensor)
    batchsize, seqlen, depth = shape
    initial_mem_4 = tf.zeros([seqlen, batchsize, memsize, depth], char_embeds_3.dtype)
    dnc_fn = make_dnc_fn(memsize, layersize)
    char_embeds_3 = tf.transpose(char_embeds_3, [1, 0, 2])
    _, char_embeds_3 = tf.scan(dnc_fn, (initial_mem_4, char_embeds_3), parallel_iterations=1)
    char_embeds_3 = tf.transpose(char_embeds_3, [1, 0, 2])
    char_embeds_3 = tf.keras.layers.LayerNormalization(axis=-1)(char_embeds_3)
    return char_embeds_3

def make_cumsum_decay_fn(decay):
    """
    Returns a function for computing cumulative sum with decay for use with tf.scan
    decay: the decay rate
    """
    def cumsum_decay_fn(accumulated, current):
        return decay*accumulated + current
    return cumsum_decay_fn

def make_rnn_fn(layersize):
    """
    Returns a function that computes a simplified RNN: dense(last_out + current_in). Note that there
    are not any actual recurrent weights and instead simply sums the context with the current input.
    layersize: int that matches the input size and is the size of the generated layer.
    """
    layer = tf.keras.layers.Dense(layersize, tf.nn.relu)
    def rnn_fn(accumulated, current):
        return layer(accumulated + current)
    return rnn_fn

def make_gatedrnn_fn(layersize):
    """
    Returns a function that computes a gated combination of the last output with the current input.
    This is similar to a GRU but greatly simplified.
    layersize: int that matches the input size and is the size of the generated layer.
    """
    layer = tf.keras.layers.Dense(layersize, tf.nn.relu)
    gate_layer = tf.keras.layers.Dense(layersize, tf.nn.sigmoid)
    normlayer = tf.keras.layers.LayerNormalization(axis=-1)
    def gatedrnn_fn(accumulated, current):
        candidate = layer(normlayer(accumulated + current))
        gates = gate_layer(candidate)
        out = (gates * candidate) + ((1 - gates) * accumulated)
        return out
    return gatedrnn_fn

def make_dnc_fn(memsize, layersize):
    """
    Returns a function that computes one step of a simplified differentiable neural computer.
    memsize: the number of words to hold in memory
    layersize: int that matches the input size and is the size of the generated layer.
    """
    readlayer = tf.keras.layers.Dense(memsize + 1, tf.nn.softmax)
    writelayer = tf.keras.layers.Dense(memsize, tf.nn.sigmoid)
    transformlayer = tf.keras.layers.Dense(layersize, tf.nn.relu)
    def dnc_fn(accumulated, current):
        # Take the memory state from the previous step concat the current input to it
        memory_3, _ = accumulated
        _, value_2 = current
        value_3 = tf.expand_dims(value_2, axis=1)
        attended_mem_3 = tf.concat([memory_3, value_3], axis=1)
        # Compute attention over memory
        read_weights_2 = readlayer(tf.reduce_sum(attended_mem_3, axis=1))
        read_weights_3 = tf.expand_dims(read_weights_2, axis=2)
        attended_mem_2 = tf.reduce_sum(read_weights_3 * attended_mem_3, axis=1)
        # Compute a new value from the attended memory
        transformed_2 = transformlayer(attended_mem_2)
        # Write the new value to memory
        write_weights_3 = tf.expand_dims(writelayer(transformed_2), axis=2)
        transformed_3 = tf.expand_dims(transformed_2, axis=1)
        memory_3 = ((1 - write_weights_3) * memory_3) + (write_weights_3 * transformed_3)
        return (memory_3, transformed_2)
    return dnc_fn

def add_relative_position(embeds_3, layer):
    """
    Takes a sequence of embeds and computes new embeds that contain relative positional information
    embeds_3: tensor with shape [batchsize, seqlen, embedsize]
    layer: a keras layer with a nonlinearity that combines the embedding with it's position signal
    """
    batchsize = tf.shape(embeds_3)[0]
    seqlen = embeds_3.shape[1]
    position_ids_1 = (1 + tf.range(seqlen, dtype=embeds_3.dtype))/tf.cast(seqlen, embeds_3.dtype)
    position_ids_3 = tf.expand_dims(tf.expand_dims(position_ids_1, axis=0), axis=-1)
    position_ids_3 = tf.tile(position_ids_3, [batchsize, 1, 1])
    embeds_3 = tf.concat([position_ids_3, embeds_3], axis=2)
    return layer(embeds_3)

def add_position_embedding(embeds_3, layer):
    """
    Takes a sequence of embeds and computes new embeds that contain relative positional information
    embeds_3: tensor with shape [batchsize, seqlen, embedsize]
    layer: a keras layer with a nonlinearity that combines the embedding with it's position signal
    """
    batchsize = tf.shape(embeds_3)[0]
    seqlen = embeds_3.shape[1]
    position_ids_2 = tf.one_hot(tf.range(seqlen), seqlen)
    position_ids_3 = tf.expand_dims(position_ids_2, axis=0)
    position_ids_3 = tf.tile(position_ids_3, [batchsize, 1, 1])
    embeds_3 = tf.concat([position_ids_3, embeds_3], axis=2)
    return layer(embeds_3)

def run_inference(model, context_string, seqlen):
    context_string = bytes(context_string, 'utf-8')
    contextlen = len(context_string)
    assert contextlen < seqlen
    while contextlen  < seqlen:
        context_padded = context_string.ljust(seqlen)
        input_ids = data_pipe.string_to_ids(tf.constant(context_padded))
        input_ids = tf.expand_dims(input_ids, axis=0)
        outputs = model.predict(input_ids, steps=1)
        logits = outputs[0, contextlen - 1, :]
        prediction = bytes(chr(np.argmax(logits)), 'utf-8')
        context_string += prediction
        contextlen = len(context_string)
    print(context_string)

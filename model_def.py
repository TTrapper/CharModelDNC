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

def rnn_block(char_embeds_3, layersize, gated):
    """
    Scans the the sequence of char_embeds_3 and applies a simplified gated rnn
    char_embeds_3: the inputs sequence with shape [batchsize, seqlen, layersize]
    layersize: int that matches the input depth
    gated: whether or not the RNN applies gated updates
    """
    char_embeds_3 = tf.transpose(char_embeds_3, [1, 0, 2])
    rnn_step_fn = make_rnn_fn(layersize, gated)
    char_embeds_3 = tf.scan(rnn_step_fn, char_embeds_3, parallel_iterations=1)
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

def make_rnn_fn(layersize, gated):
    """
    Returns a function that computes one step of a simplified RNN. Note there are not any actual
    recurrent weights and instead simply sums the context with the current input.
    layersize: int that matches the input size and is the size of the generated layer.
    gated: boolean determines whether or not to apply gated memory. Similar to a simlified GRU.
    """
    layer = tf.keras.layers.Dense(layersize, tf.nn.relu)
    if gated:
        gate_layer = tf.keras.layers.Dense(layersize, tf.nn.sigmoid)
    def rnn_step_fn(accumulated, current):
        candidate = layer(accumulated + current)
        if not gated:
            return candidate
        else:
            gates = gate_layer(candidate)
            return (gates * candidate) + ((1 - gates) * accumulated)
    return rnn_step_fn

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
        read_weights_2 = readlayer(tf.reduce_mean(attended_mem_3, axis=1))
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

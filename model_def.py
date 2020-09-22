import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import process_config
import numpy as np
import tensorflow as tf
import data_pipe
import random



def attention_keys(values, position_embeds):
    """
    Creates attention keys from values by combining each slot with the mean of all slots,
    and attaching position embeddings to the result.
    """
    batchsize, nslots, nheads, headsize = tf.unstack(tf.shape(values))
    values = tf.reshape(values, [batchsize, nslots, nheads * headsize])
    mean_slots = tf.reduce_mean(values, axis=1, keepdims=True) # batch, 1, nheads * headsize
    values += mean_slots
    return attach_position_embeds(values, position_embeds)

def attach_position_embeds(values, position_embeds):
    """
    Attaches position_embeds to each slot in values.
    values : 3D tensor with shape [batchsize, nslots, slotsize]
    position_embeds : Embedding layer with input_dim >= nslots
    """
    batchsize, nslots, slotsize = tf.unstack(tf.shape(values))
    num_embeds = position_embeds.input_dim
    pos = position_embeds(tf.range(nslots))
    pos = tf.tile(tf.expand_dims(pos, axis=0), [batchsize, 1, 1]) # batchsize, nslots, pos_size
    return tf.concat([values, pos], axis=2) # batch, nslots, slot_size + pos_size

def read(values, keys, trainable, namescope):
    """
    Reads from the array of values by applying attention over dim 1
    values : 4D tensor with shape [batchsize, nslots, nheads, headsize]
    """
    batchsize, nslots, nheads, headsize = values.shape
    weights = tf.keras.layers.Dense(nheads, None, trainable=trainable,
            name=namescope + '_read_weights')(keys)
    weights = tf.nn.softmax(weights, axis=1)
    weights = tf.expand_dims(weights, axis=3) # batch, nslots, nheads, 1
    attended_values = tf.reshape(weights * values, [batchsize, nslots, nheads * headsize])
    return tf.reduce_sum(attended_values, axis=1)

def write(new_values, keys, old_values, trainable, namescope):
    """
    Writes new values into the old values using attention
    """
    batchsize, nslots, nheads, headsize = old_values.shape
    # Add new values to the right keys and compute write weights
    new_values = tf.reshape(new_values, [batchsize, 1, nheads * headsize])
    pos_pad = tf.zeros([batchsize, 1, keys.shape[-1] - new_values.shape[-1]], keys.dtype)
    keys += tf.concat([new_values, pos_pad], axis=2)
    weights = tf.keras.layers.Dense(nheads, None, trainable=trainable,
            name=namescope + '_write_weights')(keys)
    weights = tf.nn.sigmoid(weights)
    weights = tf.reshape(weights, [-1, nslots, nheads, 1])
    # Apply write weights and update values
    new_values = tf.reshape(new_values, [batchsize, 1, nheads, headsize])
    return ((1 - weights) * old_values) + (weights * new_values)

def read_and_write(values, position_embeds, kernelsize, trainable, dropout, namescope):
    """
    Given an array of slotted values representing a neural memory array:
      1) read from those values using multi-head attention along the slots
      2) compute new values
      3) write the new values using multi-head attention along the slots
    """
    batchsize, nslots, nheads, headsize = values.shape
    keys = attention_keys(values, position_embeds)
    attended = read(values, keys, trainable, namescope)
    # Do a dense layer transform on the result of the read
    attended = tf.keras.layers.Dense(kernelsize, tf.nn.relu, trainable=trainable,
            name=namescope + '_kernel')(attended)
    attended = tf.keras.layers.Dropout(rate=dropout)(attended)
    # Project to original size in preparation for write
    attended = linear_project(attended, nheads * headsize, trainable, dropout, namescope)
    return write(attended, keys, values, trainable, namescope)

def compress(values, trainable, dropout, namescope):
    """
    Compresses the sequence by shaping the values to put neighbouring slots together and projecting
    each pair of slots down to the size of a single slot.
    """
    batchsize, nslots, nheads, headsize = values.shape
    slotsize = nheads * headsize
    values = tf.reshape(values, [batchsize, nslots // 2, 2 * slotsize])
    values = linear_project(values, slotsize, trainable, dropout, namescope)
    return tf.reshape(values, [batchsize, nslots // 2, nheads, headsize])

def make_model(config):
    inputs_collector = []
    outputs_collector = []
    char_embed_layer = tf.keras.layers.Embedding(config['numclasses'], config['char_embed_size'],
            trainable=config['train_char_embeds'])
    char_embeds = embed_characters(char_embed_layer, config, inputs_collector)
    previous_slotsize = char_embeds.shape[-1]
    for blocknum, block_config in enumerate(config['blocks']):
        namescope = 'block_{}'.format(blocknum)
        if previous_slotsize != block_config['wordsize']:
            char_embeds = tf.reshape(char_embeds, [-1, block_config['memsize'], previous_slotsize])
            char_embeds = linear_project(char_embeds, block_config['wordsize'],
                    block_config['trainable'], config['dropout'], namescope + '_in_projection')
        char_embeds = tf.reshape(char_embeds, [-1, block_config['memsize'],
            block_config['numheads'], config['char_embed_size']])
        for layernum, layer in enumerate(range(block_config['numlayers'])):
            layerscope = '_layer_{}'.format(layernum)
            char_embeds = read_and_write(char_embeds, char_embed_layer, block_config['kernelsize'],
                block_config['trainable'], config['dropout'], namescope + layerscope)
        if block_config['compress']:
            char_embeds = compress(char_embeds, block_config['trainable'], config['dropout'],
                    namescope)
        previous_slotsize = block_config['wordsize']
    batchsize, nslots, nheads, headsize = char_embeds.shape
    assert batchsize == config['batchsize']
    context = tf.reshape(char_embeds, [batchsize, nslots * nheads * headsize])
    next_char = tf.keras.layers.Dense(config['numclasses'], None, name='out_logits',
            trainable=True)(context)
    outputs_collector.append(next_char)
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
    char_embeds = tf.keras.layers.Dropout(rate=config['dropout'])(char_embeds)
    return char_embeds # batch, seqlen, embedsize


def linear_project(values, size, trainable, dropout, namescope):
    values = tf.keras.layers.Dense(size, None, trainable=trainable,
            name=namescope + '_dense')(values)
    values = tf.keras.layers.LayerNormalization(trainable=trainable,
            name=namescope + '_layernorm')(values)
    return tf.keras.layers.Dropout(rate=dropout)(values)

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

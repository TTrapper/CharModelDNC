import model_def
import process_config

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import random
import tensorflow as tf


def make_example_generator(filepath, seqlen):
    total_bytes = os.path.getsize(filepath)
    num_examples = total_bytes
    def example_generator():
        for i in range(num_examples):
            start = random.randint(0, total_bytes - seqlen)
            with open(filepath, 'rb') as f:
                f.seek(start)
                seq = f.read(seqlen)
                yield seq
    return example_generator

def file_to_dataset(filepath, config, maskinputs=True):
    batchsize = config['batchsize']
    seqlen = config['seqlen']
    example_generator = make_example_generator(filepath, 1 + seqlen)
    lines = tf.data.Dataset.from_generator(example_generator, tf.string, tf.TensorShape([]))
    lines = lines.prefetch(batchsize)
    lines = lines.map(lambda line: string_to_ids(line))
    lines = lines.map(lambda line: tf.reshape(line, [seqlen + 1])) # explicitly sets shape
    lines = lines.batch(batchsize, drop_remainder=True)
    lines = lines.map(lambda line: (line[:, :-1], line[:, -1:]))
    if maskinputs:
        # Randomly mask some of the input values
        lines = lines.map(lambda x,y: (randomly_mask_sampled_maskprob(x, 0.3), y))
        lines = lines.map(lambda x,y: (randomly_sequence_mask(x), y))
        chars_per_span_mask = 64 # Apply a span mask for every N chars in the sequence
#        for _ in range(seqlen // chars_per_span_mask):
#            lines = lines.map(lambda x,y: (randomly_span_mask(x), y))
    lines = lines.map(lambda x,y: ((x, normalize(x)), y))
    lines = lines.prefetch(4)
    return lines

def normalize(char_ids):
    """
    Maps chars in the ASCII range:
        - lowercase
        - maps all white space to the space char
    """
    ascii_upper_A = 65
    ascii_upper_Z = 90
    ascii_lower_a = 97
    difference = ascii_lower_a - ascii_upper_A
    char_ids = tf.where(tf.logical_and(char_ids >= ascii_upper_A, char_ids <= ascii_upper_Z),
            char_ids + difference, char_ids)
    # Any ASCII value less than 32 is whitespace or commands
    ascii_space = 32
    char_ids = tf.where(char_ids <= ascii_space, ascii_space, char_ids)
    return char_ids

def normalize_targets(char_ids_tuple):
    """
    Targets are organized in a tuple of sequences. Each sequence is normalized here, and the first
    target of the last sequence is separated and NOT normalized.
    """
    target_for_next_char = char_ids_tuple[-1][:, :, 0]
    char_ids_tuple = [normalize(y) for y in char_ids_tuple]
    return tuple(char_ids_tuple + [target_for_next_char])

def bag_of_chars(tensor):
    """
    Takes a sequence of chars and converts it a one-hot bag of chars representation
    """
    tensor = tf.one_hot(tensor, 256) # batch, seq, 256
    tensor = tf.reduce_sum(tensor, axis=1)
    return tf.where(tensor == 0, tensor, tf.ones_like(tensor))

def add_go_byte(tensor):
    # Create a go byte that will be attached to the beginning of the last dim of tensor
    go = tf.zeros(tensor.shape[:-1], tensor.dtype)
    go = tf.expand_dims(go, axis=go.shape.rank)
    tensor = tf.concat([go, tensor], axis=go.shape.rank - 1)
    # Trim the end of the last dim of the tensor to keep the orginal shape
    slice_start = [0] * tensor.shape.rank
    slice_size = tensor.shape.as_list()
    slice_size[-1] -= 1
    return tf.slice(tensor, slice_start, slice_size)

def mask_first_char(tensor):
    batchsize = tensor.shape[0]
    go = tf.zeros([batchsize, 1], tensor.dtype)
    return tf.concat([go, tensor[:, 1:]], axis=1)

def randomly_mask_sampled_maskprob(tensor, max_maskprob):
    """
    Randomly mask values in the tensor, where the masking rate is uniformly sampled from:
        [0, max_maskprob]
    """
    maskprob = tf.random.uniform([], minval=0, maxval=max_maskprob, dtype=tf.dtypes.float32)
    return randomly_mask(tensor, maskprob)

def randomly_mask(tensor, maskprob):
    mask = tf.random.uniform(tensor.shape, minval=0, maxval=1, dtype=tf.dtypes.float32)
    mask = tf.where(tf.less(mask, maskprob), tf.zeros_like(mask), tf.ones_like(mask))
    return tensor * tf.cast(mask, tensor.dtype)

def randomly_sequence_mask(tensor):
    """
    Mask the first N characters of the first P examples in the batch, where N is uniformly sampled
    from range(0, len(example)) for each example, and P is a constant where 0 <= P <= batchsize.
    """
    percent_masked = 0.2 # proportion of batch elements that will have a mask applied
    batchsize, maxlen = tensor.shape
    num_masked = tf.cast(percent_masked * tf.cast(batchsize, tf.float32), tf.int32)
    lengths = tf.random.uniform(shape=[num_masked], minval=0, maxval=maxlen, dtype=tf.int32)
    lengths = tf.concat([lengths, tf.zeros([batchsize - num_masked], dtype=lengths.dtype)], axis=0)
    mask = tf.sequence_mask(lengths, maxlen=maxlen)
    return tf.where(mask, tf.ones_like(tensor), tensor)

def randomly_span_mask(tensor):
    """ Mask a random span of contiguous characters within each example """
    batchsize, maxlen = tensor.shape
    mean_span_len = 8 # Tends to cover about a word or two
    stdv_span_len = 2
    def span_mask(vector): # computes and applies mask to a single row of the batch
        span_len = tf.random.normal(shape=[], mean=mean_span_len, stddev=stdv_span_len)
        span_len = tf.cast(span_len, tf.int32)
        span_len = tf.math.maximum(0, span_len)
        span_len = tf.math.minimum(maxlen, span_len)
        mask = tf.zeros([span_len], dtype=vector.dtype)
        span_start = tf.random.uniform(shape=[], minval=0, maxval=maxlen - span_len, dtype=tf.int32)
        left_pad = tf.ones([span_start], dtype=vector.dtype)
        right_pad = tf.ones([maxlen - (span_start + span_len)], dtype=vector.dtype)
        mask = tf.concat([left_pad, mask, right_pad], axis=0)
        return vector * mask
    percent_masked = 1.0 # proportion of batch elements that will have a mask applied
    num_masked = tf.cast(percent_masked * tf.cast(batchsize, tf.float32), tf.int32)
    unmasked = tensor[:-num_masked]
    masked = tf.map_fn(span_mask, tensor[-num_masked:])
    return tf.concat([unmasked, masked], axis=0)


def string_to_ids(tf_string):
    result = tf.strings.bytes_split(tf_string, 'UTF-8')
    # Decode raw bytes: data is preped to a fixed number of bytes per line so some valid utf-8
    # characters may get split into invalid utf-8 bytes if they lie on the boundary.
    result = tf.io.decode_raw(result, tf.uint8)
    result = tf.cast(result, tf.int32)
    result = tf.squeeze(result, axis=1)
    return result

def ids_to_string(tensor):
    result = tf.strings.unicode_encode(tensor, 'UTF-8', errors='ignore')
    return result

def ids_to_python_string(tensor):
    # Manually convert the ints to char bytes, then to a string. This avoids producing weird
    # characters when a unicode sequence has been broken up.
    result = tf.cast(tensor, tf.uint8).numpy()
    result = [str(bytes(line), 'utf-8', 'replace') for line in result]
    return result

if __name__ == '__main__':
    config = process_config.load_config()
    config['batchsize'] = 10
    lines = file_to_dataset('./traindata.txt', config)
    print(lines)
    lines = iter(lines)
    for batch in range(5):
        print('\n\n\nbatch: {}'.format(batch))
        example = next(lines)
        inputs, targets = example
        inputs = [ids_to_python_string(x) for x in inputs]
        targets = ids_to_python_string(targets)
        for idx in range(config['batchsize']):
            for x in inputs:
                print(x[idx].replace(chr(0), '_'))
                print()
            print(targets[idx])
            print()

            print('---------------------------------------')


import tensorflow as tf

def file_to_dataset(data_fp, batchsize, maxseqlen, maskinputs=True):
    lines = tf.data.TextLineDataset(tf.constant(data_fp))
    lines = lines.map(string_to_ids)
    lines = lines.batch(batchsize, drop_remainder=True)
    # Peek at the first example to get the shape and set it explicitly
    batchsize, seqlen = next(iter(lines)).shape
    lines = lines.map(lambda line: tf.reshape(line, [batchsize, seqlen]))
    # Add a GO byte to the beginning of the sequence.
    lines = lines.map(lambda line: tf.concat([2*tf.ones([batchsize, 1], line.dtype), line], axis=1))
    # Split x,y pair which are offset by one character
    lines = lines.map(lambda line: (line[:, :-1], line[:, 1:]))
    # Split long lines into consecutive batches which can be trained with stateful RNNs
    if seqlen > maxseqlen:
        if seqlen % maxseqlen != 0:
            raise ValueError('The maxseqlen must evenly divide the length of the lines in the input '
                'file, got maxseqlen:{} seqlen:{}'.format(maxseqlen, seqlen))
        nsplits = seqlen // maxseqlen
        lines = lines.map(lambda x,y: (tf.split(x, nsplits, axis=1), tf.split(y, nsplits, axis=1)))
        lines = lines.map(lambda x,y: (tf.reshape(x, [-1, maxseqlen]), tf.reshape(y, [-1, maxseqlen])))
        lines = lines.unbatch()
        lines = lines.batch(batchsize, drop_remainder=True)
    if maskinputs:
        # Randomly mask some of the input values
        lines = lines.map(lambda x,y: (randomly_mask(x), y))
    return lines

def randomly_mask(tensor):
    maskprob = 0.10
    mask = tf.random.uniform(tensor.shape, minval=0, maxval=1, dtype=tf.dtypes.float32)
    mask = tf.where(tf.less(mask, maskprob), tf.zeros_like(mask), tf.ones_like(mask))
    return tensor * tf.cast(mask, tensor.dtype)

def string_to_ids(tf_string):
    result = tf.strings.bytes_split(tf_string, 'UTF-8')
    # Decode raw bytes: data is preped to a fixed number of bytes per line so some valid utf-8
    # characters may get split into invalid utf-8 bytes if they lie on the boundary.
    result = tf.io.decode_raw(result, tf.uint8)
    result = tf.cast(result, tf.int32)
    result = tf.squeeze(result)
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
    batchsize = 8
    maxseqlen = 16
    lines = file_to_dataset('./traindata.txt', batchsize, maxseqlen)
    print(lines)
    lines = iter(lines)
    for batch in range(4):
        print('\nbatch: {}'.format(batch))
        example = next(lines)
        x, y = example
        print(x)
        for line in ids_to_python_string(x):
            print(line.replace(chr(0), '_'))
        print(y)
        for line in ids_to_python_string(y):
            print(line)



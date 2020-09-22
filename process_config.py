import json

def load_config():
    path = './config.json'
    config = json.load(open(path))
    config['numclasses'] = 256 # assume utf-8 bytes
    for block_config in config['blocks']:
        process_block(block_config, config['char_embed_size'])
    compute_total_seqlen(config)
    print_config(config)
    return config

def process_block(block_config, char_embed_size):
    """
    A block config contains configuration info for a compute block in the model. Each block operates
    on a 2D representation of a sequence, the total length of which is numheads * memsize
    """
    block_config['wordsize'] = block_config['numheads'] * char_embed_size # TODO rename to slotsize
    block_config['subseqlen'] = block_config['memsize']
    if 'compress' not in block_config:
        block_config['compress'] = False

def compute_total_seqlen(config):
    """
    The total sequence length (context window) for the model is the longest sub-sequence length
    seen by the blocks, multiplied by the 2 for each time the sequence has been compressed
    """
    multiplier = 1
    for block_config in config['blocks']:
        block_config['subseqlen_compressed'] = multiplier * block_config['subseqlen']
        if 'compress' in block_config and block_config['compress']:
            multiplier *= 2
    total_seqlen = 0
    config['seqlen'] = max([block_config['subseqlen_compressed'] for
            block_config in config['blocks']])

def print_config(config):
    print('CONFIGURATION')
    print('batchsize: {}'.format(config['batchsize']))
    print('context window: {}'.format(config['seqlen']))
    print('char_embed_size: {}'.format(config['char_embed_size']))
    print('dropout: {}'.format(config['dropout']))
    print('train_char_embeds: {}'.format(config['train_char_embeds']))
    print('\nBLOCKS\n')
    for block in config['blocks']:
        print('numheads: {}'.format(block['numheads']))
        print('memsize: {}'.format(block['memsize']))
        print('wordsize: {}'.format(block['wordsize']))
        print('kernelsize: {}'.format(block['kernelsize']))
        print('subseqlen (compressed) : {} ({})'.format(block['subseqlen'],
                block['subseqlen_compressed']))
        print('compress_output: {}'.format(block['compress']))
        print('numlayers: {}'.format(block['numlayers']))
        print('trainable: {}'.format(block['trainable']))
        print('-----------------------------------------')



if '__main__' == __name__:
    load_config()

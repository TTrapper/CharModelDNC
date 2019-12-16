import argparse
import json

import numpy as np
import tensorflow as tf

import model_def

parser = argparse.ArgumentParser()
parser.add_argument('--old_model', type=str, required=True, help='path to saved model')
parser.add_argument('--old_numheads', type=int, default=1, help='number of heads in the old model')
parser.add_argument('--new_config', type=str, default='./config.json',
        help='path to the desired new configuration (should be same as the old_model except for the'
             'value of numheads, which must be a multiple of the number of heads in the old_model')

def add_attention_heads(old_model_path, old_numheads, new_config):
    """
    Update the number of attention heads by loading an old model and copying it's weights into the
    new model. The result should be a new model with the read/write weights of the old model copied
    to fill the desired number of heads. Until trained, each model should produce identical results.
    """
    # Create two models with identical architectures
    config = json.load(open(new_config))
    new_numheads = config['numheads']
    assert new_numheads % old_numheads == 0
    old_model = model_def.make_model(1, config['numlayers'], config['layersize'], config['memsize'],
            old_numheads, return_state=False)
    old_model.load_weights(old_model_path)
    new_model = model_def.make_model(1, config['numlayers'], config['layersize'], config['memsize'],
            new_numheads, return_state=False)
    assert len(old_model.weights) == len(new_model.weights)
    for old_weight, new_weight in zip(old_model.weights, new_model.weights):
        if old_weight.shape == new_weight.shape:
            new_weight = new_weight.assign(old_weight)
        else:
            old_shape = old_weight.shape[-1]
            new_shape = new_weight.shape[-1]
            assert new_shape % old_shape == 0
            if old_weight.shape.rank == 2 and new_weight.shape.rank == 2:
                tileshape = [1, 1, new_shape // old_shape]
            elif old_weight.shape.rank == 1 and new_weight.shape.rank == 1:
                tileshape = [1, new_shape // old_shape]
            else:
                raise ValueError('Found weights of unexpected rank')
            tiled_weight = tf.tile(tf.expand_dims(old_weight, -1), tileshape)
            new_weight = new_weight.assign(tf.reshape(tiled_weight, new_weight.shape))
    return new_model, new_numheads

if __name__ == '__main__':
    args = parser.parse_args()
    model, new_numheads = add_attention_heads(args.old_model, args.old_numheads, args.new_config)
    model.save('{}_newheads-{}'.format(new_numheads, args.old_model), save_format='h5',
            overwrite=True)

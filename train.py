import argparse
import os

import numpy as np
import tensorflow as tf

import data_pipe
import model_def
import process_config

parser = argparse.ArgumentParser()
parser.add_argument('--restore', action='store_true')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--traindata', default='./traindata.txt', help='Path to a text file')
parser.add_argument('--validdata', default='./validdata.txt', help='Path to a text file')
parser.add_argument('--evaldata', default='./evaldata.txt', help='Path to a text file')


def parseargs(parser):
    args = parser.parse_args()
    if not args.eval and not os.path.isfile(args.traindata):
        raise FileNotFoundError('No training data file found at {}'.format(args.traindata))
    if not args.eval and not os.path.isfile(args.validdata):
        print('No validation data file found at: {}. Continuing without it.'.format(args.validdata))
        args.validdata = None
    if args.eval and not os.path.isfile(args.evaldata):
        raise FileNotFoundError('No training data file found at {}'.format(args.traindata))
    return args

def setup(restore, datapath, validdata, evalmode):
    learn_rate = 1e-4
    config = process_config.load_config()
    model = model_def.make_model(config)
    dataset = data_pipe.file_to_dataset(datapath, config, maskinputs=not evalmode)
    valid_dataset = None
    if validdata:
        valid_dataset = data_pipe.file_to_dataset(validdata, config, maskinputs=False)
    if restore:
        model.load_weights('./model.h5', by_name=True, skip_mismatch=True)
    optimizer = tf.keras.optimizers.Adam(learn_rate)
    model.compile(optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.summary()
    return dataset, valid_dataset, model

def train(model, dataset, valid_dataset):
    def save_fn(batch, logs):
        if batch % 1000 == 200:
            model.save('./model.h5', save_format='h5', overwrite=True, include_optimizer=False)
    save_callback = tf.keras.callbacks.LambdaCallback(on_batch_end=save_fn)
    callbacks = [save_callback]
    model.fit(dataset, epochs=200, verbose=1, steps_per_epoch=None, use_multiprocessing=True,
            validation_data=valid_dataset, validation_steps=None, callbacks=callbacks)

def evaluate(model, dataset):
    model.evaluate(dataset)

if '__main__' == __name__:
    args = parseargs(parser)
    datapath = args.evaldata if args.eval else args.traindata
    dataset, valid_dataset, model = setup(args.restore, datapath, args.validdata, args.eval)
    if args.eval:
        evaluate(model, dataset)
    else:
        train(model, dataset, valid_dataset)

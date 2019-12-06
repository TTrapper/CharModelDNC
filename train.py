import argparse

import numpy as np
import tensorflow as tf

import data_pipe
import model_def

parser = argparse.ArgumentParser()
parser.add_argument('--restore', action='store_true')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--data_fp', default='./traindata.txt',
    help='Path to a text file with a fixed number of bytes per line (generate with prepare_data.py')

def parseargs(parser):
    args = parser.parse_args()
    return args

def setup(restore, data_fp):
    learn_rate = 1e-4
    dataset = data_pipe.file_to_dataset(data_fp)
    batchsize = dataset.element_spec[0].shape[0]
    if restore:
        model = model_def.make_model(batchsize)
        model.load_weights('./model.h5')
    else:
        model = model_def.make_model(batchsize)
    optimizer = tf.keras.optimizers.Adam(learn_rate)
    model.compile(optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.summary()
    return dataset, model

def train(model, dataset):
    def inference_fn(batch, logs):
        if batch % 2000 == 200:
            model.save('./model.h5', save_format='h5', overwrite=True, include_optimizer=True)
            for softmax_temp in [1e-16, 0.5, 0.75]:
                model_def.run_inference(model, 'she', 512, softmax_temp)
    inference_callback = tf.keras.callbacks.LambdaCallback(on_batch_end=inference_fn)
    callbacks = [inference_callback]
    model.fit(dataset, epochs=200, verbose=1, steps_per_epoch=None, use_multiprocessing=True,
        callbacks=callbacks)

def evaluate(model, dataset):
    model.evaluate(dataset)

if '__main__' == __name__:
    args = parseargs(parser)
    dataset, model = setup(args.restore, args.data_fp)
    if args.eval:
        evaluate(model, dataset)
    else:
        train(model, dataset)

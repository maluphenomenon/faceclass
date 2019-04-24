import os
import sys
import math
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib
matplotlib.use('Agg')

from models import VAE, DCGAN
from datasets import load_data

models = {
    'vae': VAE,
    'dcgan': DCGAN,
}

def main():

    model1 = 'vae'
    model2 = 'dcgan'
    dataset1 = ''
    epoch = 200
    batchsize = 50
    output1 = 'output'
    zdims = 256



    # Make output direcotiry if not exists
    if not os.path.isdir(output1):
        os.mkdir(output1)


    datasets = load_data(dataset1)

    # Construct model
    if model1 not in models:
        raise Exception('Unknown model:', model1)

    model = models[model1](
        input_shape=datasets.shape[1:],
        z_dims=zdims,
        output=output1
    )



    # Training loop
    datasets = datasets.images * 2.0 - 1.0
    samples = np.random.normal(size=(100, zdims)).astype(np.float32)
    model.main_loop(datasets, samples,
        epochs= epoch,
        batchsize= batchsize,
        reporter=['loss', 'g_loss', 'd_loss', 'g_acc', 'd_acc'])

if __name__ == '__main__':
    main()

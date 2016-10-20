#!/usr/bin/env python

# Author : Thibault Barbie

import argparse
import chainer
from chainer import training, datasets, iterators
from chainer.training import extensions
from net import *
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=10,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    model = Regressor(Autoencoder(args.unit))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test=datasets.get_mnist(withlabel=False)
    Xtr=train
    Xte=test

    train = zip(Xtr,Xtr)
    test = zip(Xte,Xte)
    train_iter = iterators.SerialIterator(train, batch_size=args.batchsize, shuffle=True)
    test_iter = iterators.SerialIterator(test, batch_size=args.batchsize, repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    if args.gpu>=0:
        model.to_cpu()

    # Plot the results
    plt.subplot(321)
    plt.imshow(Xte[0].reshape(28,28),interpolation='none',cmap='Greys_r')

    a=model.predictor(Xte[1:2]).data.reshape(28,28)
    plt.subplot(322)
    plt.imshow(a,interpolation='none',cmap='Greys_r')


    plt.subplot(323)
    plt.imshow(Xte[1].reshape(28,28),interpolation='none',cmap='Greys_r')

    a=model.predictor(Xte[1:2]).data.reshape(28,28)
    plt.subplot(324)
    plt.imshow(a,interpolation='none',cmap='Greys_r')


    plt.subplot(325)
    plt.imshow(Xte[2].reshape(28,28),interpolation='none',cmap='Greys_r')

    a=model.predictor(Xte[2:3]).data.reshape(28,28)
    plt.subplot(326)
    plt.imshow(a,interpolation='none',cmap='Greys_r')

    plt.show()

if __name__ == '__main__':
    main()


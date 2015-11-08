#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
import gzip
import cPickle

def main(model='mlp', num_epochs=1000):



    f = file('pickledProstatesNormalized.pkl','rb')

    dataX, dataY = cPickle.load(f)

    folds = 10
    foldsize = 160


    rocf = file('rocFormattedDropout', 'wb')

    rocf.write('class\tprediction\n')

    for fold in xrange(0, folds):

        print("fold %i" %(fold))

        X_train = theano.shared(np.asarray(np.concatenate((dataX[0: fold * foldsize], dataX[(fold+1) * foldsize: 1600])),
                                               dtype=theano.config.floatX),
                                 borrow=True)
        y_train = theano.shared(np.asarray(np.concatenate((dataY[0: fold * foldsize], dataY[(fold+1) * foldsize: 1600])),
                                               dtype=theano.config.floatX),
                                 borrow=True)

        X_test = theano.shared(np.asarray(dataX[fold * foldsize: (fold+1) * foldsize],
                                               dtype=theano.config.floatX),
                                 borrow=True)

        y_test = theano.shared(np.asarray(dataY[fold * foldsize: (fold+1) * foldsize],
                                               dtype=theano.config.floatX),
                                 borrow=True)

        X_val = X_test
        y_val = y_test

        '''
        X_train = data[0][0].astype(np.float32)
        X_val = data[1][0].astype(np.float32)
        X_test = data[2][0].astype(np.float32)

        y_train = data[0][1].astype(np.float32)
        y_val = data[1][1].astype(np.float32)
        y_test = data[2][1].astype(np.float32)
        '''







        input_var = T.matrix('inputs')
        target_var = T.fvector('targets')

        batchsize = 80

        l_in = lasagne.layers.InputLayer(shape=(None, 131),
                                         input_var=input_var)

        # Apply 20% dropout to the input data:
        l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.1)

        # Add a fully-connected layer of 800 units, using the linear rectifier, and
        # initializing weights with Glorot's scheme (which is the default anyway):
        l_hid1 = lasagne.layers.DenseLayer(
                l_in_drop, num_units=66,
                nonlinearity=lasagne.nonlinearities.sigmoid,
                W=lasagne.init.GlorotUniform())

        # We'll now add dropout of 50%:
        l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.1)

        # Another 800-unit layer:
        l_hid2 = lasagne.layers.DenseLayer(
                l_hid1_drop, num_units=33,
                nonlinearity=lasagne.nonlinearities.sigmoid)

        # 50% dropout again:
        l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.1)

        l_hid3 = lasagne.layers.DenseLayer(l_hid2_drop,17,nonlinearity=lasagne.nonlinearities.sigmoid)

        l_hid3_drop = lasagne.layers.DropoutLayer(l_hid3, p=0.1)

        # Finally, we'll add the fully-connected output layer, of 10 softmax units:
        network = lasagne.layers.DenseLayer(
                l_hid3_drop, num_units=1,
                nonlinearity=lasagne.nonlinearities.sigmoid)



        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        prediction = lasagne.layers.get_output(network).flatten()
        loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
        loss = loss.mean()
        # We could add some weight decay as well here, see lasagne.regularization.

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
                loss, params, learning_rate=0.01, momentum=0.9)

        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout layers.
        test_prediction = lasagne.layers.get_output(network, deterministic=True).flatten()
        test_loss = lasagne.objectives.binary_crossentropy(test_prediction,
                                                                target_var)
        test_loss = test_loss.mean()
        # As a bonus, also create an expression for the classification accuracy:
        test_acc = T.mean(T.eq(T.round(test_prediction), target_var),
                          dtype=theano.config.floatX)

        rocfn = theano.function(inputs=[],outputs=[prediction, target_var],givens={input_var: X_test, target_var: y_test})


        index = theano.tensor.lscalar('index')

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([index], updates=updates,
                                   givens={
                                    input_var: X_train[
                                         index * batchsize: (index + 1) * batchsize
                                    ],
                                    target_var: y_train[
                                        index * batchsize: (index + 1) * batchsize
                                    ]
                                     })

        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([index], [test_loss, test_acc], givens={
                                    input_var: X_test[
                                         index * batchsize: (index + 1) * batchsize
                                    ],
                                    target_var: y_test[
                                        index * batchsize: (index + 1) * batchsize
                                    ]
                                     })

        prediction_fn = theano.function([input_var], prediction.flatten())
        loss_fn = theano.function([input_var, target_var], lasagne.objectives.binary_crossentropy(prediction,target_var))

        # Finally, launch the training loop.
        print("Starting training...")
        # We iterate over epochs:
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in xrange(0, 18):
                train_fn(batch)
                train_batches += 1

        finaloutputs = rocfn()


        targets = y_test.get_value()
        for i in xrange(0,160):
            target = targets[i]
            prediction = finaloutputs[0][i]
            if target > 0.5:
                classtring = '+1'
            else:
                classtring = '-1'

            rocf.write(classtring + '\t%f\n'%(prediction))

        # After training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in xrange(0,2):
            err, acc = val_fn(batch)
            test_err += err
            test_acc += acc
            test_batches += 1
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))

    rocf.close()

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)

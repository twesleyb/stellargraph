#!/usr/bin/env python
'''
title: Link Prediction in homogenous graphs with GraphSAGE
description: tutorial adapted from stellargraph/demos
author: tyler w bradshaw
'''

## input ----------------------------------------------------------------------
# * Cora dataset from stellargraph.datasets


## output ---------------------------------------------------------------------


## options --------------------------------------------------------------------
epochs = 20
batch_size = 20


## imports --------------------------------------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # suppress tf msgs

import stellargraph as sg

from stellargraph import globalvar
from stellargraph import datasets

from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator

from stellargraph.layer import GraphSAGE
from stellargraph.layer import HinSAGE
from stellargraph.layer import link_classification

from tensorflow import keras

from sklearn import preprocessing
from sklearn import feature_extraction
from sklearn import model_selection


## load ------------------------------------------------------------------------
# load the Cora network

dataset = datasets.Cora()
[g, _] = dataset.load(subject_as_feature=True)


## generate test and train datasets  --------------------------------------------

# initialize EdgeSplitter class
# train_test_split: generate testing dataset
# take the input graph, and randomly remove positive and negative edges
# ... repeat ...
# initialize EdgeSplitter class
# train_test_split: generate training dataset
# take the test graph, and randomly remove positive and negative edges

print("\nGenerating testing dataset:")
edge_splitter_test = EdgeSplitter(g)
[g_test, test_ids, test_labels] = \
        edge_splitter_test.train_test_split(
                p=0.1, # fraction of edges to remove
                method="global", # sample globally
                keep_connected=True) # remove edges, but don't break-up graph

# Repeat
print("\nGenerating training dataset:")
edge_splitter_train = EdgeSplitter(g_test) # NOTE: input is g_test!
[g_train, train_ids, train_labels] = \
        edge_splitter_train.train_test_split(
                p=0.1,
                method="global",
                keep_connected=True)

# NOTE: the number of edges sampled (removed) is less as g_test is used as input!


## create generators for testing and training ---------------------------------
# use the flow method supplied with (node) train_ids and train_labels (numeric
# targets) to create a generator for training

test_gen = GraphSAGELinkGenerator(
        g_test,
        batch_size,
        num_samples=[20,10])

test_flow = test_gen.flow(test_ids, test_labels, shuffle=False)

# repeat

train_gen = GraphSAGELinkGenerator(
        g_train,
        batch_size,
        num_samples=[20,10])

train_flow = train_gen.flow(train_ids, train_labels, shuffle=True)


## build the GraphSAGE model --------------------------------------------------
# build the model and expose input and output sockets of graphsage model for link prediction
# call the in_out_tensors()

# use the training generator object:
graphsage = GraphSAGE(
        layer_sizes = [20,20],
        generator = train_gen,
        bias= True,
        dropout=0.3)

[x_inp, x_out] = graphsage.in_out_tensors()

# create ouptut (prediction) layer, this takes the output of GraphSAGE, and
# applies a binary operator to them to produce the corresponding link embedding
# and passes it through a dense layer
prediction = link_classification(
        output_dim=1,
        output_act="relu",
        edge_embedding_method="ip")(x_out)

# combine input and output layers into a keras.Model
model = keras.Model(inputs=x_inp, outputs=prediction)

# compile the model
model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3), # the optimizer to be used (tf.keras.optimizers)
        loss=keras.losses.binary_crossentropy, # objective loss function (tf.keras.losses)
        metrics=["accuracy"]) # metrics to be evaluated during training and testing (tf.keras.metrics)


## evaulate untrained models --------------------------------------------------

# evaluate the initial (untrained) model on the training data:
init_train_metrics = model.evaluate(train_flow)
print("\ntrain set metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_train_metrics):
    print("\t{}: {:0.4f}".format(name, val))


# evaulate the initial (untrained) model on the testing data:
init_test_metrics = model.evaluate(test_flow)
print("\ntest set metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_test_metrics):
    print("\t{}: {:0.4f}".format(name, val))


## training -------------------------------------------------------------------

# train the model:
history = model.fit(
        train_flow,
        epochs=epochs,
        validation_data=test_flow,
        verbose=2)


## final evaluation ------------------------------------------------------------
# evaluate the trained model on the training and testing dataset

train_metrics = model.evaluate(train_flow)
print("\ntrain set metrics of the trained model:")
for name, val in zip(model.metrics_names, train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

test_metrics = model.evaluate(test_flow)
print("\ntest set metrics of the trained model:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

# done!

#!/usr/bin/env python
# coding: utf-8

# # Link prediction with GCN

# In this example, we use our implementation of the
# [GCN](https://arxiv.org/abs/1609.02907) algorithm to build a model that
# predicts citation links in the Cora dataset (see below). The problem is
# treated as a supervised link prediction problem on a homogeneous citation
# network with nodes representing papers (with attributes such as binary keyword
# indicators and categorical subject) and links corresponding to paper-paper
# citations. 
 
# To address this problem, we build a model with the following architecture.
# First we build a two-layer GCN model that takes labeled node pairs
# (`citing-paper` -> `cited-paper`)  corresponding to possible citation links,
# and outputs a pair of node embeddings for the `citing-paper` and `cited-paper`
# nodes of the pair. These embeddings are then fed into a link classification
# layer, which first applies a binary operator to those node embeddings (e.g.,
# concatenating them) to construct the embedding of the potential link. Thus
# obtained link embeddings are passed through the dense link classification
# layer to obtain link predictions - probability for these candidate links to
# actually exist in the network. The entire model is trained end-to-end by
# minimizing the loss function of choice (e.g., binary cross-entropy between
# predicted link probabilities and true link labels, with true/false citation
# links having labels 1/0) using stochastic gradient descent (SGD) updates of
# the model parameters, with minibatches of 'training' links fed into the model.

import sys

# verify that we're using the correct version of StellarGraph for this notebook
#error: tensorflow.python.keras.utils.generic_utils' has no attribute 'populate_dict_with_module_objects 
#solution: pip install tensorflow --upgrade --force-reinstall
## you may be warned about GPU
#import stellargraph as sg
#try:
#    sg.utils.validate_notebook_version("1.3.0b")
#except AttributeError:
#    raise ValueError(
#        f"This notebook requires StellarGraph version 1.3.0b, but a different version {sg.__version__} is installed.  Please see <https://github.com/stellargraph/stellargraph/issues/1172>."
#    ) from None

# imports ---------------------------------------------------------------------

# ignore tf INFO and WARNING messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import FullBatchLinkGenerator
from stellargraph.layer import GCN, LinkEmbedding

from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection

from stellargraph import globalvar
from stellargraph import datasets

#from IPython.display import display, HTML
#get_ipython().run_line_magic('matplotlib', 'inline')

## Loading the CORA network data

# (See [the "Loading from Pandas" demo](../basics/loading-pandas.ipynb) for
# details on how data can be loaded.)

dataset = datasets.Cora()
#display(HTML(dataset.description))
G, _ = dataset.load(subject_as_feature=True)

print(G.info()) # 2,708 nodes and 5,429 edges

# We aim to train a link prediction model, hence we need to prepare the train
# and test sets of links and the corresponding graphs with those links removed.
# 
# We are going to split our input graph into a train and test graphs using the
# EdgeSplitter class in `stellargraph.data`. We will use the train graph for
# training the model (a binary classifier that, given two nodes, predicts
# whether a link between these two nodes should exist or not) and the test graph
# for evaluating the model's performance on hold out data.  Each of these graphs
# will have the same number of nodes as the input graph, but the number of links
# will differ (be reduced) as some of the links will be removed during each
# split and used as the positive samples for training/testing the link
# prediction classifier.

# From the original graph G, extract a randomly sampled subset of test edges
# (true and false citation links) and the reduced graph G_test with the positive
# test edges removed:

# Define an edge splitter on the original graph G:
edge_splitter_test = EdgeSplitter(G)

# Random sample = 10% all links Randomly sample a fraction p=0.1 of all positive
# links, and same number of negative links, from G, and obtain the reduced graph
# G_test with the sampled links removed:
G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global", keep_connected=True
)

# The reduced graph G_test, together with the test ground truth set of links
# (edge_ids_test, edge_labels_test), will be used for testing the model.
# 
# Now repeat this procedure to obtain the training data for the model. From the
# reduced graph G_test, extract a randomly sampled subset of train edges (true
# and false citation links) and the reduced graph G_train with the positive
# train edges removed:

# Define an edge splitter on the reduced graph G_test:
edge_splitter_train = EdgeSplitter(G_test)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
# reduced graph G_train with the sampled links removed:
G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
    p=0.1, method="global", keep_connected=True
)

# G_train, together with the train ground truth set of links (edge_ids_train,
# edge_labels_train), will be used for training the model.

# ## Creating the GCN link model

# Next, we create the link generators for the train and test link examples to
# the model. The link generators take the pairs of nodes (`citing-paper`,
# `cited-paper`) that are given in the `.flow` method to the Keras model,
# together with the corresponding binary labels indicating whether those pairs
# represent true or false links.
# 
# The number of epochs for training the model:
epochs = 50

# For training we create a generator on the `G_train` graph, and make an
# iterator over the training links using the generator's `flow()` method:

train_gen = sg.mapper.FullBatchLinkGenerator(G_train, method="gcn")
train_flow = train_gen.flow(edge_ids_train, edge_labels_train)

test_gen = sg.mapper.FullBatchLinkGenerator(G_test, method="gcn")
test_flow = train_gen.flow(edge_ids_test, edge_labels_test)

# Now we can specify our machine learning model, we need a few more parameters
# for this:
#  * the `layer_sizes` is a list of hidden feature sizes of each layer in the
#  model. In this example we use two GCN layers with 16-dimensional hidden node
#  features at each layer.  * `activations` is a list of activations applied to
#  each layer's output * `dropout=0.3` specifies a 30% dropout at each layer. 

# We create a GCN model as follows:

gcn = GCN(
    layer_sizes=[16, 16], activations=["relu", "relu"], generator=train_gen, dropout=0.3
)

# To create a Keras model we now expose the input and output tensors of the GCN
# model for link prediction, via the `GCN.in_out_tensors` method:

x_inp, x_out = gcn.in_out_tensors()

# Final link classification layer that takes a pair of node embeddings produced
# by the GCN model, applies a binary operator to them to produce the
# corresponding link embedding (`ip` for inner product; other options for the
# binary operator can be seen by running a cell with `?LinkEmbedding` in it),
# and passes it through a dense layer:
prediction = LinkEmbedding(activation="relu", method="ip")(x_out)

# The predictions need to be reshaped from `(X, 1)` to `(X,)` to match the shape of the targets we have supplied above.
prediction = keras.layers.Reshape((-1,))(prediction)

# Stack the GCN and prediction layers into a Keras model, and specify the loss
model = keras.Model(inputs=x_inp, outputs=prediction)

## using the ADAM optimizer!
model.compile(
    optimizer=keras.optimizers.Adam(lr=0.01),
    loss=keras.losses.binary_crossentropy,
    # not just "acc" due to https://github.com/tensorflow/tensorflow/issues/41361
    metrics=["binary_accuracy"],
)

init_train_metrics = model.evaluate(train_flow)
init_test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

# Train the model:
history = model.fit(
    train_flow, epochs=epochs, validation_data=test_flow, verbose=2, shuffle=False
)

# Plot the training history:
#print(sg.utils.plot_history(history))

# Evaluate the trained model on test citation links:
train_metrics = model.evaluate(train_flow)
test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

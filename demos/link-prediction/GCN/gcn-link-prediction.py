#!/usr/bin/env python
'''
link prediction with GCN
'''

## imports

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

## Loading the CORA network data
dataset = datasets.Cora()
G, _ = dataset.load(subject_as_feature=True)

print(G.info()) # 2,708 nodes and 5,429 edges


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

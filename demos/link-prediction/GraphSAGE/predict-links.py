#!/usr/bin/env python
'''
link prediction with graphsage
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import stellargraph as sg

from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator

from stellargraph.layer import GraphSAGE, HinSAGE, link_classification

from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection

from stellargraph import globalvar
from stellargraph import datasets


# load the cora network data
dataset = datasets.Cora()
g, _ = dataset.load(subject_as_feature=True)

print(g.info())


# we aim to train a link prediction model, hence we need to prepare the
# train and test sets of links and the corresponding graphs with those
# links removed.
#
# we are going to split our input graph into a train and test graphs
# using the EdgeSplitter class in `stellargraph.data`. we will use the
# train graph for training the model (a binary classifier that, given
# two nodes, predicts whether a link between these two nodes should
# exist or not) and
# the test graph for evaluating the model's performance on hold out data.
# each of these graphs will have the same number of nodes as the input
# graph, but the number of links will differ (be reduced) as some of the
# links will be removed during each split and used as the positive samples
# for training/testing the link prediction classifier.

# from the original graph g, extract a randomly sampled subset of test
# edges (true and false citation links) and the reduced graph g_test
# with the positive test edges removed:

# define an edge splitter on the original graph g:
edge_splitter_test = EdgeSplitter(g)

# randomly sample a fraction p=0.1 of all positive links,
# and same number of negative links, from g,
# and obtain the reduced graph g_test with the sampled links removed:
g_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split( p=0.1, method="global",
        keep_connected=True)

# the reduced graph g_test, together with the test ground truth set of
# links (edge_ids_test, edge_labels_test), will be used for testing the
# model.
#
# now repeat this procedure to obtain the training data for the model
# from the reduced graph g_test, extract a randomly sampled subset of
# train edges (true and false citation links) and the reduced graph
# g_train with the positive train edges removed:

# define an edge splitter on the reduced graph g_test:
edge_splitter_train = EdgeSplitter(g_test)

# randomly sample a fraction p=0.1 of all positive links,
# and same number of negative links, from g_test, and obtain the reduced
# graph g_train with the sampled links removed:
g_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split( p=0.1, method="global",
        keep_connected=True)


# g_train, together with the train ground truth set of links
# (edge_ids_train, edge_labels_train), will be used for training the
# model.
#
# summary of g_train and g_test - note that they have the same set of
# nodes, only differing in their edge sets:
print(g_train.info())
print(g_test.info())

# next, we create the link generators for sampling and
# streaming train and test link examples to the model
# the link generators essentially "map" pairs of nodes
# to the input of GraphSAGE:
# they take minibatches of node pairs,
# sample 2-hop subgraphs with pairs of nodes head nodes extracted from those pairs, and feed them,
# together with the corresponding binary labels indicating
# whether those pairs represent T or F citation links,
# to the input layer of the GraphSAGE model,
# for sgd updates of the model parameters.
#
# specify the minibatch size (number of node pairs per
# minibatch) and the number of epochs for training the model:
batch_size = 20; epochs = 20

# specify the sizes of 1- and 2-hop neighbour samples for graphsage
# NOTE: that the length of `num_samples` list defines the number of
# layers/iterations in the graphsage model.
# in this example, we are defining a 2-layer graphsage model:
num_samples = [20, 10] # two layers

# for training we create a generator on the `g_train` graph, and make an
# iterator over the training links using the generator's `flow()` method.
# the `shuffle=true` argument is given to the `flow` method to improve
# training.
train_gen = GraphSAGELinkGenerator(g_train, batch_size, num_samples)
train_flow = train_gen.flow(edge_ids_train, edge_labels_train,
        shuffle=True)

# at test time we use the `g_test` graph and don't specify the `shuffle`
# argument (it defaults to `false`).
test_gen = GraphSAGELinkGenerator(g_test, batch_size, num_samples)
test_flow = test_gen.flow(edge_ids_test, edge_labels_test)

# build the model:
# a 2-layer graphsage model acting as node representation
# learner, with a link classification layer on concatenated citing paper
#
# graphsage part of the model, with hidden layer sizes of 50 for both
# graphsage layers, a bias term, and no dropout.
# (dropout can be switched on by specifying a positive dropout rate,
# 0 < dropout < 1) note that the length of layer_sizes list must be
# equal to the length of `num_samples`, as `len(num_samples)` defines
# the number of hops (layers) in the GraphSAGE model.
layer_sizes = [20, 20]

graphsage = GraphSAGE(layer_sizes=layer_sizes,
        generator=train_gen, bias=True, dropout=0.3)

# build the model and expose input and output sockets of graphsage model for link prediction
x_inp, x_out = graphsage.in_out_tensors()

# final link classification layer that takes a pair of node embeddings
# produced by graphsage, applies a binary operator to them to produce the
# corresponding link embedding and passes it through a dense layer:
prediction = link_classification( output_dim=1, output_act="relu",
        edge_embedding_method="ip")(x_out)

# stack the graphsage and prediction layers into a keras model,
# and specify the loss
model = keras.Model(inputs=x_inp, outputs=prediction)
model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.binary_crossentropy, metrics=["acc"])

# evaluate the initial (untrained) model on the train and test set:
init_train_metrics = model.evaluate(train_flow)

print("\ntrain set metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

init_test_metrics = model.evaluate(test_flow)

print("\ntest set metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

# train the model:
history = model.fit(
        train_flow,
        epochs=epochs,
        validation_data=test_flow,
        verbose=2)

# evaluate the trained model on test citation links:
train_metrics = model.evaluate(train_flow)

print("\ntrain set metrics of the trained model:")
for name, val in zip(model.metrics_names, train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

test_metrics = model.evaluate(test_flow)

print("\ntest set metrics of the trained model:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

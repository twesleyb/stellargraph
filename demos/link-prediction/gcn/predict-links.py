#!/usr/bin/env python
# coding: utf-8

# title: link prediction with GCN
# author: Tyler W Bradshaw
# description: apply GCN to learn links between proteins
# reference: adapted from stellargraph demo

## options
epochs = 50 # number of training epochs

## imports

# ignore tf INFO and WARNING messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd

import networkx as nx

import stellargraph as sg

from stellargraph import globalvar
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import FullBatchLinkGenerator
from stellargraph.layer import GCN, LinkEmbedding

from tensorflow import keras

from sklearn import preprocessing
from sklearn import feature_extraction
from sklearn import model_selection


## load the data

## Loading the CORA network data
dataset = sg.datasets.Cora()
cora, _ = dataset.load(subject_as_feature=True)
print(cora.info())

# nodes: papers
# edges: paperA <cites> paperB
# node features: 0 or 1 indicating absence/presence in word vec (1,433)

df = pd.read_csv('mus_hitpredict_ppis.csv',
        header=0, index_col=0)

# collect edges as a list of tuples
# nodes are entrez IDs (unique gene identifer)
edge_tuples = list(zip(df['osEntrezA'],
    df['osEntrezB']))

# create networkx graph and add edges
g = nx.Graph()

for e in edge_tuples:
    g.add_edge(*e)

# coerce to StellarGraph
G = sg.StellarGraph.from_networkx(g)

print(G.info())

import sys
sys.exit()

# Define an edge splitter on the original graph G:
edge_splitter_test = EdgeSplitter(G)

# generate test and train datasets
# randomly sample and remove edges
G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global", keep_connected=True
)

# Define an edge splitter on the reduced graph G_test:
edge_splitter_train = EdgeSplitter(G_test)

# reduced graph G_train with the sampled links removed:
G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
    p=0.1, method="global", keep_connected=True
)

## Creating the GCN link model

train_gen = FullBatchLinkGenerator(G_train,
        method="gcn")
train_flow = train_gen.flow(edge_ids_train,
        edge_labels_train)

test_gen = FullBatchLinkGenerator(G_test, method="gcn")
test_flow = train_gen.flow(edge_ids_test, edge_labels_test)

# We create a GCN model as follows:
gcn = GCN(
    layer_sizes=[16, 16], activations=["relu", "relu"], generator=train_gen, dropout=0.3
)

# expose input and output tensors of the GCN for prediction
x_inp, x_out = sg.gcn.in_out_tensors()

# generate link embedding
prediction = LinkEmbedding(activation="relu", method="ip")(x_out)

# reshape predictions to match target shape from above
prediction = keras.layers.Reshape((-1,))(prediction)

# Stack the GCN and prediction layers into a Keras model, and specify the loss
model = keras.Model(inputs=x_inp, outputs=prediction)

## compile model with ADAM optimizer
# NOTE: not just "acc" due to :
# https://github.com/tensorflow/tensorflow/issues/41361
model.compile(
    optimizer=keras.optimizers.Adam(lr=0.01),
    loss=keras.losses.binary_crossentropy,
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
    train_flow,
    epochs=epochs,
    validation_data=test_flow,
    verbose=2,
    shuffle=False
)

# Plot the training history:
#print(sg.utils.plot_history(history))
# FIXME: how to extract the relevant progress data?

# Evaluate the trained model on test citation links:
train_metrics = model.evaluate(train_flow)
test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

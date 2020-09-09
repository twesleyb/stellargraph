#!/usr/bin/env python
'''
Link Prediction in homogenous graphs with GraphSAGE
'''

## input
# * Cora dataset from stellargraph.datasets

## output

## options
epochs = 20
batch_size = 20

## imports
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


## load the cora network data
dataset = datasets.Cora()
[g, _] = dataset.load(subject_as_feature=True)

#print(_) # _ pandas Series descibing node (paper) subject-classes

# Cora dataset:
'''
 Cora: https://linqs.soe.ucsc.edu/data
The Cora dataset consists of 2,708 scientific publications classified into one
of seven classes. The citation network consists of 5429 links. Each publication
in the dataset is described by a 0/1-valued word vector indicating the
absence/presence of the corresponding word from the dictionary. The dictionary
consists of 1433 unique words.
'''

# datasets.load
'''
Help on method load in module stellargraph.datasets.datasets:
>>> load(directed=False,
        largest_connected_component_only=False,
        subject_as_feature=False,
        edge_weights=None,
        str_node_ids=False) m

    Returns: A tuple where the first element is the :class:`.StellarGraph`
    object (or :class:`.StellarDiGraph`, if ``directed == True``) with the
    nodes, node feature vectors and edges, and the second element is a pandas
    Series of the node subject class labels.
'''

#print(g.info())

'''
Name: subject, Length: 2708, dtype: object
StellarGraph: Undirected multigraph
 Nodes: 2708, Edges: 5429

 Node types:
  paper: [2708]
    Features: float32 vector, length 1440
    Edge types: paper-cites->paper

 Edge types:
    paper-cites->paper: [5429]
        Weights: all 1 (default)
        Features: none
'''

# EdgeSplitter class
'''
class EdgeSplitter(builtins.object)
EdgeSplitter(g, g_master=None)

Class for generating training and test data for link prediction in graphs.

 >>> train_test_split(
         self,
         p=0.5,
         method='global',
         probs=None,
         keep_connected=False,
         edge_label=None,
         edge_attribute_label=None,
         edge_attribute_threshold=None,
         attribute_is_datetime=None,
         seed=None)
Generates positive and negative edges and a graph that has the same nodes as
the original but the positive edges removed.

Args:
method (str): either 'global' or 'local' If 'local' then the first nodes is
sampled from all nodes in the graph, but the second node is chosen to be from
the former's local neighbourhood.

probs (list): list The probabilities for sampling a node that is k-hops from
the source node, e.g., [0.25, 0.75] means that there is a 0.25 probability that
the target node will be 1-hope away from the source node and 0.75 that it will
be 2 hops away from the source node. This only affects sampling of negative
edges if method is set to 'local'.

keep_connected (bool): If True then when positive edges are removed care is
taken that the reduced graph remains connected. If False, positive edges are
removed without guaranteeing the connectivity of the reduced graph.

edge_label (str, optional) If splitting based on edge type, then this parameter
specifies the key for the type of edges to split on.

edge_attribute_label (str, optional): The label for the edge attribute to split
on.

edge_attribute_threshold (str, optional): The threshold value applied to the
edge attribute when sampling positive examples.

attribute_is_datetime (bool, optional): Specifies if edge attribute is datetime
or not.  seed (int, optional): seed for random number generator, positive int
or 0

Returns:
1) The reduced graph (positive edges removed)
2) np.array dim N (edges) holding the node ids for the edges
3) np.array dim N holding the edge labels, 0 for negative and 1 for positive examples

 '''

# initialize EdgeSplitter class
edge_splitter_test = EdgeSplitter(g)

# train_test_split: generate testing dataset
# take the input graph, and randomly remove positive and negative edges
print("\nGenerating testing dataset:")
[g_test,test_ids,test_labels] = \
        edge_splitter_test.train_test_split(
                p=0.1, # fraction of edges to remove
                method="global", # sample globally
                keep_connected=True) # remove edges, but don't break-up graph

# Repeat

# initialize EdgeSplitter class
edge_splitter_train = EdgeSplitter(g_test) # NOTE: input is g_test!

# train_test_split: generate training dataset
# take the test graph, and randomly remove positive and negative edges
print("\nGenerating training dataset:")
[g_train, train_ids, train_labels] = \
        edge_splitter_train.train_test_split(
                p=0.1,
                method="global",
                keep_connected=True)

'''
# print(g_test.info())
StellarGraph: Undirected multigraph

Nodes (paper): 2708
Edges (cites): 5429

** Sampled 542 positive and 542 negative edges. **

Nodes (paper): 2708
Edges (cites): 4399

** Sampled 488 positive and 488 negative edges. **

 Nodes: 2708
 Edges: 4887

 Node types:
  paper: [2708]
    Features: float32 vector, length 1440
    Edge types: paper-cites->paper

 Edge types:
    paper-cites->paper: [4887]
        Weights: all 1 (default)
        Features: none

'''

'''
#print(help(GraphSAGELinkGenerator))

class GraphSAGELinkGenerator
@stellargraph.mapper.sampled_link_generators

>>> class GraphSAGELinkGenerator(BatchedLinkGenerator)
GraphSAGELinkGenerator(
        G,              # sg.Graph with node features
        batch_size,     # batch size (links) to return
        num_samples,    # num of neighbors per hop to sample
        seed=None,      # seed for reproducibility
        weighted=False) # sample using edge weights

Returns a list of len(num_samples) of features form the sampled nodes:
    len(head_nodes),num_sampled_at_layer,feature_size)
'''

# generate data for link prediction
train_gen = GraphSAGELinkGenerator(
        g_train,
        batch_size,
        num_samples = [20,10])

# use the flow method supplied with (node) train_ids and train_labels (numeric
# targets) to create a generator for training
train_flow = train_gen.flow(train_ids, train_labels, shuffle=True)

# repeat for testing dataset:
test_gen = GraphSAGELinkGenerator(
        g_test,
        batch_size,
        num_samples=[20,10])

test_flow = test_gen.flow(test_ids, test_labels, shuffle=False)

'''
# print(help(train_gen.flow))

stellargraph.mapper.sampled_link_generators:
method of stellargraph.mapper.sampled_link_generators.GraphSAGELinkGenerator
instance Creates a generator/sequence object for training or evaluation
with the supplied node ids and numeric targets.

NOTE: that the shuffle argument should be True for training and
False for prediction.

    >>> flow(
            link_ids,       # node ids in the form (source, target)
            targets=None,   # 2D array of numeric targets cooresponding to link_ids
            shuffle=False,  # shuffle links at each epoch, otherwise order is maintained
            seed=None       # random seed
            )

@return NodeSequence object

'''

# build the GraphSAGE model using the training generator object
graphsage = GraphSAGE(
        layer_sizes = [20,20],
        generator=train_gen,
        bias=True,
        dropout=0.3)

# build the model and expose input and output sockets of graphsage model for link prediction
# call the in_out_tensors()

'''
# print(help(graphsage.in_out_tensors))

stellargraph.layer.graphsage.GraphSAGE
Builds a GraphSAGE model for node or link/node pair prediction,
depending on the generator used to construct the model
(whether it is a node or link/node pair generator).

>>> in_out_tensors(
        multiplicity=None
        )

@returns:
tuple: ``(x_inp, x_out)``, where ``x_inp`` is a list of Keras input
tensors for the specified GraphSAGE model (either node or link/node pair model)
and ``x_out`` contains model output tensor(s) of shape (batch_size,
layer_sizes[-1])

'''

[x_inp, x_out] = graphsage.in_out_tensors()


# final link classification layer that takes a pair of node embeddings
# produced by graphsage, applies a binary operator to them to produce the
# corresponding link embedding and passes it through a dense layer:
# defines a function that generates binary predictions for edges classification output from node embeddings
prediction = link_classification(
        output_dim=1,
        output_act="relu",
        edge_embedding_method="ip")(x_out)


# stack the graphsage and prediction layers into a keras model,
# and specify the loss
model = keras.Model(inputs=x_inp, outputs=prediction) # input and ouptut layers

model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3), # the optimizer to be used (tf.keras.optimizers)
        loss=keras.losses.binary_crossentropy, # objective loss function (tf.keras.losses)
        metrics=["accuracy"]) # metrics to be evaluated during training and testing (tf.keras.metrics)


'''
# print(help(model.evaluate))

@tensorflow.python.keras.engine.training
Returns the loss value & metrics values for the model in test mode.
NOTE: Computation is done in batches (see the `batch_size` arg.)

>>> evaluate(
        x=None,
        y=None,
        batch_size=None,
        verbose=1,
        sample_weight=None,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        return_dict=False)

@returns:
Scalar test loss (if the model has a single output and no metrics)
or list of scalars (if the model has multiple outputs
and/or metrics). The attribute `model.metrics_names` will give you
the display labels for the scalar outputs.

'''

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


# train the model:
history = model.fit(
        train_flow,
        epochs=epochs,
        validation_data=test_flow,
        verbose=2)

# sg.utils.plot_history(history)
dir(history)


# evaluate the trained model on test citation links:
train_metrics = model.evaluate(train_flow)
print("\ntrain set metrics of the trained model:")
for name, val in zip(model.metrics_names, train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

test_metrics = model.evaluate(test_flow)
print("\ntest set metrics of the trained model:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

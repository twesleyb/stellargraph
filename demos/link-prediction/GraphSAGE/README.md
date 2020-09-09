# Link Prediction in homogenous graphs with GraphSAGE

## Cora dataset (stellargraph.datasets):
'''
 Cora: https://linqs.soe.ucsc.edu/data
The Cora dataset consists of 2,708 scientific publications classified into one
of seven classes. The citation network consists of 5429 links. Each publication
in the dataset is described by a 0/1-valued word vector indicating the
absence/presence of the corresponding word from the dictionary. The dictionary
consists of 1433 unique words.
'''

## datasets.load
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

## stellargraph.info

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

# The test graph

'''

Nodes (paper): 2708
Edges (cites): 5429
'''

`** Sampled 542 positive and 542 negative edges. **`

`** Sampled 488 positive and 488 negative edges. **`

## GraphSAGELinkGenerator

'''
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

# print(help(train_gen.flow))
'''
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

# print(help(graphsage.in_out_tensors))
'''

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

# print(help(model.evaluate))

'''
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

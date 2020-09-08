#!/usr/bin/env python
# coding: utf-8
# title: stellargraph demos
# description: Loading data into StellarGraph from NumPy
# author: stellargraph

# If your data can easily be loaded into a NumPy array, this is a great way to
# load it that has minimal overhead and offers the most control.
#
# This notebook walks through loading three kinds of graphs.
#
# - homogeneous graph with feature vectors
# - homogeneous graph with feature tensors
# - heterogeneous graph with feature vectors and tensors
#
# This notebook only uses NumPy for the node features, with Pandas used for the
# edge data. The details and options for loading edge data in this format are
# discussed in [the "Loading data into StellarGraph from Pandas"
# demo](loading-pandas.ipynb).
#
# Additionally, if the node features are in a complicated format for loading
# and/or requires significant preprocessing, loading via Pandas is likely to be
# more convenient.
#

# A StellarGraph has two basic components:
#
# * nodes, with feature arrays or tensors
# * edges, consisting of a pair of nodes as the source and target, and feature arrays or tensors
#
# a numpy array consists of a large number of values of a single type. it is
# thus appropriate for the feature arrays in nodes, but not as useful for
# edges, because the source and target node ids may be different. thus, node
# data can be input as a numpy array directly, but edge data cannot. the latter
# still uses pandas.

## imports --------------------------------------------------------------------

import stellargraph as sg
import numpy as np
import pandas as pd

## Sequential numeric graph structure -----------------------------------------
#
# As with the Pandas demo, we'll be working with a square graph. For
# simplicity, we'll start with a graph where the identifiers of nodes are
# sequential integers starting at 0:
#
# ```
# 0 -- 1
# | \  |
# |  \ |
# 3 -- 2
# ```
#
# The edges of this graph can easily be encoded as the rows of a Pandas DataFrame:

edge_dict =  {"source": [0, 1, 2, 3, 0], "target": [1, 2, 3, 0, 2]}
square_numeric_edges = pd.DataFrame(edge_dict)

print(square_numeric_edges)


# ## Homogeneous graph with sequential IDs and feature vectors ----------------
#
# Now, suppose we have some feature vectors associated with each node in our
# square graph. For instance, maybe node `0` has features `[1, -0.2]`. This can
# come in the form of a 4 × 2 matrix, with one row per node, with row `0` being
# features for the `0` node, and so on. Filling out the rest of the example
# data:

array = [[1.0, -0.2], [2.0, 0.3], [3.0, 0.0], [4.0, -0.5]]
feature_array = np.array(array, dtype=np.float32)

print(feature_array) # ~ array of features for each node

# Because our nodes have IDs `0`, `1`, ..., we can construct the `StellarGraph`
# by passing in the feature array directly, along with the edges:
square_numeric = sg.StellarGraph(nodes=feature_array, edges=square_numeric_edges)

# The `info` method gives a high-level summary of a `StellarGraph`:
print(square_numeric.info())

# On this square, it tells us that there's 4 nodes of type `default` (a
# homogeneous graph still has node and edge types, but they default to
# `default`), with 2 features, and one type of edge that touches it. It also
# tells us that there's 5 edges of type `default` that go between nodes of type
# `default`. This matches what we expect: it's a graph with 4 nodes and 5 edges
# and one type of each.
#
# The default node type and edge types can be set using the `node_type_default`
# and `edge_type_default` parameters to `StellarGraph(...)`:

square_numeric_named = sg.StellarGraph(
    edges=square_numeric_edges,
    node_type_default="corner",
    edge_type_default="line",
)

print(square_numeric_named.info())


# ## Non-sequential graph structure -------------------------------------------
#
# Requiring node identifiers to always be sequential integers from 0 is
# restrictive. Most real-world graphs don't have such neat IDs. For instance,
# maybe our graph instead uses strings as IDs:
#
# ```
# a -- b
# | \  |
# |  \ |
# d -- c
# ```

# As before, these edges get encoded as a DataFrame:
df_dict = {"source": ["a", "b", "c", "d", "a"], "target": ["b", "c", "d", "a", "c"]}
square_edges = pd.DataFrame(df_dict)

print(square_edges)


## Homogeneous graph with non-numeric IDs and feature vectors -----------------
#
# With non-sequential, non-numeric IDs, we cannot use a NumPy array directly,
# because we need to know which row of the array corresponds to which node.
# This is done with the `IndexedArray` type.

# It is a much simplified Pandas DataFrame, that is generalised to be
# more than 2-dimensional. It is available at the top level of `stellargraph`,
# and supports an `index` parameter to define the mapping from row to node. The
# `index` should have one element per row of the NumPy array.


# build an IndexedArray by passing node attributes and node names
indexed_array = sg.IndexedArray(feature_array, index=["a", "b", "c", "d"])

# use the indexed array to construct a graph:
square_named = sg.StellarGraph(
        indexed_array, # using the indexed array as input
        square_edges,
        node_type_default="corner",
        edge_type_default="line"
        )

# As before, there's 4 nodes, each with features of length 2.
print(square_named.info())


## Homogeneous graph with non-numeric IDs and feature tensors -----------------
#
# Some algorithms work with than just a feature vector associated with each
# node. For instance, if each node corresponds to a weather station, one might
# have a time series of observations like "temperature" and "pressure"
# associated with each node. This is modelled by having a multidimensional
# feature for each node.
#
# Time series algorithms within StellarGraph expect the tensor to be shaped
# like `nodes × time steps × variates`. For the weather station example,
# `nodes` is the number of weather stations, `time steps` is the number of
# points within each series and `variates` is the number of observations at
# each time step.
#
# For our square graph, we might have time series of length three, containing
# two observations.

# two observations, at three time points, for every node:
feature_tensors = np.array(
    [
        [[1.0, -0.2], [1.0, 0.1], [0.9, 0.1]],
        [[2.0, 0.3], [1.9, 0.31], [2.1, 0.32]],
        [[3.0, 0.0], [10.0, 0.0], [3.0, 0.0]],
        [[4.0, -0.5], [0.0, -1.0], [1.0, -3.0]],
    ],
    dtype=np.float32,
)

print(feature_tensors)

# Add index
indexed_tensors = sg.IndexedArray(feature_tensors, index=["a", "b", "c", "d"])

# build the graph:
square_tensors = sg.StellarGraph(
    nodes = indexed_tensors,
    edges = square_edges,
    node_type_default="corner",
    edge_type_default="line",
)


# We can see that the features of the `corner` nodes are now listed as a
# tensor, with shape 3 × 2, matching the array we created above.
print(square_tensors.info())


## Heterogeneous graphs -------------------------------------------------------
#
# Some graphs have multiple types of nodes.
#
# For example, an academic citation network that includes authors might have
# `wrote` edges connecting `author` nodes to `paper` nodes, in addition to the
# `cites` edges between `paper` nodes. There could be `supervised` edges
# between `author`s ([example](https://academictree.org)) too, or any number of
# additional node and edge types. A knowledge graph (aka RDF, triple stores or
# knowledge base) is an extreme form of an heterogeneous graph, with dozens,
# hundreds or even thousands of edge (or relation) types. Typically in a
# knowledge graph, edges and their types represent the information associated
# with a node, rather than node features.
#
# `StellarGraph` supports all forms of heterogeneous graphs.
#
# A heterogeneous `StellarGraph` can be constructed in a similar way to a
# homogeneous graph, except we pass a dictionary with multiple elements instead
# of a single element like we did in the "homogeneous graph with features"
# section and others above. For a heterogeneous graph, a dictionary has to be
# passed; passing a single `IndexedArray` does not work.
#
# Let's return to the square graph from earlier:
#
# ```
# a -- b
# | \  |
# |  \ |
# d -- c
# ```
#
# ### Feature arrays
#
# Suppose `a` is of type `foo`, and no features, but `b`, `c` and `d` are of
# type `bar` and have two features each, e.g. for `b`, `0.4, 100`. Since the
# features are different shapes (`a` has zero), they need to be modeled as
# different types, with separate `IndexedArray`s.

square_foo = sg.IndexedArray(index=["a"])

bar_features = np.array([[0.4, 100], [0.1, 200], [0.9, 300]])

print(bar_features)


square_bar = sg.IndexedArray(bar_features, index=["b", "c", "d"])


# We have the information for the two node types `foo` and `bar` in separate
# DataFrames, so we can now put them in a dictionary to create a
# `StellarGraph`. Notice that `info()` is now reporting multiple node types, as
# well as information specific to each.

# combine the two, passing a dict with attribute arrays:
square_foo_and_bar = sg.StellarGraph(
        nodes={"foo": square_foo, "bar": square_bar},
        edges=square_edges)

print(square_foo_and_bar.info())


# Node IDs (the DataFrame index) needs to be unique across all types. For
# example, renaming the `a` corner to `b` like `square_foo_overlap` in the next
# cell, is not accepted and a `StellarGraph(...)` call will throw an error

square_foo_overlap = sg.IndexedArray(index=["b"])


# Uncomment to see the error
StellarGraph({"foo": square_foo_overlap, "bar": square_bar}, square_edges)

# If the node IDs aren't unique across types, one way to make them unique is to
# add a string prefix. You'll need to add the same prefix to the node IDs used
# in the edges too. Adding a prefix can be done by replacing the index:

square_foo_overlap_prefix = sg.IndexedArray(
    square_foo_overlap.values, index=[f"foo-{s}" for s in square_foo_overlap.index]
)

square_bar_prefix = sg.IndexedArray(
    square_bar.values, index=[f"bar-{s}" for s in square_bar.index]
)


# ### Feature tensors ---------------------------------------------------------
#
# Nodes of different types can have features of completely different shapes,
# not just vectors of different lengths. For instance, suppose our `foo` node
# (`a`) has the multi-variate time series from above as a feature.

# node a tensors:
foo_tensors = np.array([[[1.0, -0.2], [1.0, 0.1], [0.9, 0.1]]])
square_foo_tensors = sg.IndexedArray(foo_tensors, index=["a"])

# again, build a graph by passing a dict of nodes and their  attributes
square_foo_tensors_and_bar = sg.StellarGraph(
    nodes={"foo": square_foo_tensors, "bar": square_bar},
    edges=square_edges
)

# We can now see that the `foo` node is listed as having a feature tensor, as desired.
print(square_foo_tensors_and_bar.info())

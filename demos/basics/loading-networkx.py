#!/usr/bin/env python
# coding: utf-8

# # Loading data into StellarGraph from NetworkX

from stellargraph import StellarGraph


# ## Loading from many graph formats, via NetworkX

import networkx as nx

# ## Homogeneous graph without features
#
# To start with, we'll start with a homogeneous graph without
# any node features. This means the graph consists of only
# nodes and edges without any information other than a unique
# identifier.
#
# Let's use a graph representing a square with a diagonal:
#
# ``` a -- b | \  | |  \ | d -- c ```


g = nx.Graph() g.add_edge("a", "b") g.add_edge("b", "c")
g.add_edge("c", "d") g.add_edge("d", "a")
# diagonal
g.add_edge("a", "c")


# The basic form of constructing a `StellarGraph` from a
# NetworkX graphs is... just passing in that graph!

# In[6]:

square = StellarGraph.from_networkx(g)


# The `info` method
# ([docs](https://stellargraph.readthedocs.io/en/stable/api.html#stellargraph.StellarGraph.info))
# gives a high-level summary of a `StellarGraph`:

# In[7]:


print(square.info())


# On this square, it tells us that there's 4 nodes of type
# `default` (a homogeneous graph still has node and edge
# types, but they default to `default`), with no features, and
# one type of edge that touches it. It also tells us that
# there's 5 edges of type `default` that go between nodes of
# type `default`. This matches what we expect: it's a graph
# with 4 nodes and 5 edges and one type of each.
#
# Similar to constructing via Pandas, the default node type
# and edge types can be set using the `node_type_default` and
# `edge_type_default` parameters to
# `StellarGraph.from_networkx(...)`:

# In[8]:


square_named = StellarGraph.from_networkx( g,
        node_type_default="paper", edge_type_default="cites")
print(square_named.info())


# ## Homogeneous graph with features
#
# For many real-world problems, we have more than just graph
# structure: we have information about the nodes and edges.
# For instance, we might have a graph of academic papers
# (nodes) and how they cite each other (edges): we might have
# information about the nodes such as the authors and the
# publication year, and even the abstract or full paper
# contents. If we're doing a machine learning task, it can be
# useful to feed this information into models. The
# `StellarGraph.from_networkx` class supports this in two
# ways:
#
# 1. loading from an attribute, which stores a numeric
# sequence 2. using a Pandas DataFrame (this is the same as
# the `StellarGraph(...)` constructor from [the "loading from
# Pandas" tutorial](loading-pandas.ipynb))
#
# Let's continue using the same graph.
#
# ### 1. Loading from an attribute
#
# If the nodes of our graph comes or can be augmented with, a
# feature attribute that contains a numeric sequence (such as
# a list or a NumPy array), `StellarGraph.from_networkx` can
# load these to create node features.
#
# The feature attributes can be assigned in many ways, such as
# via `some_graph.nodes[node_id][feature_name] = ...` or by
# iterating over the nodes. We'll do the second one here:

# In[9]:


g_feature_attr = g.copy()


def compute_features(node_id):
    # in general this could compute something based on other
    # features, but for this example, we don't have any other
    # features, so we'll just do something basic with the
    # node_id
    return [ord(node_id), len(node_id)]


for node_id, node_data in g_feature_attr.nodes(data=True):
    node_data["feature"] = compute_features(node_id)

# let's see what some of them look like:
g_feature_attr.nodes["a"], g_feature_attr.nodes["c"]


# The `node_features=...` parameter let's us tell
# `from_networkx` how to find the features. If it's a string,
# it looks for an attribute by that name in each node of the
# graph.

# In[10]:


square_feature_attr =
StellarGraph.from_networkx(g_feature_attr,
        node_features="feature")
print(square_feature_attr.info())


# Notice how `info` now says that nodes of type `default` (all
# of them) have a feature vector of length 2. Success!
#
# In a homogeneous graph like this, the features for every
# node need to have the same length.

# ### 2. Using a Pandas DataFrame
#
# Another option is to have a Pandas DataFrame of features.
# This is often more efficient, if the data comes separately
# to the graph structure, or if significant preprocessing is
# required before creating the `StellarGraph`.
#
# The structure of the dataframe is the same as the nodes
# DataFrame used for the main `StellarGraph(...)` constructor
# ([docs](https://stellargraph.readthedocs.io/en/stable/api.html#stellargraph.StellarGraph)),
# which is covered in more detail in [the "loading from
# Pandas" tutorial](loading-pandas.ipynb). That tutorial
# includes examples of loading the DataFrame from a file. In
# this tutorial, we will just work with DataFrames that have
# already been loaded.

# In[11]:


import pandas as pd

features = pd.DataFrame( {"x": [1, 2, 3, 4], "y": [-0.2, 0.3,
    0.0, -0.5]}, index=["a", "b", "c", "d"]) features


# Notice how the IDs we used for the nodes in the NetworkX
# graph are the DataFrame's index. The index is how the
# features are connected to each node, and the nodes in the
# graph and nodes in the DataFrame need to match exactly.
#
# With a DataFrame in the appropriate format, we can pass this
# to the `node_features=...` parameter too.

# In[12]:


square_feature_dataframe = StellarGraph.from_networkx(
        g_feature_attr, node_features=features)
print(square_feature_dataframe.info())


# Like with the attribute, `info` now says that our nodes have
# a feature vector of length 2.
#
# We can use Pandas to do all sorts of feature preprocessing,
# like the column filtering and one-hot encoding done in the
# other tutorial.

# ## Homogeneous graph with edge weights
#
# Some algorithms can understand edge weights, which can be
# used as a measure of the strength of the connection, or a
# measure of distance between nodes. A `StellarGraph` instance
# can have weighted edges, by specifying a `weight` attribute
# on the edges.

# In[13]:


g_weighted = nx.Graph() g_weighted.add_edge("a", "b",
        weight=1.0) g_weighted.add_edge("b", "c", weight=2.0)
g_weighted.add_edge("c", "d", weight=3.0)
g_weighted.add_edge("d", "a", weight=4.0)
# diagonal
g_weighted.add_edge("a", "c", weight=5.0)

square_weighted = StellarGraph.from_networkx(g_weighted)
print(square_weighted.info())


# Notice the output of `info` now shows additional statistics
# about edge weights.
#
# The name of the attribute can be customised using the
# `edge_weight_attr` parameter.

# In[14]:


g_weighted_other = nx.Graph() g_weighted_other.add_edge("a",
        "b", distance=1.0) g_weighted_other.add_edge("b", "c",
                distance=2.0) g_weighted_other.add_edge("c",
                        "d", distance=3.0)
                g_weighted_other.add_edge("d", "a",
                        distance=4.0)
# diagonal
g_weighted_other.add_edge("a", "c", distance=5.0)

square_weighted_other = StellarGraph.from_networkx(
        g_weighted, edge_weight_attr="distance")
print(square_weighted.info())


# ## Directed graphs
#
# Some graphs have edge directions, where going from source to
# target has a different meaning to going from target to
# source. NetworkX models this using the `DiGraph` and
# `MultiDiGraph` classes, and `StellarGraph.from_networkx`
# automatically creates a directed graph if they are passed.
#
# All of the other options like node features and edge weights
# work the same as undirected graphs.

# In[15]:


g_directed = nx.DiGraph() g_directed.add_edge("a", "b")
g_directed.add_edge("b", "c") g_directed.add_edge("c", "d")
g_directed.add_edge("d", "a")
# diagonal
g_directed.add_edge("a", "c")

square_directed = StellarGraph.from_networkx(g_directed)
print(square_directed.info())


# ## Heterogeneous graphs
#
# Some graphs have multiple types of nodes and multiple types
# of edges.
#
# `StellarGraph` supports all forms of heterogeneous graphs,
# including knowledge graphs.
#
# The types of nodes and edges in a heterogeneous graph
# created using `StellarGraph.from_networkx` are read from a
# `label` attribute by default.
#
# ### Multiple node types
#
# Suppose `a` is of type `foo`, and `b`, `c` and `d` are of
# type `bar`. We can set the `label` attribute on each node
# appropriate using the `nx.set_node_attributes` function
# ([docs](https://networkx.github.io/documentation/stable/reference/generated/networkx.classes.function.set_node_attributes.html)).

# In[16]:


g_foo_bar = g.copy() nx.set_node_attributes( g_foo_bar, {"a":
    "foo", "b": "bar", "c": "bar", "d": "bar"}, "label")

square_foo_bar = StellarGraph.from_networkx(g_foo_bar)
print(square_foo_bar.info())


# If the `label` attribute doesn't exist, the
# `node_type_default` value is used.

# In[17]:


g_foo = g.copy()
# only 'a' has a label attribute
g_foo.nodes["a"]["label"] = "foo"

square_foo_bar_default = StellarGraph.from_networkx(g_foo,
        node_type_default="bar")
print(square_foo_bar_default.info())


# The attribute used to compute the node or edge type can be
# customised via the `node_type_attr` parameter. For instance,
# we can use the `type` attribute instead of the `label` one:

# In[18]:


g_foo_other = g.copy()
# only 'a' has a type attribute
g_foo_other.nodes["a"]["type"] = "foo"

square_foo_bar_other = StellarGraph.from_networkx( g_foo,
        node_type_default="bar", node_type_attr="type")
print(square_foo_bar_other.info())


# If we have features for the nodes, the features can be
# stored in the nodes under a features attribute. The name of
# the attribute has to be the same for all types, but the
# shape or size of the attribute does not.

# In[19]:


g_foo_bar_attr = g_foo_bar.copy() nx.set_node_attributes(
        g_foo_bar_attr, {"a": [], "b": [0.4, 100], "c": [0.1,
            200], "d": [0.9, 300]}, "feature",)


# In[20]:


square_foo_bar_features_attr = StellarGraph.from_networkx(
        g_foo_bar_attr, node_features="feature")
print(square_foo_bar_features_attr.info())


# Notice how the nodes of type `foo` (`a`) have no features,
# but the nodes of type `bar` (all others) have a vector of
# length 2.
#
# Alternatively, we can use Pandas DataFrames, specifying the
# `node_features=...` parameter as a dictionary mapping a node
# type to a DataFrame of features for nodes of that type. The
# dictionary only needs to include node types that have
# features.

# In[21]:


features_bar = pd.DataFrame( {"y": [0.4, 0.1, 0.9], "z": [100,
    200, 300]}, index=["b", "c", "d"]) features_bar


# In[22]:


square_foo_bar_features_dataframe =
StellarGraph.from_networkx( g_foo_bar, node_features={"bar":
    features_bar})
print(square_foo_bar_features_dataframe.info())


# ### Multiple edge types
#
# A heterogeneous graph with multiple edge types is supported
# in the same way, by looking for a `label` attribute (the
# name can be customised with the `edge_type_attr=...`
# parameter, like `node_type_attr=...`).

# In[23]:


g_orientation = nx.Graph() g_orientation.add_edge("a", "b",
        label="horizontal") g_orientation.add_edge("b", "c",
                label="vertical") g_orientation.add_edge("c",
                        "d", label="horizontal")
                g_orientation.add_edge("d", "a",
                        label="vertical")
                g_orientation.add_edge("a", "c",
                        label="diagonal")


# In[24]:


square_orientation = StellarGraph.from_networkx(g_orientation)
print(square_orientation.info())


# ### Multiple everything
#
# A graph can have multiple node types and multiple edge
# types, with features or without and with edge weights or
# without. We can put everything together from the previous
# sections to make a single complicated `StellarGraph`.

# In[25]:


g_everything = g_orientation.copy() nx.set_node_attributes(
        g_everything, {"a": "foo", "b": "bar", "c": "bar",
            "d": "bar"}, "label")


# In[26]:


stellar_everything = StellarGraph.from_networkx( g_everything,
        node_features={"bar": features_bar})
print(stellar_everything.info())


# ## Conclusion
#
# You hopefully now know more about building a `StellarGraph`
# in various configurations via NetworkX.
#
# Revisit this document to use as a reminder, or
# [documentation](https://stellargraph.readthedocs.io/en/stable/api.html#stellargraph.StellarGraph.from_networkx)
# for the `StellarGraph.from_networkx` static method.
#
# Once you've loaded your data, you can start doing machine
# learning: a good place to start is the [demo of the GCN
# algorithm on the Cora dataset for node
# classification](../node-classification/gcn-node-classification.ipynb).
# Additionally, StellarGraph includes [many other demos of
# other algorithms, solving other tasks](../README.md).

# <table><tr><td>Run the latest release of this
# notebook:</td><td><a
# href="https://mybinder.org/v2/gh/stellargraph/stellargraph/master?urlpath=lab/tree/demos/basics/loading-networkx.ipynb"
# alt="Open In Binder" target="_parent"><img
# src="https://mybinder.org/badge_logo.svg"/></a></td><td><a
# href="https://colab.research.google.com/github/stellargraph/stellargraph/blob/master/demos/basics/loading-networkx.ipynb"
# alt="Open In Colab" target="_parent"><img
# src="https://colab.research.google.com/assets/colab-badge.svg"/></a></td></tr></table>

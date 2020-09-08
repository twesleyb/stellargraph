#!/usr/bin/env python
# coding: utf-8

# # Loading data into StellarGraph from NumPy
# 
# > This demo explains how to load data from NumPy into a form that can be used
# by the StellarGraph library. [See all other demos](../README.md).

# [The StellarGraph library](https://github.com/stellargraph/stellargraph)
# supports loading graph information from NumPy. [NumPy](https://www.numpy.org)
# is a library for working with data arrays.
# 
# If your data can easily be loaded into a NumPy array, this is a great way to
# load it that has minimal overhead and offers the most control.
# 
# This notebook walks through loading three kinds of graphs.
# 
# - homogeneous graph with feature vectors - homogeneous graph with feature
# tensors - heterogeneous graph with feature vectors and tensors
# 
# > StellarGraph supports loading data from many sources with all sorts of data
# preprocessing, via [Pandas](https://pandas.pydata.org) DataFrames,
# [NumPy](https://www.numpy.org) arrays, [Neo4j](https://neo4j.com) and
# [NetworkX](https://networkx.github.io) graphs. This notebook demonstrates
# loading data from NumPy. See [the other loading demos](README.md) for more
# details.
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
# The
# [documentation](https://stellargraph.readthedocs.io/en/stable/api.html#stellargraph.StellarGraph)
# for the `StellarGraph` class includes a compressed reminder of everything
# discussed in this file, as well as explanations of all of the parameters.
# 
# The `StellarGraph` class is available at the top level of the `stellargraph`
# library:

# install StellarGraph if running on Google Colab
import sys
if 'google.colab' in sys.modules:
  get_ipython().run_line_magic('pip', 'install -q stellargraph[demos]==1.3.0b')


# In[2]:


# verify that we're using the correct version of StellarGraph for this notebook
import stellargraph as sg

try:
    sg.utils.validate_notebook_version("1.3.0b")
except AttributeError:
    raise ValueError(
        f"This notebook requires StellarGraph version 1.3.0b, but a different version {sg.__version__} is installed.  Please see <https://github.com/stellargraph/stellargraph/issues/1172>."
    ) from None


# In[3]:


from stellargraph import StellarGraph


# ## Loading via NumPy
# 
# A StellarGraph has two basic components:
# 
# - nodes, with feature arrays or tensors
# - edges, consisting of a pair of nodes as the source and target, and feature arrays or tensors
# 
# A NumPy array consists of a large number of values of a single type. It is thus appropriate for the feature arrays in nodes, but not as useful for edges, because the source and target node IDs may be different. Thus, node data can be input as a NumPy array directly, but edge data cannot. The latter still uses Pandas.

# In[4]:


import numpy as np
import pandas as pd


# ## Sequential numeric graph structure
# 
# As with the Pandas demo, we'll be working with a square graph. For simplicity, we'll start with a graph where the identifiers of nodes are sequential integers starting at 0:
# 
# ```
# 0 -- 1
# | \  |
# |  \ |
# 3 -- 2
# ```
# 
# The edges of this graph can easily be encoded as the rows of a Pandas DataFrame:

# In[5]:


square_numeric_edges = pd.DataFrame(
    {"source": [0, 1, 2, 3, 0], "target": [1, 2, 3, 0, 2]}
)
square_numeric_edges


# ## Homogeneous graph with sequential IDs and feature vectors
# 
# Now, suppose we have some feature vectors associated with each node in our square graph. For instance, maybe node `0` has features `[1, -0.2]`. This can come in the form of a 4 × 2 matrix, with one row per node, with row `0` being features for the `0` node, and so on. Filling out the rest of the example data:

# In[6]:


feature_array = np.array(
    [[1.0, -0.2], [2.0, 0.3], [3.0, 0.0], [4.0, -0.5]], dtype=np.float32
)
feature_array


# Because our nodes have IDs `0`, `1`, ..., we can construct the `StellarGraph` by passing in the feature array directly, along with the edges:

# In[7]:


square_numeric = StellarGraph(feature_array, square_numeric_edges)


# The `info` method ([docs](https://stellargraph.readthedocs.io/en/stable/api.html#stellargraph.StellarGraph.info)) gives a high-level summary of a `StellarGraph`:

# In[8]:


print(square_numeric.info())


# On this square, it tells us that there's 4 nodes of type `default` (a homogeneous graph still has node and edge types, but they default to `default`), with 2 features, and one type of edge that touches it. It also tells us that there's 5 edges of type `default` that go between nodes of type `default`. This matches what we expect: it's a graph with 4 nodes and 5 edges and one type of each.
# 
# The default node type and edge types can be set using the `node_type_default` and `edge_type_default` parameters to `StellarGraph(...)`:

# In[9]:


square_numeric_named = StellarGraph(
    feature_array,
    square_numeric_edges,
    node_type_default="corner",
    edge_type_default="line",
)
print(square_numeric_named.info())


# ## Non-sequential graph structure
# 
# Requiring node identifiers to always be sequential integers from 0 is restrictive. Most real-world graphs don't have such neat IDs. For instance, maybe our graph instead uses strings as IDs:
# 
# ```
# a -- b
# | \  |
# |  \ |
# d -- c
# ```
# 
# As before, these edges get encoded as a DataFrame:

# In[10]:


square_edges = pd.DataFrame(
    {"source": ["a", "b", "c", "d", "a"], "target": ["b", "c", "d", "a", "c"]}
)
square_edges


# ## Homogeneous graph with non-numeric IDs and feature vectors
# 
# With non-sequential, non-numeric IDs, we cannot use a NumPy array directly, because we need to know which row of the array corresponds to which node. This is done with the `IndexedArray` ([docs](https://stellargraph.readthedocs.io/en/stable/api.html#stellargraph.IndexedArray)) type. It is a much simplified Pandas DataFrame, that is generalised to be more than 2-dimensional. It is available at the top level of `stellargraph`, and supports an `index` parameter to define the mapping from row to node. The `index` should have one element per row of the NumPy array.

# In[11]:


from stellargraph import IndexedArray


# In[12]:


indexed_array = IndexedArray(feature_array, index=["a", "b", "c", "d"])


# In[13]:


square_named = StellarGraph(
    indexed_array, square_edges, node_type_default="corner", edge_type_default="line",
)
print(square_named.info())


# As before, there's 4 nodes, each with features of length 2.

# ## Homogeneous graph with non-numeric IDs and feature tensors
# 
# Some algorithms work with than just a feature vector associated with each node. For instance, if each node corresponds to a weather station, one might have a time series of observations like "temperature" and "pressure" associated with each node. This is modelled by having a multidimensional feature for each node.
# 
# Time series algorithms within StellarGraph expect the tensor to be shaped like `nodes × time steps × variates`. For the weather station example, `nodes` is the number of weather stations, `time steps` is the number of points within each series and `variates` is the number of observations at each time step.
# 
# For our square graph, we might have time series of length three, containing two observations.

# In[14]:


feature_tensors = np.array(
    [
        [[1.0, -0.2], [1.0, 0.1], [0.9, 0.1]],
        [[2.0, 0.3], [1.9, 0.31], [2.1, 0.32]],
        [[3.0, 0.0], [10.0, 0.0], [3.0, 0.0]],
        [[4.0, -0.5], [0.0, -1.0], [1.0, -3.0]],
    ],
    dtype=np.float32,
)
feature_tensors


# In[15]:


indexed_tensors = IndexedArray(feature_tensors, index=["a", "b", "c", "d"])


# In[16]:


square_tensors = StellarGraph(
    indexed_tensors, square_edges, node_type_default="corner", edge_type_default="line",
)
print(square_tensors.info())


# We can see that the features of the `corner` nodes are now listed as a tensor, with shape 3 × 2, matching the array we created above.

# ## Heterogeneous graphs
# 
# Some graphs have multiple types of nodes.
# 
# For example, an academic citation network that includes authors might have `wrote` edges connecting `author` nodes to `paper` nodes, in addition to the `cites` edges between `paper` nodes. There could be `supervised` edges between `author`s ([example](https://academictree.org)) too, or any number of additional node and edge types. A knowledge graph (aka RDF, triple stores or knowledge base) is an extreme form of an heterogeneous graph, with dozens, hundreds or even thousands of edge (or relation) types. Typically in a knowledge graph, edges and their types represent the information associated with a node, rather than node features.
# 
# `StellarGraph` supports all forms of heterogeneous graphs.
# 
# A heterogeneous `StellarGraph` can be constructed in a similar way to a homogeneous graph, except we pass a dictionary with multiple elements instead of a single element like we did in the "homogeneous graph with features" section and others above. For a heterogeneous graph, a dictionary has to be passed; passing a single `IndexedArray` does not work.
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
# Suppose `a` is of type `foo`, and no features, but `b`, `c` and `d` are of type `bar` and have two features each, e.g. for `b`, `0.4, 100`. Since the features are different shapes (`a` has zero), they need to be modeled as different types, with separate `IndexedArray`s.

# In[17]:


square_foo = IndexedArray(index=["a"])


# In[18]:


bar_features = np.array([[0.4, 100], [0.1, 200], [0.9, 300]])
bar_features


# In[19]:


square_bar = IndexedArray(bar_features, index=["b", "c", "d"])


# We have the information for the two node types `foo` and `bar` in separate DataFrames, so we can now put them in a dictionary to create a `StellarGraph`. Notice that `info()` is now reporting multiple node types, as well as information specific to each.

# In[20]:


square_foo_and_bar = StellarGraph({"foo": square_foo, "bar": square_bar}, square_edges)
print(square_foo_and_bar.info())


# Node IDs (the DataFrame index) needs to be unique across all types. For example, renaming the `a` corner to `b` like `square_foo_overlap` in the next cell, is not accepted and a `StellarGraph(...)` call will throw an error

# In[21]:


square_foo_overlap = IndexedArray(index=["b"])


# In[22]:


# Uncomment to see the error
# StellarGraph({"foo": square_foo_overlap, "bar": square_bar}, square_edges)


# If the node IDs aren't unique across types, one way to make them unique is to add a string prefix. You'll need to add the same prefix to the node IDs used in the edges too. Adding a prefix can be done by replacing the index:

# In[23]:


square_foo_overlap_prefix = IndexedArray(
    square_foo_overlap.values, index=[f"foo-{s}" for s in square_foo_overlap.index]
)


# In[24]:


square_bar_prefix = IndexedArray(
    square_bar.values, index=[f"bar-{s}" for s in square_bar.index]
)


# ### Feature tensors
# 
# Nodes of different types can have features of completely different shapes, not just vectors of different lengths. For instance, suppose our `foo` node (`a`) has the multi-variate time series from above as a feature.

# In[25]:


foo_tensors = np.array([[[1.0, -0.2], [1.0, 0.1], [0.9, 0.1]]])
foo_tensors


# In[26]:


square_foo_tensors = IndexedArray(foo_tensors, index=["a"])


# In[27]:


square_foo_tensors_and_bar = StellarGraph(
    {"foo": square_foo_tensors, "bar": square_bar}, square_edges
)
print(square_foo_tensors_and_bar.info())


# We can now see that the `foo` node is listed as having a feature tensor, as desired.

# ## Conclusion
# 
# You hopefully now know more about building node features for a `StellarGraph` in various configurations via NumPy arrays.
# 
# For more details on graphs with directed, weighted or heterogeneous edges, see [the "Loading data into StellarGraph from Pandas" demo](loading-pandas.ipynb). All of the examples there work with `IndexedArray` instead of Pandas DataFrames for the node features.
# 
# Revisit this document to use as a reminder, or [the documentation](https://stellargraph.readthedocs.io/en/stable/api.html#stellargraph.StellarGraph) for the `StellarGraph` class.
# 
# Once you've loaded your data, you can start doing machine learning: a good place to start is the [demo of the GCN algorithm on the Cora dataset for node classification](../node-classification/gcn-node-classification.ipynb). Additionally, StellarGraph includes [many other demos of other algorithms, solving other tasks](../README.md).

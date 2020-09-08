#!/usr/bin/env python
# coding: utf-8

# # Loading data into StellarGraph from Pandas
# 
# > This demo explains how to load data into a form that can be used by the StellarGraph library. [See all other demos](../README.md).
# 

# <table><tr><td>Run the latest release of this notebook:</td><td><a href="https://mybinder.org/v2/gh/stellargraph/stellargraph/master?urlpath=lab/tree/demos/basics/loading-pandas.ipynb" alt="Open In Binder" target="_parent"><img src="https://mybinder.org/badge_logo.svg"/></a></td><td><a href="https://colab.research.google.com/github/stellargraph/stellargraph/blob/master/demos/basics/loading-pandas.ipynb" alt="Open In Colab" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a></td></tr></table>

# [The StellarGraph library](https://github.com/stellargraph/stellargraph) supports loading graph information from Pandas. [Pandas](https://pandas.pydata.org) is a library for working with data frames.
# 
# This is a great way to load data that offers a good balance between performance and convenience.
# 
# The StellarGraph library supports many deep machine learning (ML) algorithms on [graphs](https://en.wikipedia.org/wiki/Graph_%28discrete_mathematics%29). A graph consists of a set of *nodes* connected by *edges*, potentially with information associated with each node and edge. Any task using the StellarGraph library needs data to be loaded into an instance of the `StellarGraph` class. This class stores the graph structure (the nodes and the edges between them), as well as information about them:
# 
# - *node types* and *edge types*: a class or category to which the nodes and edges belong, dictating what features are available on a node, and potentially signifying some sort of semantic meaning (this is different to machine learning label for a node)
# - *node features* and *edge features*: vectors of numbers associated with each node or edge
# - *edge weights*: a number associated with each edge
# 
# All of these are optional, because they have sensible defaults if they're not relevant to the task at hand.
# 
# This notebook walks through loading several kinds of graphs using Pandas. Pandas is a reasonably efficient form of loading, that is convenient for preprocessing.
# 
# - homogeneous graph without features (a homogeneous graph is one with only one type of node and one type of edge)
# - homogeneous graph with node/edge features
# - homogeneous graph with edge weights
# - directed graphs (a graph is directed if edges have a "start" and "end" nodes, instead of just connecting two nodes)
# - heterogeneous graphs (more than one node type and/or more than one edge type) with and without node/edge features or edge weights, this includes knowledge graphs
# - real data: homogeneous graph from CSV files (an example of reading data from files and doing some preprocessing)
# 
# > StellarGraph supports loading data from many sources with all sorts of data preprocessing, via [Pandas](https://pandas.pydata.org) DataFrames, [NumPy](https://www.numpy.org) arrays, [Neo4j](https://neo4j.com) and [NetworkX](https://networkx.github.io) graphs. See [all loading demos](README.md) for more details.
# 
# The [documentation](https://stellargraph.readthedocs.io/en/stable/api.html#stellargraph.StellarGraph) for the `StellarGraph` class includes a compressed reminder of everything discussed in this file, as well as explanations of all of the parameters.
# 
# The `StellarGraph` class is available at the top level of the `stellargraph` library:

# In[1]:


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


# ## Loading from anything, via Pandas
# 
# Pandas DataFrames are tables of data that can be created from [many input sources](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html), such as [CSV files](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) and [SQL databases](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html). StellarGraph builds on this power by allowing construction from these DataFrames.

# In[4]:


import pandas as pd


# Pandas is widely supported by other libraries and products, like [scikit-learn](http://scikit-learn.github.io/stable), and thus a user of StellarGraph gets to benefit from these easily too.

# ## Homogeneous graph without features
# 
# We'll start with a homogeneous graph without any node features. This means the graph consists of only nodes and edges without any information other than a unique identifier.
# 
# The basic form of constructing a `StellarGraph` is passing in an edge `DataFrame` with two columns (`source` and `target`), where each row represents a pair of nodes that are connected. Let's construct a `StellarGraph` representing a square with a diagonal:
# 
# ```
# a -- b
# | \  |
# |  \ |
# d -- c
# ```
# 
# We'll start with a synthetic DataFrame defined in code here (there's some examples later of reading DataFrames from files).
# 
# Each row represents a connection: for instance, the first one is the edge from `a` to `b`.

# In[5]:


square_edges = pd.DataFrame(
    {"source": ["a", "b", "c", "d", "a"], "target": ["b", "c", "d", "a", "c"]}
)
square_edges


# Given our edges, we can create a `StellarGraph` directly:

# In[6]:


square = StellarGraph(edges=square_edges)


# The `info` method ([docs](https://stellargraph.readthedocs.io/en/stable/api.html#stellargraph.StellarGraph.info)) gives a high-level summary of a `StellarGraph`:

# In[7]:


print(square.info())


# On this square, it tells us that there's 4 nodes of type `default` (a homogeneous graph still has node and edge types, but they default to `default`), with no features, and one type of edge that touches it. It also tells us that there's 5 edges of type `default` that go between nodes of type `default`. This matches what we expect: it's a graph with 4 nodes and 5 edges and one type of each.
# 
# The default node type and edge types can be set using the `node_type_default` and `edge_type_default` parameters to `StellarGraph(...)`:

# In[8]:


square_named = StellarGraph(
    edges=square_edges, node_type_default="corner", edge_type_default="line"
)
print(square_named.info())


# The names of the columns used for the edges can be controlled with the `source_column` and `target_column` parameters to `StellarGraph(...)`. For instance, maybe our graph comes from a file with `first` and `second` columns:

# In[9]:


square_edges_first_second = square_edges.rename(
    columns={"source": "first", "target": "second"}
)
square_edges_first_second


# In[10]:


square_first_second = StellarGraph(
    edges=square_edges_first_second, source_column="first", target_column="second"
)
print(square_first_second.info())


# ## Homogeneous graph with features
# 
# For many real-world problems, we have more than just graph structure: we have information about the nodes and edges. For instance, we might have a graph of academic papers (nodes) and how they cite each other (edges): we might have information about the nodes such as the authors and the publication year, and even the abstract or full paper contents. If we're doing a machine learning task, it can be useful to feed this information into models. The `StellarGraph` class supports this using a Pandas DataFrame: each row corresponds to a feature vector for a node or edge.
# 
# ### Node features
# 
# Let's imagine the nodes have two features, which might be their coordinates, or maybe some other piece of information. We'll continue using synthetic DataFrames, but these could easily be read from a file. (There's an example in the "Real data: Homogeneous graph from CSV files" section at the end of this notebook.)

# In[11]:


square_node_data = pd.DataFrame(
    {"x": [1, 2, 3, 4], "y": [-0.2, 0.3, 0.0, -0.5]}, index=["a", "b", "c", "d"]
)
square_node_data


# `StellarGraph` uses the index of the DataFrame as the connection between a node and a row of the DataFrame. Notice that the `square_features` DataFrame has `a`, ..., `d` as its index, matching the identifiers used in the edges.
# 
# We've now got all the right node data, in addition to the edges from before, so now we can create a `StellarGraph`.

# In[12]:


square_node_features = StellarGraph(square_node_data, square_edges)
print(square_node_features.info())


# Notice the output of `info` now says that the nodes of the `default` type have 2 features.
# 
# We can also give the node and edge types helpful names, using either the `node_type_default`/`edge_type_default` parameters we saw before, or by passing the DataFrames in with a dictionary, where the key is the name of the type.

# In[13]:


square_named_node_features = StellarGraph(
    {"corner": square_node_data}, {"line": square_edges}
)
print(square_named_node_features.info())


# ### Edge features
# 
# Edges can have features in the same way as nodes. Any columns that don't have a special meaning are taken as feature vector elements. This means that the source and target columns are not included in the feature vectors (nor are the weight or edge type columns, that are discussed later).
# 
# Let's imagine the edges have 3 features each.

# In[14]:


square_edge_data = pd.DataFrame(
    {
        "source": ["a", "b", "c", "d", "a"],
        "target": ["b", "c", "d", "a", "c"],
        "A": [-1, 2, -3, 4, -5],
        "B": [0.4, 0.1, 0.9, 0, 0.9],
        "C": [12, 34, 56, 78, 90],
    }
)
square_edge_data


# In[15]:


square_named_features = StellarGraph(
    {"corner": square_node_data}, {"line": square_edge_data}
)
print(square_named_features.info())


# Notice the output of `info` now says that the edges of the `line` type have 3 features, in addition to the 2 features for each node of type `corner`.

# ## Homogeneous graph with edge weights
# 
# Some algorithms can understand edge weights, which can be used as a measure of the strength of the connection, or a measure of distance between nodes. A `StellarGraph` instance can have weighted edges, by including a `weight` column in the DataFrame of edges.
# 
# We'll continue with the synthetic square example, by adding that extra `weight` column into the DataFrame. This column might be part of the data naturally, or it might need to be computed. Either of these is fine with Pandas: in the first case, it can be loaded at the same time as loading the source and target information, and in the second, the full power of Pandas is available to compute it (such as manipulating other information associated with the edge DataFrame, or even by comparing the nodes at each end).

# In[16]:


square_weighted_edges = pd.DataFrame(
    {
        "source": ["a", "b", "c", "d", "a"],
        "target": ["b", "c", "d", "a", "c"],
        "weight": [1.0, 0.2, 3.4, 5.67, 1.0],
    }
)
square_weighted_edges


# In[17]:


square_weighted = StellarGraph(edges=square_weighted_edges)
print(square_weighted.info())


# Notice the output of `info` now shows additional information about edge weights.
# 
# Edges weights can be used with node and edge features; for instance, we create a similar graph to the last graph in the "Homogeneous graph with features" section that has our edge weights:

# In[18]:


square_weighted_edge_data = pd.DataFrame(
    {
        "source": ["a", "b", "c", "d", "a"],
        "target": ["b", "c", "d", "a", "c"],
        "weight": [1.0, 0.2, 3.4, 5.67, 1.0],
        "A": [-1, 2, -3, 4, -5],
        "B": [0.4, 0.1, 0.9, 0, 0.9],
        "C": [12, 34, 56, 78, 90],
    }
)
square_weighted_edge_data


# In[19]:


square_features_weighted = StellarGraph(
    {"corner": square_node_data}, {"line": square_weighted_edge_data}
)
print(square_features_weighted.info())


# ## Directed graphs
# 
# Some graphs have edge directions, where going from source to target has a different meaning to going from target to source.
# 
# A directed graph can be created by using the `StellarDiGraph` class instead of the `StellarGraph` one. The construction is almost identical, and we can reuse any of the DataFrames that we created in the sections above. For instance, continuing from the previous cell, we can have a directed homogeneous graph with node features and edge weights.

# In[20]:


from stellargraph import StellarDiGraph

square_features_weighted_directed = StellarDiGraph(
    {"corner": square_node_data}, {"line": square_weighted_edge_data}
)
print(square_features_weighted_directed.info())


# Everything discussed about `StellarGraph` in this file also works with `StellarDiGraph`, including parameters like `node_type_default` and `source_column`.

# ## Heterogeneous graphs
# 
# Some graphs have multiple types of nodes and multiple types of edges.
# 
# For example, an academic citation network that includes authors might have `wrote` edges connecting `author` nodes to `paper` nodes, in addition to the `cites` edges between `paper` nodes. There could be `supervised` edges between `author`s ([example](https://academictree.org)) too, or any number of additional node and edge types. A knowledge graph (aka RDF, triple stores or knowledge base) is an extreme form of an heterogeneous graph, with dozens, hundreds or even thousands of edge (or relation) types. Typically in a knowledge graph, edges and their types represent the information associated with a node, rather than node features.
# 
# `StellarGraph` supports all forms of heterogeneous graphs.
# 
# A heterogeneous `StellarGraph` can be constructed in a similar way to a homogeneous graph, except we pass a dictionary with multiple elements instead of a single element like we did for the Cora examples in the "homogeneous graph with features" section and others above. For a heterogeneous graph, a dictionary has to be passed; passing a single DataFrame does not work.
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
# ### Multiple node types
# 
# Suppose `a` is of type `foo`, and no features, but `b`, `c` and `d` are of type `bar` and have two features each, e.g. for `b`, `y = 0.4, z = 100`. Since the features are different shapes (`a` has zero), they need to be modeled as different types, with separate `DataFrame`s.

# In[21]:


square_foo = pd.DataFrame(index=["a"])
square_foo


# In[22]:


square_bar = pd.DataFrame(
    {"y": [0.4, 0.1, 0.9], "z": [100, 200, 300]}, index=["b", "c", "d"]
)
square_bar


# We have the information for the two node types `foo` and `bar` in separate DataFrames, so we can now put them in a dictionary to create a `StellarGraph`. Notice that `info()` is now reporting multiple node types, as well as information specific to each.

# In[23]:


square_foo_and_bar = StellarGraph({"foo": square_foo, "bar": square_bar}, square_edges)
print(square_foo_and_bar.info())


# Node IDs (the DataFrame index) needs to be unique across all types. For example, renaming the `a` corner to `b` like `square_foo_overlap` in the next cell, is not accepted and a `StellarGraph(...)` call will throw an error

# In[24]:


square_foo_overlap = pd.DataFrame({"x": [-1]}, index=["b"])
square_foo_overlap


# In[25]:


# Uncomment to see the error
# StellarGraph({"foo": square_foo_overlap, "bar": square_bar}, square_edges)


# If the node IDs aren't unique across types, one way to make them unique is to add a string prefix. You'll need to add the same prefix to the node IDs used in the edges too. Adding a prefix can be done by replacing the index:

# In[26]:


square_foo_overlap_prefix = square_foo_overlap.set_index(
    "foo-" + square_foo_overlap.index.astype(str)
)
square_foo_overlap_prefix


# In[27]:


square_bar_prefix = square_bar.set_index("bar-" + square_bar.index.astype(str))
square_bar_prefix


# ### Multiple edge types: type column
# 
# Graphs with multiple edge types can be simpler. Since there are often no features on the edges, we can pass a DataFrame with an additional column for the type, specifying it via the `edge_type_column` parameter. If there are features on the edges, multiple edge types can also be created in the same way as multiple node types, by passing with a dictionary of DataFrames.
# 
# For example, suppose the edges in our square graph have types based on their orientation.

# In[28]:


square_edges_types = square_edges.assign(
    orientation=["horizontal", "vertical", "horizontal", "vertical", "diagonal"]
)
square_edges_types


# In[29]:


square_orientation = StellarGraph(
    edges=square_edges_types, edge_type_column="orientation"
)
print(square_orientation.info())


# Edge weights are supported, in the same way as a homogeneous graph above, with a `weight` column:

# In[30]:


square_edges_types_weighted = square_edges_types.assign(weight=[1.0, 0.2, 3.4, 5.67, 1.0])
square_edges_types_weighted


# In[31]:


square_orientation_weighted = StellarGraph(
    edges=square_edges_types_weighted, edge_type_column="orientation"
)
print(square_orientation_weighted.info())


# ### Multiple edge types: edge features
# 
# As mentioned above, if there are multiple edge types and the edges have edge features, one will typically need to pass a dictionary of DataFrames similar to multiple node types. The features of each type can be different.
# 
# Note: Edges also have IDs (the DataFrame index, like nodes), and they need to be unique across all edge types.

# In[32]:


square_edges_horizontal = pd.DataFrame(
    {"source": ["a", "c"], "target": ["b", "d"], "A": [-1, -3]}, index=[0, 2]
)
square_edges_vertical = pd.DataFrame(
    {"source": ["b", "d"], "target": ["c", "a"], "B": [0.1, 0], "C": [34, 78]},
    index=[1, 3],
)
square_edges_diagonal = pd.DataFrame({"source": ["a"], "target": ["c"]}, index=[4])

# example:
square_edges_horizontal


# In[33]:


square_orientation_separate = StellarGraph(
    edges={
        "horizontal": square_edges_horizontal,
        "vertical": square_edges_vertical,
        "diagonal": square_edges_diagonal,
    },
)
print(square_orientation_separate.info())


# Notice that `vertical` edges have 2 features, `horizontal` have 1, and `diagonal` have 0.
# 
# Edge weights can be specified with this multiple-DataFrames form too. Any or all of the DataFrames for an edge type can contain a `weight` column.

# In[34]:


square_edges_horizontal_weighted = square_edges_horizontal.assign(weight=[12.3, 45.6])
square_edges_horizontal_weighted


# In[35]:


square_orientation_separate_weighted = StellarGraph(
    edges={
        "horizontal": square_edges_horizontal_weighted,
        "vertical": square_edges_vertical,
        "diagonal": square_edges_diagonal,
    },
)
print(square_orientation_separate_weighted.info())


# ### Multiple everything
# 
# A graph can have multiple node types and multiple edge types, with features or without, with edge weights or without and with `edge_type_column=...` (shown here) or with multiple DataFrames for edge types. We can put everything together from the previous sections to make a single complicated `StellarGraph`.

# In[36]:


square_everything = StellarGraph(
    {"foo": square_foo, "bar": square_bar},
    square_edges_types_weighted,
    edge_type_column="orientation",
)
print(square_everything.info())


# ### Directed heterogeneous graphs
# 
# A heterogeneous graph can be directed by using `StellarDiGraph` to construct it, similar to a homogeneous graph.

# In[37]:


from stellargraph import StellarDiGraph

square_everything_directed = StellarDiGraph(
    {"foo": square_foo, "bar": square_bar},
    square_edges_types_weighted,
    edge_type_column="orientation",
)
print(square_everything_directed.info())


# ## Real data: Homogeneous graph from CSV files
# 
# We've been using a synthetic square graph with perfectly formatted data as an example for this whole notebook, because it helps us focus on just the core `StellarGraph` functionality. Real life isn't so simple; there's usually files to wrangle and formats to convert, so we'll finish this demo covering some example steps to go from data in files to a `StellarGraph`.
# 
# We'll work with the Cora dataset from <https://linqs.soe.ucsc.edu/data>:
# 
# > The Cora dataset consists of 2708 scientific publications classified into one of seven classes. The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words. The README file in the dataset provides more details. 
# 
# The dataset contains two files: `cora.cites` and `cora.content`.
# 
# `cora.cites` is a tab-separated values (TSV) file of the graph edges. The first column identifies the cited paper, and the second column identifies the paper that cites it. The first three lines of the file look like:
# 
# ```
# 35	1033
# 35	103482
# 35	103515
# ...
# ```
# 
# `cora.content` is also a TSV file of information about each node (paper), with 1435 columns: the first column is the node ID (matching the IDs used in `cora.cites`), the next 1433 are the 0/1-values of word vectors, and the last is the subject area class of the paper. The first three lines of the file look like (with the 1423 of the 0/1 columns truncated)
# 
# ```
# 31336	0	0	...	0	1	0	0	0	0	0	0	Neural_Networks
# 1061127	0	0	...	1	0	0	0	0	0	0	0	Rule_Learning
# 1106406	0	0	...	0	0	0	0	0	0	0	0	Reinforcement_Learning
# ...
# ```
# 
# This graph is homogeneous (all nodes are papers, and all edges are citations), with node features (the 0/1-values) but no edge weights.
# 
# The StellarGraph library provides the `datasets` module ([docs](https://stellargraph.readthedocs.io/en/stable/api.html#module-stellargraph.datasets)) for working with some common datasets via classes like `Cora` ([docs](https://stellargraph.readthedocs.io/en/stable/api.html#stellargraph.datasets.Cora)). It can download the necessary files via the `download` method. (The `load` method also converts it into a `StellarGraph`, but that's too helpful for this tutorial: we're learning how to do that ourselves.)

# In[38]:


from stellargraph.datasets import Cora
import os

cora = Cora()
cora.download()

# the base_directory property tells us where it was downloaded to:
cora_cites_file = os.path.join(cora.base_directory, "cora.cites")
cora_content_file = os.path.join(cora.base_directory, "cora.content")


# We've now got the files on disk, so we can read them using the `pd.read_csv` function. Despite the "CSV" in the name, this function can be used to read TSV files too. The files don't have a row of column headings, so we'll want to set our own.
# 
# First, the edges. We can use `source` and `target` as the column headings, to match `StellarGraph`'s defaults. However, the natural phrasing is "paper X cites paper Y", not "paper Y is cited by paper X", so we use the columns in reverse order to match.

# In[39]:


cora_cites = pd.read_csv(
    cora_cites_file,
    sep="\t",  # tab-separated
    header=None,  # no heading row
    names=["target", "source"],  # set our own names for the columns
)
cora_cites


# Now, the nodes. Again, we have to choose the columns' names. The names of the 0/1-columns don't matter so much, but we can give the first column (of IDs) and the last one (of subjects) useful names.

# In[40]:


cora_feature_names = [f"w{i}" for i in range(1433)]

cora_raw_content = pd.read_csv(
    cora_content_file,
    sep="\t",  # tab-separated
    header=None,  # no heading row
    names=["id", *cora_feature_names, "subject"],  # set our own names for the columns
)
cora_raw_content


# As we saw above when adding node features, `StellarGraph` uses the index of the DataFrame as the connection between a node and a row of the DataFrame. Currently our dataframe just has a simple numeric range as the index, but it needs to be using the `id` column. Pandas offers [a few ways to control the indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#set-reset-index); in this case, we want to replace the current index by moving the `id` column to it, which is done most easily with `set_index`:

# In[41]:


cora_content_str_subject = cora_raw_content.set_index("id")
cora_content_str_subject


# We're almost ready to create the `StellarGraph`, we just have to do something about the non-numeric `subject` column. Many machine learning models only work on numeric features, requiring text and other data to be converted before apply; the models in StellarGraph are no different.
# 
# There are two options, depending on the task:
# 
# 1. remove the `subject` column entirely: many uses of Cora are predicting the `subject` of a node, given all of the graph structure and other information, so including it as information in the graph is giving the answer directly
# 2. convert it to numeric via [one-hot](https://en.wikipedia.org/wiki/One-hot) encoding, where we have 7 columns of 0 and 1, one for each subject value (similar to the 1433 other `w...` features)
# 
# We'll look at both (feel free to skip ahead to 2).
# 
# ### 1. Removing columns
# 
# Let's start with the first, removing the columns. The `drop` method ([docs](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html)) lets us remove one or more columns.

# In[42]:


cora_content_no_subject = cora_content_str_subject.drop(columns="subject")
cora_content_no_subject


# We've got all the right node data, and the right edges, so now we can create a `StellarGraph` using the techniques we saw in the "homogeneous graph with features" section above.

# In[43]:


cora_no_subject = StellarGraph({"paper": cora_content_no_subject}, {"cites": cora_cites})
print(cora_no_subject.info())


# If we're trying to predict the subject, we'll probably need to use the `subject` labels as ground-truth labels in a supervised or semi-supervised machine learning task. This can be extracted from the DataFrame and held separately, to be passed in as training, validation or test examples.

# In[44]:


cora_subject = cora_content_str_subject["subject"]
cora_subject


# This is a normal Pandas Series, and so can be manipulated with any of the functions that support it. For example, if we wanted to train a machine learning algorithm using 25% of the nodes, we could use the `train_test_split` function ([docs](http://www.scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)) from [the scikit-learn library](https://scikit-learn.org/).

# In[45]:


from sklearn import model_selection

cora_train, cora_test = model_selection.train_test_split(
    cora_subject, train_size=0.25, random_state=123
)
cora_train


# In[46]:


cora_test


# This dataset, with this preparation, is used in [a demo of the GCN algorithm for node classification](../node-classification/gcn-node-classification.ipynb). The task is to predict the subject of each node.

# ### 2. One-hot encoding
# 
# Now, let's look at the other approach: converting the subjects to numeric features. The `pd.get_dummies` function ([docs](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html)) can do this for us, by adding extra columns (7, in this case), based on the unique values.

# In[47]:


cora_content_one_hot_subject = pd.get_dummies(
    cora_content_str_subject, columns=["subject"]
)
cora_content_one_hot_subject


# Using this DataFrame, we can create a `StellarGraph` with 1440 features per node instead of 1433 like the previous section.

# In[48]:


cora_one_hot_subject = StellarGraph(
    {"paper": cora_content_one_hot_subject}, {"cites": cora_cites}
)
print(cora_one_hot_subject.info())


# ## Conclusion
# 
# You hopefully now know more about building a `StellarGraph` in various configurations via Pandas DataFrames, including some feature preprocessing in the "Real data: Homogeneous graph from CSV files" section.
# 
# Revisit this document to use as a reminder, or [the documentation](https://stellargraph.readthedocs.io/en/stable/api.html#stellargraph.StellarGraph) for the `StellarGraph` class.
# 
# Once you've loaded your data, you can start doing machine learning: a good place to start is the [demo of the GCN algorithm on the Cora dataset for node classification](../node-classification/gcn-node-classification.ipynb). Additionally, StellarGraph includes [many other demos of other algorithms, solving other tasks](../README.md).

# <table><tr><td>Run the latest release of this notebook:</td><td><a href="https://mybinder.org/v2/gh/stellargraph/stellargraph/master?urlpath=lab/tree/demos/basics/loading-pandas.ipynb" alt="Open In Binder" target="_parent"><img src="https://mybinder.org/badge_logo.svg"/></a></td><td><a href="https://colab.research.google.com/github/stellargraph/stellargraph/blob/master/demos/basics/loading-pandas.ipynb" alt="Open In Colab" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a></td></tr></table>

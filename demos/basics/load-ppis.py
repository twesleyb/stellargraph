#!/usr/bin/env python
# coding: utf-8
# title: stellargraph demos
# description: Loading data into StellarGraph from NumPy
# author: stellargraph

## input
# * mus_ppi_adjm.csv

## imports

import numpy as np
import pandas as pd
import networkx as nx
import stellargraph as sg

# load the data
df = pd.read_csv('mus_hitpredict_ppis.csv',

# collect edges as a list of tuples
edge_tuples = list(zip(df['osEntrezA'],df['osEntrezB']))

# create networkx graph and add edges
g = nx.Graph()
[g.add_edge(*e) for e in edge_tuples]

# coerce to StellarGraph
G = sg.StellarGraph.from_networkx(g)

print(G.info())

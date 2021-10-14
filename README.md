# Hypergraphs for Multimorbidity Research

[![Tests](https://github.com/jim-rafferty/multimorbidity_hypergraphs/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/jim-rafferty/multimorbidity_hypergraphs/actions/workflows/python-package-conda.yml) 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5569023.svg)](https://doi.org/10.5281/zenodo.5569023)

This is a collection of tools for constructing and analysing hypergraphs from 
data. Hypergraphs are very general and powerful objects for data analysis which 
connect nodes and edges. As for binary graphs, nodes can conect to any number of 
edges but in a hypergraph, edges can connect to any number of nodes which leads
to some very useful features! 

This set of tools is for the analysis of large scale data with hypergraphs, 
i.e, a specialist
toolkit that will focus on a small number of specific features. If you 
are looking for a general tool kit for hypergraphs including visualisation, check 
out hypernetx (https://github.com/pnnl/HyperNetX). 

A publication describing what this code does can be found here: https://www.sciencedirect.com/science/article/pii/S1532046421002458

Install using pip: `pip install multimorbidity-hypergraphs`

To import: `import multimorbidity_hypergraphs as hgt`

Run tests using `pytest` in the top directory.
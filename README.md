# Bachelor-Thesis
The Bachelor Thesis of Philipp Zimmermann

---

## Topic Description

`TODO`

---

## Basic Understanding of GNNs
Covers basic understanding of graph neural networks [1]

### Graphs
- data structures
- model set of objects and their relationships
- nodes represent objects
- edges represent relationships


- formal: `G = (V, E)`
  - G : graph
  - V : set of nodes / vertex
  - E : set of edges


### Graph Neural Networks (GNNs)
- nodes add information from neighbors
- last layer combines all information
- output:
  - node classification
  - link prediction
  - graph classification

#### Node classification
- every node has a label
- new nodes get labels assigned based on classification


- example:
  - two proteins in a network of 16 proteins
  - 0th and 15th are related to a cancerous disease
  - network should classify which proteins are the most related to each other

#### Link prediction
- predict likelihood whether two nodes are linked or not


- example:
  - social networks
  - predict whether two people know each other or not

#### Graph classification
- classify graphs into different classes
- help discovering patterns in user's interaction in social networks


- example:
  - social network
  - group users by age
  - targeted advertisements

### CNNs and Network Embedding
#### Convolutional Neural Networks
- share weights
- share local connections
- consist of many layers

#### Network Embedding
- transformation of input networks to low dimensional vectors

### Python Libraries
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest)
- [Graph Nets](https://github.com/deepmind/graph_nets)
- [Deep Graph Library DGL](https://www.dgl.ai/)


- [Zachary's Karate Club problem](https://docs.dgl.ai/tutorials/basics/1_first.html#sphx-glr-download-tutorials-basics-1-first-py)

### Models

### Applications

---

## References
All references that are included in the bachelor thesis are listed below.

#### Websites
- [1] https://www.section.io/engineering-education/an-introduction-to-graph-neural-network/


#### Research Paper
- Basic GNN paper
  - https://arxiv.org/abs/1609.02907 (transduction)
  - https://arxiv.org/abs/1706.02216 (induction)
- Link stealing attack
  - https://arxiv.org/abs/2005.02131

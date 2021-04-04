# Basic Understanding of GNNs

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
- [Beginners Guide for DGL](https://docs.dgl.ai/tutorials/basics/2_basics.html)

### Models
#### Recurrent Graph Neural Networks - RGNNs
- aim to learn node representations using RNNS
- they assume that nodes in the graph exchange messages constantly
- this exchange continues until a stable equilibrium is achieved

#### Convolutional Graph Neural Networks - CGNNs
- generalization of of convolutional operations from grid to graph format
- many layers of convolution

#### Graph Auto-Encoders - GAEs
- deep neural networks, that generate new graphs
- use bottleneck principle (GANs)
- link prediction in [citation networks](https://arxiv.org/pdf/1611.07308.pdf)

#### Spatial-Temporal Graph Neural Networks - STGNNs
- consider spatial and temporal dependencies

### Applications
#### Computer Vision
- scene graph generation
  - separate image data
  - achieve semantic graph and the relationship between the objects
- Action Recognition
  - learn patterns in human actions
  - detect locations of human joints
  - linked by skeletons

#### Recommendation Systems
- users and items as nodes
- model ...
  - users to user's relationship
  - items to items relationship
  - users to items relationship
- tell importance of an item to a user

#### Natural Language Processing
- classification of text data
- relationship between works

# Bachelor-Thesis
> Philipp Zimmermann - 6th Semester B.Sc. Cybersecurity


## Topic Description

Given a graph dataset `dataset` ( nodes and edges ). Let `target` be a Graph Neural Network (GraphSAGE, GAT, GCN) that is trained on `dataset` to perform node classification. Let `attacker` be a Multi Layer Perceptron that is trained on features generated based on the output-posteriors of `target` to perform link stealing attacks.

Let `i` and `j` be two nodes in `dataset`. The goal is to predict whether the edge `(i,j)` exists or not. Therefor `target` is queried twice, once for each node, giving back two posteriors for the labels `post_i` and `post_j`. The distance between `post_i` and `post_j` is calculated eight times (cosine, euclidean, correlation, chebyshev, braycurtis, manhattan, canberra, sqeuclidean) and concatenated, being the input feature for `attacker`. Based on that `attacker` now decides whether `i` and `j` are connected or not.

## References
These references have been used for initial research / getting familiar with the topic. Every reference that was used for the Thesis can be find in the Bibliography section.

#### Websites
- [1] https://www.section.io/engineering-education/an-introduction-to-graph-neural-network/

#### Videos
- [1] Node Classification: https://www.youtube.com/watch?v=ex2qllcVneY
- [2] Graph Introduction and Label Propagation: https://www.youtube.com/watch?v=OI0Jo-5d190
- [3] Graph Convolutional Neural Networks: https://www.youtube.com/watch?v=2KRAOZIULzw
- [4] Message Passing: https://www.youtube.com/watch?v=ijmxpItkRjc
- [5] Relational Graph Neural Networks: https://www.youtube.com/watch?v=wJQQFUcHO5U
- [6] Implementation of Zachary's Karate Club problem: https://www.youtube.com/watch?v=8qTnNXdkF1Q
- [7] GraphSAGE - Inductive Representation Learning on Large Graphs: https://www.youtube.com/watch?v=vinQCnizqDA

#### Repositories
- [1] GraphSAGE: https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage
- [2] GAT: https://github.com/dmlc/dgl/blob/master/examples/pytorch/gat

#### Research Paper
- [1] Basic GNN - Transductive: https://arxiv.org/abs/1609.02907
- [2] Basic GNN - Inductive: https://arxiv.org/abs/1706.02216
- [3] Link stealing attacks: https://arxiv.org/abs/2005.02131

## Timeline

**Start** : 1. April 2021

| Design State                         | Time in weeks | Start
|---                                   |---            |---
| Familiarization Phase                | 2             | 01/04
| Design Experiment                    | 4             | 08/04
| Writing ( Research, Evaluation, ... )| -             | 30/04
| Intro Talk Preparation               | -             | 14/05
| Final Talk Preparation               | -             | -

**End** : in progress

# Bachelor-Thesis
> Philipp Zimmermann

## Topic Description

Given a graph dataset `d` ( nodes and edges ). Let `f` be a GraphSAGE Graph Neural Network that is trained on `d` to predict labels of nodes. Let `a` be a Fully Connected Neural Network that is trained on the output-posteriors of `f` to perform a link stealing attack.

Let `i` and `j` be two nodes in `d`. The goal is to predict whether the edge `(i,j)` exists or not. Therefor `f` is queried twice, once for each node, giving back two posteriors for the labels `post_i` and `post_j`. These are now concatenated, being the input feature for `a`. Based on that `a` now decides whether `i` and `j` are connected or not.

## References
All references that are included in the bachelor thesis itself or used for initial research are listed below.

### Initial Research

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
- [3] Forked DGL-Repository: https://github.com/Pilladian/dgl

#### Research Paper
- [1] Basic GNN - Transductive: https://arxiv.org/abs/1609.02907
- [2] Basic GNN - Inductive: https://arxiv.org/abs/1706.02216
- [3] Link stealing attack: https://arxiv.org/abs/2005.02131

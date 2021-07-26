# Bachelor-Thesis
> Philipp Zimmermann - 6th Semester B.Sc. Cybersecurity


## Abstract

Since nowadays graphs are a common way to store and visualize data, Machine Learning
algorithms have been improved to directly operate on them. In most cases the graph
itself can be deemed confidential, since the owner of the data often spends much time
and resources collecting and preparing the data. In our work, we show, that so called
inductive trained graph neural networks can reveal sensitive information about their
training graph. We focus on extracting information about the edges of the target training
graph by observing the predictions of the target model in so called link stealing attacks.
In prior work, He et al. proposed the first link stealing attacks on graph neural networks,
focusing on the transductive learning setting. More precisely, given black box access
to a graph neural network, the authors were able to predict, whether two nodes of a
graph that was used for training the model, are linked or not. In our work, we now focus
on the inductive setting. Specifically, given black box access to a graph neural network
model that was trained inductively, we are able to predict whether there exists a link
between any two nodes of the training graph or not. Our experiments show that there
exist efficient ways to properly reveal sensitive information about the training graphs of
inductive trained graph neural networks, which leads to big privacy concerns. Depending
on datasets and the graph neural network model we achieve up to 0.8955 F1-Score while
having an average performance of 0.7817 F1-Score regarding all our attacks.

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

**Registration** : 14. June 2021

| Design State                         | Time in weeks | Start
|---                                   |---            |---
| Familiarization Phase                | 2             | 01/04
| Design Experiment                    | 4             | 08/04
| Writing ( Research, Evaluation, ... )| 12            | 30/04
| Reviews                              | -             | 22/07
| Intro Talk Preparation               | 4             | 14/05
| Final Talk Preparation               | -             | -
| Printing                             | -             | -

**Submission** : 14. September 2021

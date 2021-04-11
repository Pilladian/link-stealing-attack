# Bachelor-Thesis
The Bachelor Thesis of Philipp Zimmermann

---

## Topic Description

Given a Social Network represented as a graph. Let `f` be the GNN that is trained to predict labels (e.g. salary or education status). To calculate the embeddings of the nodes GraphSAGE is used.

- Training
  - for every node v in the Graph
    - collect all neighbors of v
    - aggregate neighbors embeddings from the prior iteration
    - concatenate current embedding with aggregated one
    - use this embedding as input for `f`

- Testing
  - new node inserted in the graph
  - generate embeddings with GraphSAGE Algorithm
  - query `f` with embedding to get a label prediction
  - output is a posterior


Given the posteriors of two nodes that have been inserted, is it possible to extract information about their connection? More precisely, is it possible to predict whether two people are connected, with at least one node between them, based on the output posterior of the trained model?

---

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
- [1] DGL: https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage

#### Research Paper
- [1] Basic GNN - Transductive: https://arxiv.org/abs/1609.02907
- [2] Basic GNN - Inductive: https://arxiv.org/abs/1706.02216
- [3] Link stealing attack: https://arxiv.org/abs/2005.02131

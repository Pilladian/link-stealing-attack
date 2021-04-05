# Bachelor-Thesis
The Bachelor Thesis of Philipp Zimmermann

---

## Topic Description

Given a Social Network represented as a graph. Let `f` be the GNN that is trained to predict labels (e.g. salary of education status). To create the embeddings of the nodes GraphSAGE is used.

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


Given two posteriors of two nodes that have been inserted, is it possible to extract information about their connection. More precisely, is it possible to predict whether two people are connected based on the output posterior of the trained model?


A GNN (GraphSAGE) is trained on multiple subgraphs of a graph. At the testing phase (unknown / unseen subgraphs) the GNN outputs a posterior. Based on the posteriors I try to reveal information about centroid nodes. Especially if they are connected somehow.

E.g.: I try to predict whether two people on facebook, A and B, know each other (is there a path including multiple other people from A to B) based on the posteriors output of the GNN.

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

#### Research Paper
- [1] Basic GNN - Transductive: https://arxiv.org/abs/1609.02907
- [2] Basic GNN - Inductive: https://arxiv.org/abs/1706.02216
- [3] Link stealing attack: https://arxiv.org/abs/2005.02131

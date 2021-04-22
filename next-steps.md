#### [ 2021-04-09 ]

- [x] Get familiar with DGL implementation of GraphSAGE, maybe also check the document
- [ ] Read 
  - [x] [GAT](https://arxiv.org/abs/1710.10903)
  - [ ] [GIN](https://arxiv.org/abs/1810.00826)
  - [ ] [GCN](https://arxiv.org/abs/1609.02907) paper

#### [ 2021-04-16 ]

- [x] Add argparse for different settings
  - [x] Log results ( readable linup and json for evaluation )
  - [x] Verbose output while running script ( per attack )
  - [x] Different GNN types
  - [x] Different datasets
- [x] Implement the baseline
  - [x] Accuracy for baseline with traingraph ( remove all edges )
  - [x] Accuracy for baseline with testgraph ( remove all edges )
- [x] More metrices
  - [x] Precision
  - [x] Recall
  - [x] F1-Score
  - [x] Accuracy
- [ ] Work on impromevents
  - [x] More edges survive in graph (are not deleted)
    - [x] 5% surviving edges ( 95% of deleted edges can be used for training )
    - [x] 10% surviving edges ( 90% of deleted edges can be used for training )
    - [x] 20% surviving edges ( 80% of deleted edges can be used for training )
    - [x] 50% surviving edges ( 50% of deleted edges can be used for training )
    - [x] 80% surviving edges ( 20% of deleted edges can be used for training ) 
  - [ ] Different types of GNNs
    - [x] GAT
  - [ ] Node Classification Datasets

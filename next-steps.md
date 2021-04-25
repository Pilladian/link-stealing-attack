#### [ 2021-04-09 ]

- [x] Get familiar with DGL implementation of GraphSAGE, maybe also check the document
- [ ] Read 
  - [x] [GAT](https://arxiv.org/abs/1710.10903)
  - [ ] [GIN](https://arxiv.org/abs/1810.00826)
  - [ ] [GCN](https://arxiv.org/abs/1609.02907)

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
    - [x] 20% surviving edges ( 20% of edges can be used for training )
    - [ ] 40% surviving edges ( 40% of edges can be used for training )
    - [ ] 60% surviving edges ( 60% of edges can be used for training )
    - [x] 80% surviving edges ( 80% of edges can be used for training ) 
  - [ ] Different types of GNNs
    - [x] GAT
    - [ ] GIN
    - [x] GCN
  - [ ] Node Classification Datasets
    - [x] PPI

#### [ 2021-04-23 ]

- [ ] Include old and new datasets ( Testing and Training -> Randomize instead of ids )
- [ ] Create Evaluation Markdown file for attacks
- [ ] Thread Model ( Knowledge of Attacker )
- [ ] Attack Methodology ( What does attacker need to do for attacking )
- [ ] Visualization ( Plot )

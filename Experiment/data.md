## GraphSAGE

#### Baseline - Test graph used for evaluation
| Attack | Target Domain | Attacker Domain |     F1-Score
| -------|---------------|-----------------|---------------
| baseline_test_post | Cora | Cora         |     73.69
| baseline_test_dist | Cora | Cora         |     72.83
| baseline_test_dist | Cora | Citeseer     |     **78.75**
| baseline_test_dist | Cora | Pubmed       |     75.58

| Attack | Target Domain | Attacker Domain |     F1-Score
| -------|---------------|-----------------|---------------
| baseline_test_post | Citeseer | Citeseer |     77.29
| baseline_test_dist | Citeseer | Citeseer |     **79.65**
| baseline_test_dist | Citeseer | Cora     |     70.14
| baseline_test_dist | Citeseer | Pubmed   |     76.26

| Attack | Target Domain | Attacker Domain |     F1-Score
| -------|---------------|-----------------|---------------
| baseline_test_post | Pubmed | Pubmed     |     73.39
| baseline_test_dist | Pubmed | Pubmed     |     75.47
| baseline_test_dist | Pubmed | Cora       |     69.67
| baseline_test_dist | Pubmed | Citeseer   |     **79.63**


#### Surviving Edges 40%
| Attack | Target Domain | Attacker Domain   |     F1-Score
| -------|---------------|-------------------|---------------
| surviving_edges_40p_post | Cora | Cora     |     79.65
| surviving_edges_40p_dist | Cora | Cora     |     79.52
| surviving_edges_40p_dist | Cora | Citeseer |     **84.30**
| surviving_edges_40p_dist | Cora | Pubmed   |     78.15

| Attack | Target Domain | Attacker Domain |     F1-Score
| -------|---------------|-----------------|---------------
| surviving_edges_40p_post | Citeseer | Citeseer     |     76.83
| surviving_edges_40p_dist | Citeseer | Citeseer     |     **84.69**
| surviving_edges_40p_dist | Citeseer | Cora         |     79.44
| surviving_edges_40p_dist | Citeseer | Pubmed       |     79.75

| Attack | Target Domain | Attacker Domain |     F1-Score
| -------|---------------|-----------------|---------------
| surviving_edges_40p_post | Pubmed | Pubmed       |     76.02
| surviving_edges_40p_dist | Pubmed | Pubmed       |     80.98
| surviving_edges_40p_dist | Pubmed | Cora         |     74.44
| surviving_edges_40p_dist | Pubmed | Citeseer     |     **85.31**

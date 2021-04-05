# Basic Understanding of Graph Neural Networks

## Scenario
Given a Social Network represented as a graph. The nodes are people in the network. Two people are connected if at least one follows the other one.

Each node has features. In this case an IP-address, this persons age, gender and name and the highest education. As a label for each node we set the salary per year. The label represents the information we are interested in for prediction. Based on the given information we train a Graph Neural Network.

- **Input**
  - Features of the node
  - Social Graph


- **Training**
  - learning a mapping function from node features and graph to the label


- **Testing**
  - apply network to unlabeled data to predict it based on features and the graph

That means that only one matrix is shared over the whole graph, which gets updated over the time.

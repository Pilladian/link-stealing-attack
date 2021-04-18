# Python 3.8.5

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Imports
import argparse
import json
import os
from src.utils import *
from src import *


class Experiment:

    def __init__(self, gnn, dataset):
        self.gnn_type = gnn
        self.dataset_name = dataset
        self.descs = {}
        self.targets = {}
        self.attacker = {}
        self.results = {}

    def initialize(self):
        # load dataset
        self.dataset = load_data(self.dataset_name)
        self.original_graph = self.dataset[0]
        self.num_classes = self.dataset.num_classes
        # split dataset
        self._split_dataset()

    def _split_dataset(self):
        split = self.original_graph.number_of_nodes() * 0.5
        train_mask = torch.zeros(self.original_graph.number_of_nodes(), dtype=torch.bool)
        test_mask = torch.zeros(self.original_graph.number_of_nodes(), dtype=torch.bool)

        for a in range(self.original_graph.number_of_nodes()):
            train_mask[a] = a < split
            test_mask[a] = a >= split

        traingraph = self.original_graph.subgraph(train_mask)
        testgraph = self.original_graph.subgraph(test_mask)

        # remove self loops
        self.traingraph = dgl.remove_self_loop(traingraph)
        self.testgraph = dgl.remove_self_loop(testgraph)

    def _create_target(self, tname, traingraph):
        self.targets[tname] = Target(self.gnn_type, traingraph, self.num_classes)
        self.targets[tname].train()

    def evaluate_attack(self, attack_name, graph):
        target = self.targets[attack_name]
        tacc = target.evaluate(graph)

        attacker = self.attacker[attack_name]
        aprec, arecall, af1, aacc = attacker.evaluate(attacker.test_nid)

        self.results[attack_name] = {'target': tacc,
                                     'attacker': {'prec': aprec,
                                                  'recall': arecall,
                                                  'f1-score': af1,
                                                  'acc': aacc}}
        print_attack_results(tacc, aprec, arecall, af1, aacc)


    def baseline(self):
        # Baseline 1 - Train on traingraph - Test on traingraph
        attack_name = 'baseline_1'
        # description of attack
        self.descs[attack_name] = '''\n      --- Baseline 1 ---

      The target model in this attack is trained on the traingraph-subset of the original dataset.
      The attacker samples his dataset on the traingraph-subset as well.
      Both models are evaluated on the traingraph-subset.'''
        print(self.descs[attack_name])
        # target
        self._create_target(attack_name, self.traingraph)
        # attacker
        self.attacker[attack_name] = Attacker(self.targets[attack_name], self.traingraph)
        self.attacker[attack_name].create_modified_graph(0) # delete all edges -> 0 percent survivors
        self.attacker[attack_name].sample_data(0.2, 0.4)
        self.attacker[attack_name].train()
        # evaluate baseline 1
        self.evaluate_attack(attack_name, self.traingraph)

        # Baseline 2 - Train on traingraph - Test on testgraph
        attack_name = 'baseline_2'
        # description of attack
        self.descs[attack_name] = '''\n      --- Baseline 2 ---

      The target model in this attack is trained on the traingraph-subset of the original dataset.
      The attacker samples his dataset on the testgraph-subset of the original dataset.
      Both models are evaluated on the traingraph-subset.'''
        print(self.descs[attack_name])
        # target
        self._create_target('baseline_2', self.traingraph)
        # attacker
        self.attacker[attack_name] = Attacker(self.targets[attack_name], self.testgraph)
        self.attacker[attack_name].create_modified_graph(0) # delete all edges -> 0 percent survivors
        self.attacker[attack_name].sample_data(0.2, 0.4)
        self.attacker[attack_name].train()
        # evaluate baseline 2
        self.evaluate_attack('baseline_2', self.testgraph)




def main(args):
    # Init
    os.system('clear')
    print_desc('init')
    # Datasets
    datasets = print_datasets(args.dataset)
    # GNNs
    gnns = print_gnns(args.gnn)

    # create Attack objects
    experiments = []
    for gnn in gnns:
        for dataset in datasets:
            exp = Experiment(gnn, dataset)
            exp.initialize()
            experiments.append(exp)

    # run attacks
    for experiment in experiments:
        print(f'  [+] Run Attacks on {experiment.gnn_type} trained with {experiment.dataset_name}')
        experiment.baseline()

    # Conclude all results
    final_evaluation(experiments)
        # [2.0] [Toggle] Description
        # [2.1] Target Model
            # [2.1.1] Parameter
            # [2.1.2] Dataset-Stats
            # [2.1.3] Parameter

        # [2.2] Attacker Model
            # [2.2.1] Parameter
            # [2.2.2] Dataset-Stats
            # [2.2.3] Parameter


    # [3] 1. Modification ( more Edges )

    # [4] 2. Modification ( different  )

if __name__ == '__main__':

    # cmd args
    parser = argparse.ArgumentParser(description='Link-Stealing Attack')
    parser.add_argument("--desc",
                        action="store_false",
                        help="Ouput Process Descriptions")

    parser.add_argument("--dataset",
                        type=str,
                        default="all",
                        help="Select Dataset [all, cora, citeseer, pubmed]")

    parser.add_argument("--gnn",
                        type=str,
                        default='all',
                        help="Type of GNN [all, graphsage, ...]")


    args = parser.parse_args()
    main(args)


#       GNN         Dataset     B1 Target     B1 Attacker     B2 Target     B2 Attacker
#      ---------------------------------------------------------------------------------
#       graphsage   cora        0.7851        0.7411          0.8006        0.6802
#       graphsage   citeseer    0.7264        0.7214          0.7222        0.7206
#       graphsage   pubmed      0.8497        0.6535          0.8513        0.6875

# add time

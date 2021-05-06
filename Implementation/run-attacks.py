# Python 3.8.5

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Imports
import argparse
import random
import json
import os
from src.utils import *
from src import *


class Experiment:

    def __init__(self, gnn, dataset, verbose):
        self.verbose = verbose
        self.gnn_name = gnn
        self.dataset_name = dataset
        self.attacker = {}
        self.results = {}

    def initialize(self):
        # load dataset
        self.dataset = load_data(self.dataset_name)
        self.original_graph = self.dataset[0]
        self.num_classes = self.dataset.num_classes
        # load subgraphs
        self._load_dataset_subgraphs()
        # load target model
        self._load_target_model()

    def update_parameter(self, d):
        # original dataset d
        self.dataset = load_data(d)

        # get train / test graph
        self.dir = f'./models/{d}/{self.gnn_name}/'
        traingraph, labels1 = load_graphs(f'{self.dir}traingraph.bin', [0])
        testgraph, labels2 = load_graphs(f'{self.dir}testgraph.bin', [0])

        self.traingraph = traingraph[0]
        self.testgraph = testgraph[0]

        # load model
        self.target = Target(self.gnn_name, self.traingraph, self.dataset.num_classes)
        self.target._initialize()
        self.target.model.load_state_dict(torch.load(f'{self.dir}model.pt'))

    def _load_dataset_subgraphs(self):
        self.dir = f'./models/{self.dataset_name}/{self.gnn_name}/'
        traingraph, labels1 = load_graphs(f'{self.dir}traingraph.bin', [0])
        testgraph, labels2 = load_graphs(f'{self.dir}testgraph.bin', [0])

        self.traingraph = traingraph[0]
        self.testgraph = testgraph[0]

    def _load_target_model(self):
        # train-graph not important since model is only queried
        self.target = Target(self.gnn_name, self.traingraph, self.dataset.num_classes)
        self.target._initialize()
        self.target.model.load_state_dict(torch.load(f'{self.dir}model.pt'))

    def evaluate_attack(self, attack_name, graph, verbose=False):
        tacc = self.target.evaluate(graph)

        attacker = self.attacker[attack_name]
        aprec, arecall, af1, aacc = attacker.evaluate(attacker.test_nid)

        self.results[attack_name] = {'target': {'acc': tacc},
                                     'attacker': {'prec': aprec,
                                                  'recall': arecall,
                                                  'f1-score': af1,
                                                  'acc': aacc}}
        if verbose:
            print_attack_results(tacc, aprec, arecall, af1, aacc)


    def baseline_1(self):
        """
        Baseline 1:
        The target model in this attack is trained on the traingraph-subset of the original dataset.
        The attacker model samples its dataset on the traingraph-subset and removes all edges.
        Both models are evaluated on the traingraph-subset.
        """
        # Baseline 1 - Train on traingraph - Test on traingraph
        attack_name = f'baseline_1'
        print_attack_start(attack_name)

        # attacker
        self.attacker[attack_name] = Attacker(self.target, self.traingraph)
        self.attacker[attack_name].create_modified_graph(0) # delete all edges -> 0 percent survivors
        self.attacker[attack_name].sample_data_posteriors(0.2, 0.4)
        self.attacker[attack_name].train()

        # evaluate baseline 1
        print_attack_done(attack_name)
        self.evaluate_attack(attack_name, self.traingraph, verbose=self.verbose)

    def baseline_2(self):
        """
        Baseline 2:
        The target model in this attack is trained on the traingraph-subset of the original dataset.
        The attacker model samples its dataset on the testgraph-subset and removes all edges.
        Both models are evaluated on the testgraph-subset.
        """
        # Baseline 2 - Train on traingraph - Test on testgraph
        attack_name = 'baseline_2'
        print_attack_start(attack_name)

        # attacker
        self.attacker[attack_name] = Attacker(self.target, self.testgraph)
        self.attacker[attack_name].create_modified_graph(0) # delete all edges -> 0 percent survivors
        self.attacker[attack_name].sample_data_posteriors(0.2, 0.4)
        self.attacker[attack_name].train()

        # evaluate baseline 2
        print_attack_done(attack_name)
        self.evaluate_attack('baseline_2', self.testgraph, verbose=self.verbose)


    def surviving_edges(self, survivors):
        """
        The target model in this attack is trained on the traingraph-subset of the original dataset.
        The attacker model samples its dataset on the testgraph-subset and removes almost all edges.
        Both models are evaluated on the testgraph-subset.
        """
        # Surviving Edges
        attack_name = f'surviving_edges_{int(survivors*100)}p'
        print_attack_start(attack_name)

        # attacker
        self.attacker[attack_name] = Attacker(self.target, self.testgraph)
        self.attacker[attack_name].create_modified_graph(survivors)
        self.attacker[attack_name].sample_data_posteriors(0.2, 0.4)
        self.attacker[attack_name].train()

        # evaluate surviving_edges
        print_attack_done(attack_name)
        self.evaluate_attack(attack_name, self.testgraph, verbose=self.verbose)

    def baseline_1_dist(self, d1, d2):
        """
        Baseline 1_dist:
        The target model in this attack is trained on the traingraph-subset of the original dataset d1.
        The attacker model samples its dataset on the traingraph-subset of dataset d2 and removes all edges.
        Both models are evaluated on their traingraph-subset.
        """
        # Baseline 1_distances - Train on traingraph (d1) - Test on traingraph (d2)
        attack_name = f'baseline_1_dist_{d1}_{d2}'
        print_attack_start(attack_name)

        # attacker
        self.attacker[attack_name] = Attacker(self.target, self.traingraph)
        self.attacker[attack_name].create_modified_graph(0) # delete all edges -> 0 percent survivors
        self.attacker[attack_name].sample_data_vector_distances(0.2, 0.4)
        self.attacker[attack_name].train()

        # evaluate baseline 1_distances
        self.update_parameter(d2)
        self.attacker[attack_name].target_model = self.target
        self.attacker[attack_name].graph = self.traingraph
        self.attacker[attack_name].create_modified_graph(0)
        self.attacker[attack_name].sample_data_vector_distances(0.2, 0.4)
        print_attack_done(attack_name)
        self.evaluate_attack(attack_name, self.traingraph, verbose=self.verbose)

    def baseline_2_dist(self, d1, d2):
        """
        Baseline 2_dist:
        The target model in this attack is trained on the traingraph-subset of the original dataset d1.
        The attacker model samples its dataset on the testgraph-subset of dataset d2 and removes all edges.
        Both models are evaluated on their testgraph-subset.
        """
        # Baseline 2_distances - Train on traingraph (d1) - Test on testgraph (d2)
        attack_name = f'baseline_2_dist_{d1}_{d2}'
        print_attack_start(attack_name)

        # attacker
        self.attacker[attack_name] = Attacker(self.target, self.testgraph)
        self.attacker[attack_name].create_modified_graph(0) # delete all edges -> 0 percent survivors
        self.attacker[attack_name].sample_data_vector_distances(0.2, 0.4)
        self.attacker[attack_name].train()

        # evaluate baseline 2_distances
        self.update_parameter(d2)
        self.attacker[attack_name].target_model = self.target
        self.attacker[attack_name].graph = self.testgraph
        self.attacker[attack_name].create_modified_graph(0)
        self.attacker[attack_name].sample_data_vector_distances(0.2, 0.4)
        print_attack_done(attack_name)
        self.evaluate_attack(attack_name, self.testgraph, verbose=self.verbose)

    def surviving_edges_dist(self, survivors, d1, d2):
        """
        The target model in this attack is trained on the traingraph-subset of the original dataset.
        The attacker model samples its dataset on the testgraph-subset and removes almost all edges.
        Both models are evaluated on the testgraph-subset.
        """
        # Surviving Edges
        attack_name = f'surviving_edges_dist_{int(survivors*100)}p'
        print_attack_start(attack_name)

        # attacker
        self.attacker[attack_name] = Attacker(self.target, self.testgraph)
        self.attacker[attack_name].create_modified_graph(survivors) # delete all edges -> 0 percent survivors
        self.attacker[attack_name].sample_data_vector_distances(0.2, 0.4)
        self.attacker[attack_name].train()

        # evaluate baseline 2_distances
        self.update_parameter(d2)
        self.attacker[attack_name].target_model = self.target
        self.attacker[attack_name].graph = self.testgraph
        self.attacker[attack_name].create_modified_graph(survivors)
        self.attacker[attack_name].sample_data_vector_distances(0.2, 0.4)
        print_attack_done(attack_name)
        self.evaluate_attack(attack_name, self.testgraph, verbose=self.verbose)


def main(args):
    # Init
    print_init(args)

    # Datasets
    datasets = print_datasets(args.dataset)
    # GNNs
    gnns = print_gnns(args.gnn)

    # create Experiments
    experiments = []
    for gnn in gnns:
        for dataset in datasets:
            exp = Experiment(gnn, dataset, args.verbose)
            exp.initialize()
            experiments.append(exp)

    # run attacks
    for experiment in experiments:
        print(f'\n\n  [+] Run Attacks on {experiment.gnn_name} trained with {experiment.dataset_name}\n')
        experiment.baseline_1()
        experiment.baseline_2()
        experiment.surviving_edges(0.20)
        experiment.surviving_edges(0.40)
        experiment.surviving_edges(0.60)
        experiment.surviving_edges(0.80)

    # target trained on different domain than attacker uses for queries
    distance_experiments = []
    if len(datasets) > 1:
        for gnn in gnns:
            for d1 in datasets:
                for d2 in datasets:
                    if d1 != d2:
                        print(f'\n\n  [+] Run Attacks on {gnn} trained on {d1} while attacker was trained on {d2}\n')
                        exp = Experiment(gnn, d1, args.verbose)
                        exp.initialize()
                        exp.baseline_1_dist(d1, d2)
                        exp.baseline_2_dist(d1, d2)
                        exp.surviving_edges_dist(0.20, d1, d2)
                        exp.surviving_edges_dist(0.40, d1, d2)
                        exp.surviving_edges_dist(0.60, d1, d2)
                        exp.surviving_edges_dist(0.80, d1, d2)
                        distance_experiments.append(exp)

    # Conclude all results
    final_evaluation(experiments, log=args.log, clear=args.clear)
    final_evaluation_distances(distance_experiments, log=args.log)


if __name__ == '__main__':

    # cmd args
    parser = argparse.ArgumentParser(description='Link-Stealing Attack')

    parser.add_argument("--log",
                        action="store_true",
                        help="Log Ouput in ./log/lineup.txt and ./log/results.json")

    parser.add_argument("--clear",
                        action="store_true",
                        help="Clear all files in ./log/")

    parser.add_argument("--verbose",
                        action="store_true",
                        help="Output Attack results")

    parser.add_argument("--dataset",
                        type=str,
                        default="all",
                        help="Select Dataset [all, cora, citeseer, pubmed]")

    parser.add_argument("--gnn",
                        type=str,
                        default='all',
                        help="Type of GNN [all, graphsage, gcn, gat]")


    args = parser.parse_args()
    main(args)

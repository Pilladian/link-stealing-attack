# Python 3.8.5

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Imports
import copy
import json
import torch
import random
import dgl
import torch.nn.functional as F
from src.graphsage import GraphSAGE
from src.mlp import MLP
from src.gat import GAT
from src.gcn import GCN
import time
from scipy.spatial import distance
from src.utils import load_data, load_graphs, print_attack_results, print_attack_start, print_attack_done

# ---------------------------------------------------------------------------------------------

# Metrics implementation
def _stats(pred, labels):
    pos_mask = labels == 1
    neg_mask = labels == 0

    tp = torch.sum(pred[pos_mask] == labels[pos_mask])
    tn = torch.sum(pred[neg_mask] == labels[neg_mask])

    fp = torch.sum(pred[pos_mask] != labels[pos_mask])
    fn = torch.sum(pred[neg_mask] != labels[neg_mask])

    return tp, tn, fp, fn

def _precision(pred, labels):
    tp, tn, fp, fn = _stats(pred, labels)
    return tp / (tp + fp)

def _recall(pred, labels):
    tp, tn, fp, fn = _stats(pred, labels)
    return tp / (tp + fn)

def _f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

def _accuracy(pred, labels):
    return torch.sum(pred == labels).item() * 1.0 / len(labels)

# ---------------------------------------------------------------------------------------------

# Target implementation
class Target:

    def __init__(self, gnn, graph, num_classes):
        self.gnn_name = gnn
        self.graph = copy.deepcopy(graph)
        self.num_classes = num_classes
        self._load_parameter()

    def _load_parameter(self):
        # read parameter from file
        with open(f'config/{self.gnn_name}.conf') as json_file:
            self.parameter = json.load(json_file)
            self.gpu = True if self.parameter['gpu'] > 0 else False

    def train(self):
        # initialize model
        self._initialize()

        for epoch in range(self.parameter['n_epochs']):
            self.model.train()
            # forward
            if self.gnn_name == 'graphsage':
                logits = self.model(self.graph, self.graph.ndata['feat'])
            elif self.gnn_name in ['gat', 'gcn']:
                g = dgl.remove_self_loop(self.graph)
                self.model.g = dgl.add_self_loop(g)
                logits = self.model(self.graph.ndata['feat'])
            loss = F.cross_entropy(logits, self.graph.ndata['label'])
            # update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _initialize(self):
        # GPU
        if self.gpu:
            torch.cuda.set_device(self.gpu)
            self.traingraph.cuda()

        # create model
        if self.gnn_name == 'graphsage':
            self.model = GraphSAGE(
                            self.graph.ndata['feat'].shape[1],
                            self.parameter['n_hidden'],
                            self.num_classes,
                            self.parameter['n_layers'],
                            F.relu,
                            self.parameter['dropout'],
                            self.parameter['aggregator_type'])

        elif self.gnn_name == 'gat':
            heads = ([self.parameter['n_heads']] * self.parameter['n_layers']) + [self.parameter['n_outheads']]
            self.model = GAT(
                            self.graph,
                            self.parameter['n_layers'],
                            self.graph.ndata['feat'].shape[1],
                            self.parameter['n_hidden'],
                            self.num_classes,
                            heads,
                            F.elu,
                            self.parameter['dropout'],
                            self.parameter['dropout'],
                            self.parameter['negative_slope'],
                            self.parameter['residual'])

        elif self.gnn_name == 'gcn':
            self.model = GCN(
                            self.graph,
                            self.graph.ndata['feat'].shape[1],
                            self.parameter['n_hidden'],
                            self.num_classes,
                            self.parameter['n_layers'],
                            F.relu,
                            self.parameter['dropout'])

        # load model to gpu
        if self.gpu:
            self.model.cuda()

        # optimizer
        self.optimizer = torch.optim.Adam(
                                        self.model.parameters(),
                                        lr=self.parameter['lr'],
                                        weight_decay=self.parameter['weight_decay'])

    def evaluate(self, graph):
        if self.gpu:
            graph.cuda()

        self.model.eval()
        with torch.no_grad():
            # query model
            if self.gnn_name == 'graphsage':
                logits = self.model(graph, graph.ndata['feat'])
            elif self.gnn_name in ['gat', 'gcn']:
                g = dgl.remove_self_loop(graph)
                self.model.g = dgl.add_self_loop(g)
                logits = self.model(graph.ndata['feat'])
            labels = graph.ndata['label']
            _, pred = torch.max(logits, dim=1)

            # accuracy
            return _accuracy(pred, labels)

    def get_posteriors(self, graph, id):
        if self.gpu:
            graph.cuda()

        self.model.eval()
        with torch.no_grad():
            # query model
            if self.gnn_name == 'graphsage':
                logits = self.model(graph, graph.ndata['feat'])
            elif self.gnn_name in ['gat', 'gcn']:
                g = dgl.remove_self_loop(graph)
                self.model.g = dgl.add_self_loop(g)
                logits = self.model(graph.ndata['feat'])
            logits = logits[id]
            # return posteriors predicted by the model
            return logits

# ---------------------------------------------------------------------------------------------

# Attacker implementation
class Attacker:

    def __init__(self, target, graph):
        self.target_model = target
        self.graph = copy.deepcopy(graph)
        self._load_parameter()

    def _load_parameter(self):
        # read parameter from file
        with open(f'config/mlp.conf') as json_file:
            self.parameter = json.load(json_file)
            self.gpu = True if self.parameter['gpu'] > 0 else False

    def create_modified_graph(self, survivors):
        # modified graph
        self.modified_graph = dgl.remove_self_loop(copy.deepcopy(self.graph))
        orig_num_of_edges = self.modified_graph.number_of_edges()

        # pos_samples - edges that existed in graph
        pos = []
        for p in range(int(orig_num_of_edges * (1 - survivors))):
            edge_id = random.randint(0, self.modified_graph.num_edges() - 1)
            src, dst = self.modified_graph.find_edges([edge_id])[0].item(), self.modified_graph.find_edges([edge_id])[1].item()
            self.modified_graph.remove_edges([self.modified_graph.edge_id(src, dst)])
            pos.append(((src, dst), True))

        # neg_samples - edges that do not exist in (modified_)graph
        neg = []
        for n in range(int(orig_num_of_edges * (1 - survivors))):
            src, dst = random.randint(0, self.graph.num_nodes() - 1), random.randint(0, self.graph.num_nodes() - 1)
            while self.graph.has_edges_between(src, dst) and (src, dst) not in neg and src != dst:
                src, dst = random.randint(0, self.graph.num_nodes() - 1), random.randint(0, self.graph.num_nodes() - 1)
            neg.append(((src, dst), False))

        # create raw dataset
        self.raw_dataset = pos + neg
        random.shuffle(self.raw_dataset)

    def sample_data_posteriors(self, train_val, val_test):
        """
        [1] Query target on two nodes
        [2] Get their posteriors for node classification
        [3] Concatenate posteriors to form the input feature vector for the attacker model
        """
        size = len(self.raw_dataset) - 1
        self.feature_amount = self.target_model.get_posteriors(self.modified_graph, 0).shape[0] * 2
        self.features = torch.zeros((size + 1, self.feature_amount), dtype=torch.float)
        self.labels = torch.zeros(size + 1, dtype=torch.long)

        for i, ((src, dst), label) in enumerate(self.raw_dataset):
            # query target model to get posteriors
            post_src = self.target_model.get_posteriors(self.modified_graph, src)
            post_dst = self.target_model.get_posteriors(self.modified_graph, dst)
            feature = torch.cat((post_src, post_dst))
            # create feature and label
            self.features[i] = feature
            self.labels[i] = torch.ones(1, 1) if label else torch.zeros(1, 1)

        # train, eval, test
        self.train_mask = torch.zeros(size, dtype=torch.bool)
        self.val_mask = torch.zeros(size, dtype=torch.bool)
        self.test_mask = torch.zeros(size, dtype=torch.bool)

        train_val_split = int(size * train_val)
        val_test_split  = train_val_split + int(size * val_test)

        # train masks
        for a in range(size):
            self.train_mask[a] = a < train_val_split
            self.val_mask[a] = a >= train_val_split and a < val_test_split
            self.test_mask[a] = a >= val_test_split

        # node ids
        self.train_nid = self.train_mask.nonzero().squeeze()
        self.val_nid = self.val_mask.nonzero().squeeze()
        self.test_nid = self.test_mask.nonzero().squeeze()

    def sample_data_vector_distances(self, train_val, val_test):
        """
        [1] Query target on two nodes
        [2] Get their posteriors for node classification
        [3] Create input feature vector for attacker model
            [3.1] Calculate distances between posterior vectors
            [3.2] Concatenate results to form a 8-dim input feature vector
        """
        size = len(self.raw_dataset) - 1
        self.feature_amount = 8
        self.features = torch.zeros((size + 1, self.feature_amount), dtype=torch.float)
        self.labels = torch.zeros(size + 1, dtype=torch.long)

        for i, ((src, dst), label) in enumerate(self.raw_dataset):
            # query target model to get posteriors
            post_src = self.target_model.get_posteriors(self.modified_graph, src)
            post_dst = self.target_model.get_posteriors(self.modified_graph, dst)
            src_list = [x.item() for x in post_src]
            dst_list = [x.item() for x in post_dst]

            # cosine distance
            # definition: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html
            cosine_distance = distance.cosine(src_list, dst_list)

            # euclidean distance
            # definition: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html?highlight=euclidean#scipy.spatial.distance.euclidean
            euclidean_distance = distance.euclidean(src_list, dst_list)

            # correlation distance
            # definition: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.correlation.html?highlight=correlation#scipy.spatial.distance.correlation
            correlation_distance = distance.correlation(src_list, dst_list)

            # chebyshev distance
            # definition https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.chebyshev.html?highlight=chebyshev#scipy.spatial.distance.chebyshev
            chebyshev_distance = distance.chebyshev(src_list, dst_list)

            # braycurtis distance
            # definition: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.braycurtis.html?highlight=braycurtis#scipy.spatial.distance.braycurtis
            braycurtis_distance = distance.braycurtis(src_list, dst_list)

            # manhattan distance
            # definition: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cityblock.html?highlight=manhattan
            manhattan_distance = distance.cityblock(src_list, dst_list)

            # canberra distance
            # definition: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.canberra.html?highlight=canberra#scipy.spatial.distance.canberra
            canberra_distance = distance.canberra(src_list, dst_list)

            # sqeuclidean distance
            # definition: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.sqeuclidean.html?highlight=sqeuclidean#scipy.spatial.distance.sqeuclidean
            sqeuclidean_distance = distance.sqeuclidean(src_list, dst_list)

            feature = torch.tensor([cosine_distance,
                                    euclidean_distance,
                                    correlation_distance,
                                    chebyshev_distance,
                                    braycurtis_distance,
                                    manhattan_distance,
                                    canberra_distance,
                                    sqeuclidean_distance])
            # create feature and label
            self.features[i] = feature
            self.labels[i] = torch.ones(1, 1) if label else torch.zeros(1, 1)

        # train, eval, test
        self.train_mask = torch.zeros(size, dtype=torch.bool)
        self.val_mask = torch.zeros(size, dtype=torch.bool)
        self.test_mask = torch.zeros(size, dtype=torch.bool)

        train_val_split = int(size * train_val)
        val_test_split  = train_val_split + int(size * val_test)

        # train masks
        for a in range(size):
            self.train_mask[a] = a < train_val_split
            self.val_mask[a] = a >= train_val_split and a < val_test_split
            self.test_mask[a] = a >= val_test_split

        # node ids
        self.train_nid = self.train_mask.nonzero().squeeze()
        self.val_nid = self.val_mask.nonzero().squeeze()
        self.test_nid = self.test_mask.nonzero().squeeze()

    def train(self):
        # initialize model
        self._initialize()

        dur = []
        for epoch in range(self.parameter['n_epochs']):
            self.model.train()
            t0 = time.time()

            # forward
            logits = self.model(self.features)
            loss = F.cross_entropy(logits[self.train_nid], self.labels[self.train_nid])

            # update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            dur.append(time.time() - t0)

    def _initialize(self):
        # create model
        self.model = MLP(self.feature_amount,
                         self.parameter['n_hidden'],
                         2,
                         self.parameter['n_layers'],
                         F.relu,
                         self.parameter['dropout'])

        # load model to gpu
        if self.gpu:
            torch.cuda.set_device(self.parameter['gpu'])
            self.model.cuda()

        # optimizer
        self.optimizer = torch.optim.Adam(
                                        self.model.parameters(),
                                        lr=self.parameter['lr'],
                                        weight_decay=self.parameter['weight_decay'])

    def evaluate(self, nid):
        self.model.eval()
        with torch.no_grad():
            # query model
            logits = self.model(self.features)
            logits = logits[nid]
            labels = self.labels[nid]
            _, pred = torch.max(logits, dim=1)

            # metrices
            precision = _precision(pred, labels)
            recall = _recall(pred, labels)
            f1_score = _f1_score(precision, recall)
            acc = _accuracy(pred, labels)

            return (precision, recall, f1_score, acc)

# ---------------------------------------------------------------------------------------------

# Experiment implementation
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
        traingraph, _ = load_graphs(f'{self.dir}traingraph.bin', [0])
        testgraph, _ = load_graphs(f'{self.dir}testgraph.bin', [0])

        self.traingraph = traingraph[0]
        self.testgraph = testgraph[0]

        # load model
        self.target = Target(self.gnn_name, self.traingraph, self.dataset.num_classes)
        self.target._initialize()
        self.target.model.load_state_dict(torch.load(f'{self.dir}model.pt'))

    def _load_dataset_subgraphs(self):
        self.dir = f'./models/{self.dataset_name}/{self.gnn_name}/'
        traingraph, _ = load_graphs(f'{self.dir}traingraph.bin', [0])
        testgraph, _ = load_graphs(f'{self.dir}testgraph.bin', [0])

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

    # Same Domain Attacks
    # Attack 1 : posteriors as features
    def baseline_train_same_domain_post(self):
        # Baseline Train Same Domain Posteriors : Train on traingraph - Test on traingraph
        attack_name = f'baseline_train_same_domain_post'
        print_attack_start(attack_name)

        # attacker
        self.attacker[attack_name] = Attacker(self.target, self.traingraph)
        self.attacker[attack_name].create_modified_graph(0) # delete all edges -> 0 percent survivors
        self.attacker[attack_name].sample_data_posteriors(0.2, 0.4)
        self.attacker[attack_name].train()

        # evaluate
        print_attack_done(attack_name)
        self.evaluate_attack(attack_name, self.testgraph, verbose=self.verbose)

    def baseline_test_same_domain_post(self):
        # Baseline Test Same Domain Posteriors : Train on traingraph - Test on testgraph
        attack_name = 'baseline_test_same_domain_post'
        print_attack_start(attack_name)

        # attacker
        self.attacker[attack_name] = Attacker(self.target, self.testgraph)
        self.attacker[attack_name].create_modified_graph(0) # delete all edges -> 0 percent survivors
        self.attacker[attack_name].sample_data_posteriors(0.2, 0.4)
        self.attacker[attack_name].train()

        # evaluate
        print_attack_done(attack_name)
        self.evaluate_attack(attack_name, self.testgraph, verbose=self.verbose)

    def surviving_edges_same_domain_post(self, survivors):
        # Surviving Edges Same Domain Posteriors
        attack_name = f'surviving_edges_{int(survivors*100)}p_same_domain_post'
        print_attack_start(attack_name)

        # attacker
        self.attacker[attack_name] = Attacker(self.target, self.traingraph)
        self.attacker[attack_name].create_modified_graph(survivors)
        self.attacker[attack_name].sample_data_posteriors(0.2, 0.4)
        self.attacker[attack_name].train()

        # evaluate
        print_attack_done(attack_name)
        self.evaluate_attack(attack_name, self.testgraph, verbose=self.verbose)

    # Attack 2 : distances as features
    def baseline_train_same_domain_dist(self):
        # Baseline Train Same Domain Distances : Train on traingraph - Test on traingraph
        attack_name = f'baseline_train_same_domain_dist'
        print_attack_start(attack_name)

        # attacker
        self.attacker[attack_name] = Attacker(self.target, self.traingraph)
        self.attacker[attack_name].create_modified_graph(0) # delete all edges -> 0 percent survivors
        self.attacker[attack_name].sample_data_vector_distances(0.2, 0.4)
        self.attacker[attack_name].train()

        # evaluate
        print_attack_done(attack_name)
        self.evaluate_attack(attack_name, self.testgraph, verbose=self.verbose)

    def baseline_test_same_domain_dist(self):
        # Baseline Test Same Domain Distances : Train on traingraph - Test on testgraph
        attack_name = 'baseline_test_same_domain_dist'
        print_attack_start(attack_name)

        # attacker
        self.attacker[attack_name] = Attacker(self.target, self.testgraph)
        self.attacker[attack_name].create_modified_graph(0) # delete all edges -> 0 percent survivors
        self.attacker[attack_name].sample_data_vector_distances(0.2, 0.4)
        self.attacker[attack_name].train()

        # evaluate
        print_attack_done(attack_name)
        self.evaluate_attack(attack_name, self.testgraph, verbose=self.verbose)

    def surviving_edges_same_domain_dist(self, survivors):
        # Surviving Edges Same Domain Distances
        attack_name = f'surviving_edges_{int(survivors*100)}p_same_domain_dist'
        print_attack_start(attack_name)

        # attacker
        self.attacker[attack_name] = Attacker(self.target, self.traingraph)
        self.attacker[attack_name].create_modified_graph(survivors)
        self.attacker[attack_name].sample_data_vector_distances(0.2, 0.4)
        self.attacker[attack_name].train()

        # evaluate
        print_attack_done(attack_name)
        self.evaluate_attack(attack_name, self.testgraph, verbose=self.verbose)

    # Different Domain Attacks - d1 = DA and d2 = Dft
    # Attack 3 : distances as features
    def baseline_train_diff_domain_dist(self, d1, d2):
        # Baseline 1_distances - Train on traingraph (d1) - Test on traingraph (d2)
        attack_name = f'baseline_train_{d1}_{d2}_diff_domain_dist'
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

    def baseline_test_diff_domain_dist(self, d1, d2):
        # Baseline 2_distances - Train on traingraph (d1) - Test on testgraph (d2)
        attack_name = f'baseline_test_{d1}_{d2}_diff_domain_dist'
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

    def surviving_edges_diff_domain_dist(self, survivors, d1, d2):
        # Surviving Edges
        attack_name = f'surviving_edges_{int(survivors*100)}p_{d1}_{d2}_diff_domain_dist'
        print_attack_start(attack_name)

        # attacker
        self.attacker[attack_name] = Attacker(self.target, self.traingraph)
        self.attacker[attack_name].create_modified_graph(survivors)
        self.attacker[attack_name].sample_data_vector_distances(0.2, 0.4)
        self.attacker[attack_name].train()

        # evaluate baseline 2_distances
        self.update_parameter(d2)
        self.attacker[attack_name].target_model = self.target
        self.attacker[attack_name].graph = self.traingraph
        self.attacker[attack_name].create_modified_graph(survivors)
        self.attacker[attack_name].sample_data_vector_distances(0.2, 0.4)
        print_attack_done(attack_name)
        self.evaluate_attack(attack_name, self.traingraph, verbose=self.verbose)

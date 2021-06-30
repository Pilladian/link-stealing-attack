import numpy as np
import matplotlib.pyplot as plt

# Attack 1
def a1_cora():

    plt.figure(figsize=(12,6))

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [70.028, 77.35, 76.45, 78.30, 76.39]
    gat = [64.90, 76.72, 77.11, 75.04, 72.86]
    gcn = [74.08, 73.40, 74.63, 72.69, 64.50]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.115
    y_size = 0.5
    plt.bar(X_axis - size, graphsage, size, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    plt.title("Attack 1 - Cora")
    plt.ylim([0, 100])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-1-cora.png')

def a1_citeseer():

    plt.figure(figsize=(12,6))

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [76.06, 78.47, 78.42, 81.07, 78.84]
    gat = [74.40, 77.89, 80.64, 77.94, 76.52]
    gcn = [72.80, 74.54, 76.03, 76.23, 74.61]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.115
    y_size = 0.5
    plt.bar(X_axis - size, graphsage, size, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    plt.title("Attack 1 - CiteSeer")
    plt.ylim([0, 100])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-1-citeseer.png')

def a1_pubmed():

    plt.figure(figsize=(12,6))

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [0, 0, 0, 0, 0]
    gat = [0, 0, 0, 0, 0]
    gcn = [0, 0, 0, 0, 0]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.115
    y_size = 0.5
    plt.bar(X_axis - size, graphsage, size, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    plt.title("Attack 1 - Pubmed")
    plt.ylim([0, 100])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-1-pubmed.png')

a1_cora()
a1_citeseer()
# a1_pubmed()

# ------------------------------------------------------------------------------

# Attack 2
def a2_cora():

    plt.figure(figsize=(12,6))

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [77.28, 78.30, 80.56, 83.15, 83.34]
    gat = [79.32, 80.20, 81.17, 83.58, 85.05]
    gcn = [69.96, 72.87, 73.09, 74.72, 73.11]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.115
    y_size = 0.5
    plt.bar(X_axis - size, graphsage, size, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    plt.title("Attack 2 - Cora")
    plt.ylim([0, 100])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-2-cora.png')

def a2_citeseer():

    plt.figure(figsize=(12,6))

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [82.80, 82.25, 84.41, 85.56, 87.74]
    gat = [83.55, 84.727, 86.90, 87.25, 89.09]
    gcn = [73.84, 73.01, 74.83, 74.86, 75.02]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.115
    y_size = 0.5
    plt.bar(X_axis - size, graphsage, size, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    plt.title("Attack 2 - CiteSeer")
    plt.ylim([0, 100])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-2-citeseer.png')

def a2_pubmed():

    plt.figure(figsize=(12,6))

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [0, 0, 0, 0, 0]
    gat = [0, 0, 0, 0, 0]
    gcn = [0, 0, 0, 0, 0]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.115
    y_size = 0.5
    plt.bar(X_axis - size, graphsage, size, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    plt.title("Attack 2 - Pubmed")
    plt.ylim([0, 100])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-2-pubmed.png')

a2_cora()
a2_citeseer()
# a2_pubmed()

# ------------------------------------------------------------------------------

# Attack 3
def a3_cora_citeseer():

    plt.figure(figsize=(12,6))

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [0, 0, 0, 0, 0]
    gat = [0, 0, 0, 0, 0]
    gcn = [0, 0, 0, 0, 0]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.115
    y_size = 0.5
    plt.bar(X_axis - size, graphsage, size, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    plt.title("Attack 3 - Cora (Target) - CiteSeer (Attacker)")
    plt.ylim([0, 100])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-3-cora-citeseer.png')
def a3_cora_pubmed():

    plt.figure(figsize=(12,6))

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [0, 0, 0, 0, 0]
    gat = [0, 0, 0, 0, 0]
    gcn = [0, 0, 0, 0, 0]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.115
    y_size = 0.5
    plt.bar(X_axis - size, graphsage, size, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    plt.title("Attack 3 - Cora (Target) - Pubmed (Attacker)")
    plt.ylim([0, 100])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-3-cora-pubmed.png')
def a3_citeseer_cora():

    plt.figure(figsize=(12,6))

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [0, 0, 0, 0, 0]
    gat = [0, 0, 0, 0, 0]
    gcn = [0, 0, 0, 0, 0]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.115
    y_size = 0.5
    plt.bar(X_axis - size, graphsage, size, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    plt.title("Attack 3 - CiteSeer (Target) - Cora (Attacker)")
    plt.ylim([0, 100])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-3-citeseer-cora.png')
def a3_citeseer_pubmed():

    plt.figure(figsize=(12,6))

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [0, 0, 0, 0, 0]
    gat = [0, 0, 0, 0, 0]
    gcn = [0, 0, 0, 0, 0]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.115
    y_size = 0.5
    plt.bar(X_axis - size, graphsage, size, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    plt.title("Attack 3 - CiteSeer (Target) - Pubmed (Attacker)")
    plt.ylim([0, 100])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-3-citeseer-pubmed.png')
def a3_pubmed_cora():

    plt.figure(figsize=(12,6))

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [0, 0, 0, 0, 0]
    gat = [0, 0, 0, 0, 0]
    gcn = [0, 0, 0, 0, 0]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.115
    y_size = 0.5
    plt.bar(X_axis - size, graphsage, size, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    plt.title("Attack 3 - Pubmed (Target) - Cora (Attacker)")
    plt.ylim([0, 100])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-3-pubmed-cora.png')
def a3_pubmed_citeseer():

    plt.figure(figsize=(12,6))

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [0, 0, 0, 0, 0]
    gat = [0, 0, 0, 0, 0]
    gcn = [0, 0, 0, 0, 0]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.115
    y_size = 0.5
    plt.bar(X_axis - size, graphsage, size, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    plt.title("Attack 3 - Pubmed (Target) - CiteSeer (Attacker)")
    plt.ylim([0, 100])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-3-pubmed-citeseer.png')

#a3_cora_citeseer()
#a3_cora_pubmed()
#a3_citeseer_cora()
#a3_citeseer_pubmed()
#a3_pubmed_cora()
#a3_pubmed_citeseer()

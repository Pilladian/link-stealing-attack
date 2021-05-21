import numpy as np
import matplotlib.pyplot as plt

# Attack 1
def a1_cora():

    plt.figure(figsize=(12,6))

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [68.36, 73.68, 77.08, 74.90, 75.76]
    gat = [65.29, 73.18, 75.36, 75.44, 76.70]
    gcn = [68.07, 69.19, 65.43, 75.65, 56.68]

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

    graphsage = [77.38, 75.01, 80.64, 80.74, 81.26]
    gat = [73.65, 73.90, 77.87, 79.09, 78.62]
    gcn = [73.27, 74.45, 75.29, 75.20, 63.87]

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

    graphsage = [73.39, 75.73, 76.02, 75.52, 77.57]
    gat = [73.42, 73.40, 73.49, 74.94, 74.96]
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
a1_pubmed()

# ------------------------------------------------------------------------------

# Attack 2
def a2_cora():

    plt.figure(figsize=(12,6))

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [72.06, 73.29, 80.00, 82.59, 80.24]
    gat = [78.06, 82.36, 82.93, 87.46, 87.92]
    gcn = [65.41, 67.21, 72.80, 69.29, 74.33]

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

    graphsage = [78.94, 81.69, 82.85, 84.80, 89.07]
    gat = [78.06, 82.36, 82.93, 87.46, 87.92]
    gcn = [71.27, 75.62, 74.45, 74.49, 80.00]

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

    graphsage = [75.47, 79.37, 80.98, 81.72, 83.21]
    gat = [77.30, 79.63, 80.35, 82.73, 82.37]
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
a2_pubmed()

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

a3_cora_citeseer()
a3_cora_pubmed()
a3_citeseer_cora()
a3_citeseer_pubmed()
a3_pubmed_cora()
a3_pubmed_citeseer()

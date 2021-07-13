import numpy as np
import matplotlib.pyplot as plt

# Attack 1
def a1_cora():

    plt.figure(figsize=(12,6))
    plt.grid(axis = 'y', zorder=0)
    
    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [0.70028, 0.7735, 0.7645, 0.7830, 0.7639]
    gat = [0.6490, 0.7672, 0.7711, 0.7504, 0.7286]
    gcn = [0.7408, 0.7340, 0.7463, 0.7269, 0.6450]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.1
    y_size = 0.005
    plt.bar(X_axis - size, graphsage, size, zorder = 3, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, zorder = 3, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, zorder = 3, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    #plt.title("Attack 1 - Cora")
    plt.ylim([0, 1.1])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-1-cora.png')

def a1_citeseer():

    plt.figure(figsize=(12,6))
    plt.grid(axis = 'y', zorder=0)

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [0.7606, 0.7847, 0.7842, 0.8107, 0.7884]
    gat = [0.7440, 0.7789, 0.8064, 0.7794, 0.7652]
    gcn = [0.7280, 0.7454, 0.7603, 0.7623, 0.7461]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.1
    y_size = 0.005
    plt.bar(X_axis - size, graphsage, size, zorder = 3, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, zorder = 3, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, zorder = 3, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    #plt.title("Attack 1 - CiteSeer")
    plt.ylim([0, 1.1])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-1-citeseer.png')

def a1_pubmed():

    plt.figure(figsize=(12,6))
    plt.grid(axis = 'y', zorder=0)

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [0.7457, 0.7451, 0.7546, 0.7553, 0.7482]
    gat = [0.7534, 0.7536, 0.7541, 0.7525, 0.7698]
    gcn = [0.73173, 0.75828, 0.70828, 0.5836, 0.41295]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.1
    y_size = 0.005
    plt.bar(X_axis - size, graphsage, size, zorder = 3, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, zorder = 3, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, zorder = 3, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    #plt.title("Attack 1 - Pubmed")
    plt.ylim([0, 1.1])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-1-pubmed.png')

a1_cora()
a1_citeseer()
a1_pubmed()

# ------------------------------------------------------------------------------

# Attack 2
def a2_cora():

    plt.figure(figsize=(12,6))
    plt.grid(axis = 'y', zorder=0)

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [0.7728, 0.7830, 0.8056, 0.8315, 0.8334]
    gat = [0.7932, 0.8020, 0.8117, 0.8358, 0.8505]
    gcn = [0.6996, 0.7287, 0.7309, 0.7472, 0.7311]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.1
    y_size = 0.005
    plt.bar(X_axis - size, graphsage, size, zorder = 3, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, zorder = 3, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, zorder = 3, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    #plt.title("Attack 2 - Cora")
    plt.ylim([0, 1.1])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-2-cora.png')

def a2_citeseer():

    plt.figure(figsize=(12,6))
    plt.grid(axis = 'y', zorder=0)

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [0.8280, 0.8225, 0.8441, 0.8556, 0.8774]
    gat = [0.8355, 0.84727, 0.8690, 0.8725, 0.8909]
    gcn = [0.7384, 0.7301, 0.7483, 0.7486, 0.7502]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.1
    y_size = 0.005
    plt.bar(X_axis - size, graphsage, size, zorder = 3, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, zorder = 3, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, zorder = 3, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    #plt.title("Attack 2 - CiteSeer")
    plt.ylim([0, 1.1])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-2-citeseer.png')

def a2_pubmed():

    plt.figure(figsize=(12,6))
    plt.grid(axis = 'y', zorder=0)

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [0.7607, 0.7764, 0.7986, 0.8101, 0.8248]
    gat = [0.7595, 0.7806, 0.7972, 0.8131, 0.8134]
    gcn = [0.75572, 0.76629, 0.781149, 0.79757, 0.81162]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.1
    y_size = 0.005
    plt.bar(X_axis - size, graphsage, size, zorder = 3, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, zorder = 3, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, zorder = 3, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    #plt.title("Attack 2 - Pubmed")
    plt.ylim([0, 1.1])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-2-pubmed.png')

a2_cora()
a2_citeseer()
a2_pubmed()

# ------------------------------------------------------------------------------

# Attack 3
def a3_cora_citeseer():

    plt.figure(figsize=(12,6))
    plt.grid(axis = 'y', zorder=0)

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [0.8141, 0.8127, 0.8515, 0.8637, 0.8793]
    gat = [0.8289, 0.8414, 0.8656, 0.8798, 0.8854]
    gcn = [0.6912, 0.7247, 0.7322, 0.7472, 0.7630]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.1
    y_size = 0.005
    plt.bar(X_axis - size, graphsage, size, zorder = 3, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, zorder = 3, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, zorder = 3, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    #plt.title("Attack 3 - Cora (Attacker) - CiteSeer (Target)")
    plt.ylim([0, 1.10])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-3-cora-citeseer.png')
def a3_cora_pubmed():

    plt.figure(figsize=(12,6))
    plt.grid(axis = 'y', zorder=0)

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [0.7417, 0.7795, 0.7988, 0.8132, 0.8257]
    gat = [0.7562, 0.7801, 0.7991, 0.8074, 0.8142]
    gcn = [0.6890, 0.7721, 0.7828, 0.7931, 0.8085]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.1
    y_size = 0.005
    plt.bar(X_axis - size, graphsage, size, zorder = 3, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, zorder = 3, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, zorder = 3, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    #plt.title("Attack 3 - Cora (Attacker) - Pubmed (Target)")
    plt.ylim([0, 1.10])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-3-cora-pubmed.png')
def a3_citeseer_cora():

    plt.figure(figsize=(12,6))
    plt.grid(axis = 'y', zorder=0)

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [0.7639, 0.7925, 0.8149, 0.8246, 0.8543]
    gat = [0.7810, 0.8100, 0.8217, 0.8326, 0.8373]
    gcn = [0.7395, 0.7217, 0.7374, 0.7574, 0.7195]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.1
    y_size = 0.005
    plt.bar(X_axis - size, graphsage, size, zorder = 3, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, zorder = 3, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, zorder = 3, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    #plt.title("Attack 3 - CiteSeer (Attacker) - Cora (Target)")
    plt.ylim([0, 1.10])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-3-citeseer-cora.png')
def a3_citeseer_pubmed():

    plt.figure(figsize=(12,6))
    plt.grid(axis = 'y', zorder=0)

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [0.7476, 0.7809, 0.8020, 0.8164, 0.8205]
    gat = [0.7610, 0.7771, 0.7981, 0.7989, 0.8114]
    gcn = [0.6882, 0.7687, 0.7835, 0.7960, 0.8092]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.1
    y_size = 0.005
    plt.bar(X_axis - size, graphsage, size, zorder = 3, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, zorder = 3, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, zorder = 3, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    #plt.title("Attack 3 - CiteSeer (Attacker) - Pubmed (Target)")
    plt.ylim([0, 1.10])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-3-citeseer-pubmed.png')
def a3_pubmed_cora():

    plt.figure(figsize=(12,6))
    plt.grid(axis = 'y', zorder=0)

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [0.7082, 0.8040, 0.8130, 0.8402, 0.8349]
    gat = [0.7347, 0.7898, 0.8313, 0.8255, 0.8312]
    gcn = [0.6636, 0.7435, 0.7486, 0.7444, 0.7392]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.1
    y_size = 0.005
    plt.bar(X_axis - size, graphsage, size, zorder = 3, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, zorder = 3, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, zorder = 3, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    #plt.title("Attack 3 - Pubmed (Attacker) - Cora (Target)")
    plt.ylim([0, 1.10])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-3-pubmed-cora.png')
def a3_pubmed_citeseer():

    plt.figure(figsize=(12,6))
    plt.grid(axis = 'y', zorder=0)

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [0.8253, 0.8198, 0.8416, 0.8584, 0.8738]
    gat = [0.8383, 0.8440, 0.8609, 0.8724, 0.8804]
    gcn = [0.7587, 0.7320, 0.7442, 0.7620, 0.7694]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.1
    y_size = 0.005
    plt.bar(X_axis - size, graphsage, size, zorder = 3, label = 'GraphSAGE', color='tab:olive')
    for i, value in enumerate(graphsage):
        plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, zorder = 3, label = 'GAT')
    for i, value in enumerate(gat):
        plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, zorder = 3, label = 'GCN')
    for i, value in enumerate(gcn):
        plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

    plt.xticks(X_axis, bartypes)
    plt.xlabel("Percentage of Known Edges")
    plt.ylabel("Attack F1-Score")
    #plt.title("Attack 3 - Pubmed (Attacker) - CiteSeer (Target)")
    plt.ylim([0, 1.10])
    plt.legend(loc = 'upper left')
    plt.savefig('attack-3-pubmed-citeseer.png')

a3_cora_citeseer()
a3_cora_pubmed()
a3_citeseer_cora()
a3_citeseer_pubmed()
a3_pubmed_cora()
a3_pubmed_citeseer()

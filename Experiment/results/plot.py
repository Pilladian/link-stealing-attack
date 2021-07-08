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

# a1_cora()
# a1_citeseer()
# a1_pubmed()

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

# a2_cora()
# a2_citeseer()
# a2_pubmed()

# ------------------------------------------------------------------------------

# Attack 3
def a3_cora_citeseer():

    plt.figure(figsize=(12,6))
    plt.grid(axis = 'y', zorder=0)

    bartypes = ['0', '20', '40', '60', '80']

    graphsage = [0.7777, 0.8057, 0.8454, 0.8693, 0.8857]
    gat = [0.8300, 0.8478, 0.8628, 0.8601, 0.8827]
    gcn = [0.6750, 0.7333, 0.7386, 0.7373, 0.7735]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.1
    y_size = 0.005
    plt.bar(X_axis - size, graphsage, size, zorder = 3, label = 'GraphSAGE', color='tab:olive')
    #for i, value in enumerate(graphsage):
    #    plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, zorder = 3, label = 'GAT')
    #for i, value in enumerate(gat):
    #    plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, zorder = 3, label = 'GCN')
    #for i, value in enumerate(gcn):
    #    plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

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

    graphsage = [0.7544, 0.7801, 0.8032, 0.8045, 0.8231]
    gat = [0.7562, 0.7840, 0.8068, 0.8151, 0.8178]
    gcn = [0.6838, 0.7741, 0.7875, 0.7886, 0.8045]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.1
    y_size = 0.005
    plt.bar(X_axis - size, graphsage, size, zorder = 3, label = 'GraphSAGE', color='tab:olive')
    #for i, value in enumerate(graphsage):
    #    plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, zorder = 3, label = 'GAT')
    #for i, value in enumerate(gat):
    #    plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, zorder = 3, label = 'GCN')
    #for i, value in enumerate(gcn):
    #    plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

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

    graphsage = [0.7620, 0.7740, 0.8167, 0.8323, 0.8485]
    gat = [0.7771, 0.7860, 0.8355, 0.8406, 0.8279]
    gcn = [0.7322, 0.7121, 0.7213, 0.7567, 0.7058]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.1
    y_size = 0.005
    plt.bar(X_axis - size, graphsage, size, zorder = 3, label = 'GraphSAGE', color='tab:olive')
    #for i, value in enumerate(graphsage):
    #    plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, zorder = 3, label = 'GAT')
    #for i, value in enumerate(gat):
    #    plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, zorder = 3, label = 'GCN')
    #for i, value in enumerate(gcn):
    #    plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

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

    graphsage = [0.7494, 0.7798, 0.7961, 0.8218, 0.8106]
    gat = [0.7638, 0.7860, 0.7961, 0.7939, 0.8104]
    gcn = [0.6832, 0.7687, 0.7846, 0.7979, 0.8124]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.1
    y_size = 0.005
    plt.bar(X_axis - size, graphsage, size, zorder = 3, label = 'GraphSAGE', color='tab:olive')
    #for i, value in enumerate(graphsage):
    #    plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, zorder = 3, label = 'GAT')
    #for i, value in enumerate(gat):
    #    plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, zorder = 3, label = 'GCN')
    #for i, value in enumerate(gcn):
    #    plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

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

    graphsage = [0.7358, 0.8099, 0.8245, 0.8640, 0.8571]
    gat = [0.7192, 0.8075, 0.8159, 0.8336, 0.8358]
    gcn = [0.6561, 0.7287, 0.7507, 0.7279, 0.7145]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.1
    y_size = 0.005
    plt.bar(X_axis - size, graphsage, size, zorder = 3, label = 'GraphSAGE', color='tab:olive')
    #for i, value in enumerate(graphsage):
    #    plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, zorder = 3, label = 'GAT')
    #for i, value in enumerate(gat):
    #    plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, zorder = 3, label = 'GCN')
    #for i, value in enumerate(gcn):
    #    plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

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

    graphsage = [0.8283, 0.8360, 0.8278, 0.8612, 0.8837]
    gat = [0.8363, 0.8327, 0.8580, 0.8783, 0.8908]
    gcn = [0.7651, 0.7330, 0.7542, 0.7748, 0.7749]

    X_axis = np.arange(len(bartypes))

    size = 0.25
    lsize = 0.1
    y_size = 0.005
    plt.bar(X_axis - size, graphsage, size, zorder = 3, label = 'GraphSAGE', color='tab:olive')
    #for i, value in enumerate(graphsage):
    #    plt.text(X_axis[i] - size - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis, gat, size, zorder = 3, label = 'GAT')
    #for i, value in enumerate(gat):
    #    plt.text(X_axis[i] - lsize, value + y_size, s=f'{value:0.2f}')

    plt.bar(X_axis + size, gcn, size, zorder = 3, label = 'GCN')
    #for i, value in enumerate(gcn):
    #    plt.text(X_axis[i] + size  - lsize, value + y_size, s=f'{value:0.2f}')

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

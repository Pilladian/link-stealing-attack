# Attack 1
cora_graphsage = [0.70028, 0.7735, 0.7645, 0.7830, 0.7639]
cora_gat = [0.6490, 0.7672, 0.7711, 0.7504, 0.7286]
cora_gcn = [0.7408, 0.7340, 0.7463, 0.7269, 0.6450]

citeseer_graphsage = [0.7606, 0.7847, 0.7842, 0.8107, 0.7884]
citeseer_gat = [0.7440, 0.7789, 0.8064, 0.7794, 0.7652]
citeseer_gcn = [0.7280, 0.7454, 0.7603, 0.7623, 0.7461]

pubmed_graphsage = [0.7457, 0.7451, 0.7546, 0.7553, 0.7482]
pubmed_gat = [0.7534, 0.7536, 0.7541, 0.7525, 0.7698]
pubmed_gcn = [0.73173, 0.75828, 0.70828, 0.5836, 0.41295]

a = cora_graphsage + cora_gat + cora_gcn \
    + citeseer_graphsage + citeseer_gat + citeseer_gcn \
    + pubmed_graphsage + pubmed_gat + pubmed_gcn

# Attack 2
cora_graphsage = [0.7728, 0.7830, 0.8056, 0.8315, 0.8334]
cora_gat = [0.7932, 0.8020, 0.8117, 0.8358, 0.8505]
cora_gcn = [0.6996, 0.7287, 0.7309, 0.7472, 0.7311]

citeseer_graphsage = [0.8280, 0.8225, 0.8441, 0.8556, 0.8774]
citeseer_gat = [0.8355, 0.84727, 0.8690, 0.8725, 0.8909]
citeseer_gcn = [0.7384, 0.7301, 0.7483, 0.7486, 0.7502]

pubmed_graphsage = [0.7607, 0.7764, 0.7986, 0.8101, 0.8248]
pubmed_gat = [0.7595, 0.7806, 0.7972, 0.8131, 0.8134]
pubmed_gcn = [0.75572, 0.76629, 0.781149, 0.79757, 0.81162]

b = cora_graphsage + cora_gat + cora_gcn \
    + citeseer_graphsage + citeseer_gat + citeseer_gcn \
    + pubmed_graphsage + pubmed_gat + pubmed_gcn

#b = [0.7728, 0.7932, 0.6996, 0.8280, 0.8355, 0.7384, 0.7607, 0.7595, 0.75572]

print(float(sum(a) / len(a)))
print(float(sum(b) / len(b)))
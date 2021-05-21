import matplotlib.pyplot as plt


plt.clf()
bar_width = 0.8

attacks = ['Cora\nCiteseer', 'Cora\nPubmed',
           'Citeseer\nCora', 'Citeseer\nPubmed',
           'Pubmed\nCora', 'Pubmed\nCiteseer']

# GraphSAGE - Baseline 2
#values = [78.09, 75.90,
#          71.12, 76.49,
#          70.45, 76.94]

# GraphSAGE - Surviving Edges 80%
values = [91.31, 85.12,
          87.94, 85.97,
          88.17, 93.09]

pos = [i + bar_width for i, _ in enumerate(attacks)]

plt.ylabel('Attack F1-Score')
plt.title('GraphSAGE - Surviving Edges 80% - Different Datasets')
plt.ylim([0, 120])

# GNN Type centered between 3 bars
tick_pos = [val for val in pos]
plt.xticks(tick_pos, attacks)

# bars
colors = ['tab:olive', 'tab:olive', 'tab:purple', 'tab:purple', 'tab:orange', 'tab:orange']
for i, a in enumerate(attacks):
    plt.bar(pos[i], values[i], label=a, width=bar_width, color=colors[i])
    plt.text(x=i+0.52 , y =values[i]+1 , s=f"{values[i]}" , fontdict=dict(fontsize=10))


path = f"./graphsage_survining_edges_80p_diff.jpg"
plt.savefig(path)

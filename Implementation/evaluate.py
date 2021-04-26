# Python 3.8.5

import json
import matplotlib.pyplot as plt

def get_plot(data, name):
    bar_width = 0.1

    gnns = [type for type, _ in data.items()]
    values = dict()

    for dataset in list(data[list(data.keys())[0]].keys()):
        values[dataset] = []

    for type, cont in data.items():
        for i, (dataset, cont2) in enumerate(cont.items()):
            values[dataset].append(cont2[name]['attacker']['f1-score'])

    pos = []
    pos.append([i / 2 for i, _ in enumerate(gnns)])
    pos.append([val + bar_width for val in pos[0]])
    pos.append([val + bar_width for val in pos[1]])

    plt.xlabel('Graph Neural Network Type')
    plt.ylabel('Attack F1-Score')
    plt.title(name)
    plt.ylim([0, 1.2])

    # GNN Type centered between 3 bars
    tick_pos = [val + (bar_width * 2) / 2 for val in pos[0]]
    plt.xticks(tick_pos, gnns)

    # bars
    colors = ['tab:olive', 'tab:purple', 'tab:orange']
    for i, (dataset, vals) in enumerate(values.items()):
        plt.bar(pos[i], values[dataset], label=dataset, width=bar_width, color=colors[i])

    if name == 'baseline_1':
        plt.legend()
    path = f"./eval/plots/{name}.jpg"
    plt.savefig(path)

    return path[1:]

def get_results(data, name):
    content = """<table>
        <thead>
            <tr>
                <th>Type</th>
                <th>Dataset</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Accuracy</th>
            </tr>
        </thead>
        <tbody>"""

    dataset_amount = len(data[list(data.keys())[0]].keys())
    # target models
    for type, cont in data.items():
        f = f"""
        <tr>
            <td rowspan={dataset_amount}>{type}</td>"""
        for i, (dataset, cont2) in enumerate(cont.items()):
            if i == 0:
                f += f"""
            <td>{dataset}</td>
            <td>{cont2[name]['attacker']['prec']:0.4f}</td>
            <td>{cont2[name]['attacker']['recall']:0.4f}</td>
            <td>{cont2[name]['attacker']['f1-score']:0.4f}</td>
            <td>{cont2[name]['attacker']['acc']:0.4f}</td>
        </tr>"""
            else:
                f += f"""
        <tr>
            <td>{dataset}</td>
            <td>{cont2[name]['attacker']['prec']:0.4f}</td>
            <td>{cont2[name]['attacker']['recall']:0.4f}</td>
            <td>{cont2[name]['attacker']['f1-score']:0.4f}</td>
            <td>{cont2[name]['attacker']['acc']:0.4f}</td>
        </tr>"""
        content += f

    content += f"""
    </tbody>
    </table>"""

    return content

def main():
    # load json
    with open('./log/results.json') as json_file:
        data = json.load(json_file)

    # create markdown file
    content = """# Link Stealing Attacks - Evaluation

> Link Stealing Attacks on inductive trained Graph Neural Networks

## Target Models

#### Task
Given a graph with a few labeled nodes the model performs label prediction on the unlabeled ones. For calculation of the feature / embedding vectors, all model types calculate the embedding of node `i` based on their neighborhood (connected nodes).

#### Models and their Accuracy

<table>
    <thead>
        <tr>
            <th>Type</th>
            <th>Dataset</th>
            <th>Accuracy</th>
        </tr>
    </thead>
    <tbody>"""

    dataset_amount = len(data[list(data.keys())[0]].keys())
    # target models
    for type, cont in data.items():
        f = f"""
        <tr>
            <td rowspan={dataset_amount}>{type}</td>"""
        for i, (dataset, cont2) in enumerate(cont.items()):
            if i == 0:
                f += f"""
            <td>{dataset}</td>
            <td>{cont2['baseline_2']['target']['acc']:0.4f}</td>
        </tr>"""
            else:
                f += f"""
        <tr>
            <td>{dataset}</td>
            <td>{cont2['baseline_2']['target']['acc']:0.4f}</td>
        </tr>"""
        content += f

    content += f"""
</tbody>
</table>"""


    content += f"""

## Attacker model - Multi-Layer Perceptron

#### Task
Given a Graph Neural Network the model performs a link stealing attack based on the posteriors that are generated by the GNN.

#### Parameter
| Parameter     | Value
|------         |------
| Type          | MLP
| Epochs        | 200
| Hidden Layer  | 2
| Hidden Nodes  | 16
| Learning Rate | 0.01
| Dropout       | 0.5
| Optimizer     | Adam

#### Threat Model
- Black Box Access ( Query Access ) to Target Model `target`

#### Attack Methodology
> Example: Social Network like Instagram or Facebook

- Social Network
    - Nodes: People
    - Edges: Connection between people if they know each other


- Target Model `target` that has been trained on the Social Network to perform some Task
    - Input: Node ID
    - Output: Some posterior


- Create Raw Attacker Training Dataset `raw-train`
    - Collect `pos` ( node pairs of people that are connected )
    - Collect `neg` ( node pairs of people that are not connected )
    - <span style="color:green">Possible because public profiles reveal such information</span>
        - Private profile `private` follows Public profile `public`
            - `public` has `private` as follower
            - append (`private`, `public`, True) to `pos`
        - Private profile `private` does not follow Public Profile `public`
            - `public` has `private` not as follower
            - append (`private`, `public`, False) to `neg`


- Sample `attacker-train` with `raw-train`
    - Query `target` on both nodes
    - Get posteriors for both nodes
    - Concatinate the posteriors as feature
    - Use 1 (`pos`) or 0 (`neg`) as label


- Train Attacker Model `attacker`
    - Input: Posterior Concatination
    - Output: Prediction whether both nodes are connected or not

#### Approach

> Example: Social Network like Instagram or Facebook

Now it is possible to predict whether two private accounts are connected to each other or not. Since information like this are sensitive, this is a privacy breach.



## Attacks

> Example: Social Network like Instagram or Facebook

A GNN was trained on Instagram profiles to predict the salary of people. To train the  `attacker` one could use its own profile, its follower and also the follower of its own follower. The network now contains people that one is connected to and people one doesn't know.

#### Baseline 1
Use the Social Network Graph, the Target Model was trained on to also train the Attacker Model (<span style="color:red">Knowledge of the dataset needed</span>). Remove all edges but keep in mind, which nodes have been connected. Sample `pos` with nodes that have been connected. Sample `neg` with nodes that haven't. Query the GNN with the modified Social Network Graph to get posteriors to sample features. Train `attacker` on the sampled dataset.

Predict whether one knows people or not.

##### Results
{get_results(data, 'baseline_1')}
<img src=\"{get_plot(data, "baseline_1")}\" alt="drawing" width="520"/>

#### Baseline 2
Unfollow everybody but keep in mind, that one know them. Sample `pos` with one self and its former follower. Sample `neg` with one and accounts one doesn't know. Query the GNN with ones modified network to get posteriors to sampled features. Train `attacker` on the sampled dataset.

Predict whether one knows people or not.

##### Results
{get_results(data, 'baseline_2')}
<img src=\"{get_plot(data, "baseline_2")}\" alt="drawing" width="520"/>

#### Surviving Edges 20
Unfollow 80% but keep in mind, that one know them. Sample `pos` with one self, its former follower but also its remaining follower. Sample `neg` with one and accounts one doesn't know. Query the GNN with ones modified network to get posteriors to sampled features. Train `attacker` on the sampled dataset.

Predict whether one knows people or not.

##### Results
{get_results(data, 'surviving_edges_20p')}
<img src=\"{get_plot(data, "surviving_edges_20p")}\" alt="drawing" width="520"/>

#### Surviving Edges 40
Unfollow 60% but keep in mind, that one know them. Sample `pos` with one self, its former follower but also its remaining follower. Sample `neg` with one and accounts one doesn't know. Query the GNN with ones modified network to get posteriors to sampled features. Train `attacker` on the sampled dataset.

Predict whether one knows people or not.

##### Results
{get_results(data, 'surviving_edges_40p')}
<img src=\"{get_plot(data, "surviving_edges_40p")}\" alt="drawing" width="520"/>

#### Surviving Edges 60
Unfollow 40% but keep in mind, that one know them. Sample `pos` with one self, its former follower but also its remaining follower. Sample `neg` with one and accounts one doesn't know. Query the GNN with ones modified network to get posteriors to sampled features. Train `attacker` on the sampled dataset.

Predict whether one knows people or not.

##### Results
{get_results(data, 'surviving_edges_60p')}
<img src=\"{get_plot(data, "surviving_edges_60p")}\" alt="drawing" width="520"/>

#### Surviving Edges 80
Unfollow 20% but keep in mind, that one know them. Sample `pos` with one self, its former follower but also its remaining follower. Sample `neg` with one and accounts one doesn't know. Query the GNN with ones modified network to get posteriors to sampled features. Train `attacker` on the sampled dataset.

Predict whether one knows people or not.

##### Results
{get_results(data, 'surviving_edges_80p')}
<img src=\"{get_plot(data, "surviving_edges_80p")}\" alt="drawing" width="520"/>

"""

    # write Evaluation
    with open('./eval/Evaluation.md', 'w') as ev:
        ev.write(content)


if __name__ == '__main__':

    main()

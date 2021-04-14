# Basic Understanding of Graphs

## Definition
- data structures
- model set of objects and their relationships
- nodes represent objects
- edges represent relationships


- formal: `G = (V, E)`
  - G : graph
  - V : set of nodes / vertex
  - E : set of edges


## Example

- Social Network
  - Nodes: Persons
    - Label: Salary
    - Features: IP, Age, Email, Education status
  - Edges: A -> B exists, if person A befriends person B


## Vocabulary
- Degree of a node
  - amount of edges the node has
  - E.g.:
    - A - B - C
      - degree of A: 1
      - degree of B: 2
      - degree of C: 1
    - A <-> B -> C
      - degree of A: ( 1, 1 )
      - degree of B: ( 1, 1 )
      - degree of C: ( 1, 0 )

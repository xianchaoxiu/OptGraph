# Description

Minimize the global cost of a multi-commodity capacitated network design (MCND) problem by tracking discrete asset deployment. We must route multiple commodities $k$, each having an origin, a destination, and a specific commodity demand $d^k$, across a directed graph of nodes $i$ and arcs $(i,j)$. The model simultaneously optimizes continuous fractional commodity flows $x_{i,j}^k \in [0, 1]$ and integer design variables $y_{i,j}$ representing the number of operational facilities to install on each arc $(i,j)$. Total bundled capacity constraints and flow balance requirements must be maintained across all arcs and nodes.

# Model

[Objective]
Minimize the sum of multi-commodity continuous routing costs and arc-specific discrete facility installation costs:
$$ \text{Minimize } Z = \sum_{k \in K} \sum_{(i,j) \in A} d^{k} \cdot \text{Cost}_{i,j}^{k} \cdot x_{i,j}^{k} + \sum_{(i,j) \in A} \text{FixedCost}_{i,j} \cdot y_{i,j} $$

[Constraints]
1. Flow Conservation Constraints: For each commodity $k$ and node $i$, the net outbound fractional flow must match its source/sink status ($\delta_i^k$):
$$ \sum_{j: (i,j) \in A} x_{i,j}^{k} - \sum_{j: (j,i) \in A} x_{j,i}^{k} = \delta_{i}^{k} \quad \forall i \in N, \forall k \in K $$
*(Where $\delta_i^k = 1$ if node $i$ is the origin of $k$, $-1$ if it is the destination, and $0$ otherwise)*

2. Bundle Capacity Constraints: The aggregate real demand flow routed across arc $(i,j)$ cannot exceed the capacity provided by the installed facilities:
$$ \sum_{k \in K} d^{k} \cdot x_{i,j}^{k} \leq \text{Capacity}_{i,j} \cdot y_{i,j} \quad \forall (i,j) \in A $$

# Code

```python
import pulp

nodes = ["Node_0", "Node_1", "Node_2"]
commodities = ["Comm_0", "Comm_1"]
arcs = [("Node_0", "Node_1"), ("Node_1", "Node_2"), ("Node_0", "Node_2")]

# Commodity mapping structures
demand = {k: None for k in commodities}
od_pairs = {k: (None, None) for k in commodities}  # (Origin, Destination)

capacity = {arc: None for arc in arcs}
fixed_cost = {arc: None for arc in arcs}
var_cost = {(k, arc[0], arc[1]): None for k in commodities for arc in arcs}

# [INSERT_INSTANCE_DATA_HERE]

problem = pulp.LpProblem("MCND_Problem", pulp.LpMinimize)

# Decision Variables: x is continuous fractional flow, y is integer design counts
x = pulp.LpVariable.dicts("x", (commodities, arcs), lowBound=0, upBound=1, cat='Continuous')
y = pulp.LpVariable.dicts("y", arcs, lowBound=0, cat='Integer')

# Objective Function
routing_part = pulp.lpSum(demand[k] * var_cost[k, arc[0], arc[1]] * x[k][arc] for k in commodities for arc in arcs)
fixed_part = pulp.lpSum(fixed_cost[arc] * y[arc] for arc in arcs)
problem += routing_part + fixed_part

# 1. Flow Conservation Constraints
for k in commodities:
    origin, destination = od_pairs[k]
    for i in nodes:
        out_flow = pulp.lpSum(x[k][arc] for arc in arcs if arc[0] == i)
        in_flow = pulp.lpSum(x[k][arc] for arc in arcs if arc[1] == i)
        
        delta = 0
        if i == origin:
            delta = 1
        elif i == destination:
            delta = -1
            
        problem += (out_flow - in_flow == delta), f"Flow_Balance_{k}_{i}"

# 2. Bundle Capacity Constraints
for arc in arcs:
    i, j = arc
    problem += pulp.lpSum(demand[k] * x[k][arc] for k in commodities) <= capacity[arc] * y[arc], f"Capacity_{i}_{j}"

problem.solve(pulp.PULP_CBC_CMD(msg=False))

status_str = pulp.LpStatus[problem.status]
print("Solver Status:", status_str)

if status_str == "Optimal":
    obj_val = pulp.value(problem.objective)
    print("Objective Value:", obj_val)
    print("result:", obj_val)
else:
    print("result: None")
```

# Typical Error

[Symptom] Capacity bottleneck or flow leakage. If `capacity` parameters are set too small or `y` is erroneously defined as a continuous variable instead of an integer type, or if structural connectivity constraints fail to enforce strict matrix matches, the solver triggers an "Infeasible" or fractional design error, returning "result: None".
[Fix Hint] Ensure the design variable explicitly uses `cat='Integer'`. Validate that network graph cuts have sufficient maximum capacity bounds to route individual commodity demands from their origins to their destinations before executing the solver.

# Type

Network Flow

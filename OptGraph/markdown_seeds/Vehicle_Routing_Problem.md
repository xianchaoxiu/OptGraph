# Description

Minimize the total transportation distance for a fleet of capacity-constrained vehicles serving a set of customers $i$ from a central depot. Each customer has a specific inventory demand $d_i$, a required service time $s_i$, and a strict service time window $[e_i, l_i]$. The model decides binary active routing edges ($x_{i,j} \in \{0,1\}$) and tracks continuous arrival times $t_i$ to ensure vehicles satisfy collective delivery caps without breaking customer-specific timing windows or forming illegal sub-loops.

# Model

[Objective]
Minimize the global fleet transit distance across all active traversal paths:
$$ \text{Minimize } Z = \sum_{i \in \text{Nodes}} \sum_{j \in \text{Nodes}: j \neq i} \text{Distance}_{i,j} \cdot x_{i,j} $$

[Constraints]
1. Core Flow Assignment Boundaries: Every single customer node $i$ (excluding the central depot) must be entered exactly once and left exactly once by a vehicle:
$$ \sum_{j \in \text{Nodes}: j \neq i} x_{i,j} = 1 \quad \forall i \in \text{Customers} $$
$$ \sum_{j \in \text{Nodes}: j \neq i} x_{j,i} = 1 \quad \forall i \in \text{Customers} $$

2. Arrival Time Windows and Big-M Linearization: The vehicle arrival time $t_j$ at a downstream node $j$ must track previous arrival times $t_i$, transit distances, and service times $s_i$, while staying within the localized time windows $[e_j, l_j]$:
$$ t_i + s_i + \text{Distance}_{i,j} - t_j \leq M \cdot (1 - x_{i,j}) \quad \forall i, j \in \text{Nodes}: j \neq i $$
$$ e_j \leq t_j \leq l_j \quad \forall j \in \text{Nodes} $$

3. Vehicle Capacity Slacks: Accumulating demands along any active routing tour cannot exceed the total vehicle cargo hold volume ($Q$):
$$ u_i + d_j - u_j \leq Q \cdot (1 - x_{i,j}) \quad \forall i, j \in \text{Nodes}: j \neq i $$
$$ d_i \leq u_i \leq Q \quad \forall i \in \text{Customers} $$
*(Where $u_i$ tracks the cumulative vehicle load after serving customer $i$)*

# Code

```python
import pulp

# Depot is indexed as "Depot", customers are dynamically generated
customers = [f"Cust_{i}" for i in range(5)]  # Represents n_customers
nodes = ["Depot"] + customers

demand = {i: None for i in customers}
demand["Depot"] = 0

# Time windows represented as tuples: (lower_bound, upper_bound)
time_window = {i: (None, None) for nodes in nodes}
service_time = {i: None for i in customers}
service_time["Depot"] = 0

distance = {i: {j: None for j in nodes} for i in nodes}
vehicle_capacity = None  # Represents vehicle_capacity
M = None  # Large constant for linkage conditions

# [INSERT_INSTANCE_DATA_HERE]

problem = pulp.LpProblem("CVRPTW_Problem", pulp.LpMinimize)

# Decision Variables: x is binary arc traversal, t is arrival time, u is cumulative cargo load
x = pulp.LpVariable.dicts("x", (nodes, nodes), cat='Binary')
t = pulp.LpVariable.dicts("t", nodes, lowBound=0, cat='Continuous')
u = pulp.LpVariable.dicts("u", nodes, lowBound=0, cat='Continuous')

# Objective Function
problem += pulp.lpSum(distance[i][j] * x[i][j] for i in nodes for j in nodes if i != j)

# Constraints
# 1. Flow assignment loop for customers
for i in customers:
    problem += pulp.lpSum(x[i][j] for j in nodes if i != j) == 1, f"Leave_{i}"
    problem += pulp.lpSum(x[j][i] for j in nodes if i != j) == 1, f"Enter_{i}"

# 2. Time window boundaries and Big-M arrival tracking linkage
for i in nodes:
    problem += t[i] >= time_window[i][0], f"Time_Window_Lower_{i}"
    problem += t[i] <= time_window[i][1], f"Time_Window_Upper_{i}"

for i in nodes:
    for j in nodes:
        if i != j:
            # Multi-conditional sequence enforcement via M
            problem += t[i] + service_time[i] + distance[i][j] - t[j] <= M * (1 - x[i][j]), f"Time_Link_{i}_{j}"
            # Cumulative capacity load tracking via Q
            problem += u[i] + demand[j] - u[j] <= vehicle_capacity * (1 - x[i][j]), f"Capacity_Link_{i}_{j}"

# 3. Capacity cell ranges for active customers
for i in customers:
    problem += u[i] >= demand[i], f"Load_Lower_{i}"
    problem += u[i] <= vehicle_capacity, f"Load_Upper_{i}"

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

[Symptom] Time window conflict or loose M rupture. If the large constant $M$ is set too small (less than the longest horizon timescale), it fails to relax the constraints for inactive arcs ($x_{i,j}=0$), causing the solver to return a false "Infeasible" status, resulting in "result: None".
[Fix Hint] Ensure $M$ is mathematically bounded to be greater than $\max(l_i) + \max(\text{service\_time}) + \max(\text{distance})$. Check that customer demand elements never individually exceed the `vehicle_capacity` limit.

# Type

Vehicle Routing Problem

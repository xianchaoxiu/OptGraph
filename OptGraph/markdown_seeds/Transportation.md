# Description

Minimize the total cost to transport commodities from origins $i$ to destinations $j$. Each origin $i$ has a fixed supply capacity, and each destination $j$ has a fixed demand requirement. The flow $x_{i,j}$ on each route must satisfy all supply boundaries and demand targets.

# Model

[Objective]
Minimize the total transportation cost across all shipping routes:
$$ \text{Minimize } Z = \sum_{i \in \text{Origins}} \sum_{j \in \text{Destinations}} \text{Cost}_{i,j} \cdot x_{i,j} $$

[Constraints]
1. Supply Capacity Constraints: Total outbound flow from each origin $i$ must exactly equal its fixed supply capacity:
$$ \sum_{j \in \text{Destinations}} x_{i,j} = \text{Supply}_i \quad \forall i \in \text{Origins} $$

2. Demand Satisfaction Constraints: Total inbound flow to each destination $j$ must exactly satisfy its specific demand requirement:
$$ \sum_{i \in \text{Origins}} x_{i,j} = \text{Demand}_j \quad \forall j \in \text{Destinations} $$

# Code

```python
import pulp

origins = ["Origin_A", "Origin_B", "Origin_C"]
destinations = ["Dest_X", "Dest_Y", "Dest_Z"]

supply = {i: None for i in origins}
demand = {j: None for j in destinations}
cost = {i: {j: None for j in destinations} for i in origins}

# [INSERT_INSTANCE_DATA_HERE]

problem = pulp.LpProblem("Transportation_Problem", pulp.LpMinimize)

# Decision variables mapped to x[i][j] from math model
x = pulp.LpVariable.dicts("x", (origins, destinations), lowBound=0, cat='Continuous')

# Objective: Minimize total transportation cost
problem += pulp.lpSum(cost[i][j] * x[i][j] for i in origins for j in destinations)

# Constraints
for i in origins:
    problem += pulp.lpSum(x[i][j] for j in destinations) == supply[i], f"Supply_{i}"

for j in destinations:
    problem += pulp.lpSum(x[i][j] for i in origins) == demand[j], f"Demand_{j}"

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

[Symptom] Unbalanced supply and demand causes "Infeasible" status, returning "result: None".
[Fix Hint] Introduce a dummy origin $i$ or dummy destination $j$ with a zero-cost matrix to absorb the surplus before building constraints.

# Type

Transportation

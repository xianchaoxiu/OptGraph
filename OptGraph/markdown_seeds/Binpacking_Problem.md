# Description

Minimize the total number of uniform bins $b$ used to pack a set of items $i$ without violating the weight capacity of any activated bin. There are $n$ items ($i \in \{1 \dots n\}$), each characterized by an individual weight $W_i$. All bins have a rigid, identical maximum capacity cap $C$. The model utilizes binary variables to decide whether a bin is activated ($y_b \in \{0,1\}$) and whether an item is allocated to a specific bin ($x_{i,b} \in \{0,1\}$).

# Model

[Objective]
Minimize the total number of bins activated for packing:
$$ \text{Minimize } Z = \sum_{b \in \text{Bins}} y_{b} $$

[Constraints]
1. Item Assignment Constraints: Every single item $i$ must be allocated to exactly one bin:
$$ \sum_{b \in \text{Bins}} x_{i,b} = 1 \quad \forall i \in \text{Items} $$

2. Bin Capacity and Linkage Constraints: Total weight of items assigned to bin $b$ cannot exceed its capacity $C$, and items can only be placed in a bin if that bin is activated ($y_b = 1$):
$$ \sum_{i \in \text{Items}} W_{i} \cdot x_{i,b} \leq C \cdot y_{b} \quad \forall b \in \text{Bins} $$

# Code

```python
import pulp

items = [f"Item_{i}" for i in range(5)]  # Represents n_items
# In worst-case, number of bins equals number of items
bins = [f"Bin_{b}" for b in range(len(items))]

weight = {i: None for i in items}
bin_capacity = None  # Represents uniform bin_capacity

# [INSERT_INSTANCE_DATA_HERE]

problem = pulp.LpProblem("Bin_Packing", pulp.LpMinimize)

# Decision Variables
y = pulp.LpVariable.dicts("y", bins, cat='Binary')
x = pulp.LpVariable.dicts("x", (items, bins), cat='Binary')

# Objective: Minimize activated bins
problem += pulp.lpSum(y[b] for b in bins)

# Constraints
for i in items:
    problem += pulp.lpSum(x[i][b] for b in bins) == 1, f"Assign_Item_{i}"

for b in bins:
    problem += pulp.lpSum(weight[i] * x[i][b] for i in items) <= bin_capacity * y[b], f"Capacity_Link_{b}"

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

[Symptom] Unbounded sizing or weight leak. If an item's weight exceeds the maximum `bin_capacity` ($W_i > C$), or if the linkage missing variable $y_b$ is implemented loosely, the solver returns "Infeasible" or fractional placement, returning "result: None".
[Fix Hint] Validate that $\max_{i}(W_i) \leq C$ in the input pipeline before running optimization. Ensure the capacity linkage uses the tight form `weight * x <= capacity * y`.

# Type

Binpacking Problem

# Description

Maximize the total value of selecting a subset of items $i$ without exceeding the maximum knapsack capacity. There are $n$ available items ($i \in \{1 \dots n\}$). Each item $i$ has a specific value $V_i$ and weight $W_i$ randomly generated within defined ranges. The total capacity limit $C$ is dynamically scaled by a capacity ratio against the total weight of all items. The objective is to determine the binary selection ($x_i \in \{0,1\}$) for each item to maximize the global payoff.

# Model

[Objective]
Maximize the total accumulated value of the selected items:
$$ \text{Maximize } Z = \sum_{i \in \text{Items}} V_{i} \cdot x_{i} $$

[Constraints]
1. Knapsack Total Weight Limit: The sum of the weights of all selected items must not exceed the maximum scaled capacity $C$:
$$ \sum_{i \in \text{Items}} W_{i} \cdot x_{i} \leq C $$

# Code

```python
import pulp

items = [f"Item_{i}" for i in range(3)]  # Represents n_items

values = {i: None for i in items}
weights = {i: None for i in items}
capacity = None  # Represents the computed C from capacity_ratio

# [INSERT_INSTANCE_DATA_HERE]

problem = pulp.LpProblem("01_Knapsack_Problem", pulp.LpMaximize)

# Decision variables: x[i] is 1 if item i is selected, 0 otherwise
x = pulp.LpVariable.dicts("x", items, cat='Binary')

# Objective: Maximize total value
problem += pulp.lpSum(values[i] * x[i] for i in items)

# Constraint: Weight limit
problem += pulp.lpSum(weights[i] * x[i] for i in items) <= capacity, "Capacity_Constraint"

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

[Symptom] Unbounded or continuous relaxation error. If the decision variable `cat` is accidentally defined as 'Continuous' instead of 'Binary', or if capacity $C$ is improperly calculated as a negative number or `None`, the solver fractionalizes items or outputs an error, resulting in "result: None".
[Fix Hint] Ensure the selection variable is strictly bounded as a discrete type: `cat='Binary'`. Always validate that the dynamic capacity respects $C > 0$ after parsing the `capacity_ratio`.

# Type

Knapsack

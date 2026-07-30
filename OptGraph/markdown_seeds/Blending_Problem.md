# Description

Minimize the total raw material purchasing cost while formulating a chemical or metallurgical blend from a set of available alloys $i$. There are $n$ alloys ($i \in \{1 \dots n\}$), each containing a specific composition percentage $P_{i,k}$ of chemical elements $k$. The optimization determines the continuous blending input quantities $x_i$ to ensure that the final chemical composition of each element $k$ strictly matches the targeted percentage threshold requirements.

# Model

[Objective]
Minimize the aggregate procurement cost of raw input alloys:
$$ \text{Minimize } Z = \sum_{i \in \text{Alloys}} \text{Cost}_{i} \cdot x_{i} $$

[Constraints]
1. Global Volume Normalization Constraint: The sum of the blending proportions across all selected input materials must perfectly equal 1.0 (fully balanced mass representation):
$$ \sum_{i \in \text{Alloys}} x_{i} = 1.0 $$

2. Target Material Ingredient Quality Caps: For each targeted chemical element $k$, the blended mass concentration must satisfy the strict desired output percentage specification:
$$ \sum_{i \in \text{Alloys}} P_{i,k} \cdot x_{i} = \text{DesiredPercentage}_{k} \quad \forall k \in \text{Elements} $$

# Code

```python
import pulp

alloys = [f"Alloy_{i}" for i in range(5)]      # Represents n_alloys
elements = [f"Element_{k}" for k in range(3)]  # Represents n_elements

cost = {i: None for i in alloys}
# composition[i][k] represents the fractional composition P_{i,k} of element k in alloy i
composition = {i: {k: None for k in elements} for i in alloys}
desired_percentages = {k: None for k in elements}

# [INSERT_INSTANCE_DATA_HERE]

problem = pulp.LpProblem("Alloy_Blending_Problem", pulp.LpMinimize)

# Decision Variables: x[i] is the continuous blend proportion fraction of alloy i
x = pulp.LpVariable.dicts("x", alloys, lowBound=0, upBound=1, cat='Continuous')

# Objective: Minimize total blend cost
problem += pulp.lpSum(cost[i] * x[i] for i in alloys)

# Constraints
# 1. Total blend mass normalization
problem += pulp.lpSum(x[i] for i in alloys) == 1.0, "Total_Mass_Balance"

# 2. Element composition threshold targets match
for k in elements:
    problem += pulp.lpSum(composition[i][k] * x[i] for i in alloys) == desired_percentages[k], f"Element_Target_{k}"

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

[Symptom] Fractional expression rupture or compositional infeasibility. If the user frames ratios non-linearly (e.g., `x / sum(x)`), it causes compilation failures. If targeted thresholds are outside the boundary limits of available stock ingredients, the solver drops into "Infeasible" status, returning "result: None".
[Fix Hint] Always express fractional restrictions in their pre-multiplied linear form. Run a data verification pre-check: ensure that $\min_{i}(P_{i,k}) \leq \text{desired\_percentage}[k] \leq \max_{i}(P_{i,k})$ holds true for all ingredients.

# Type

Blending Problem

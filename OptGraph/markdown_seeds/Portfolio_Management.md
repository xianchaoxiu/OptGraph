# Description

Minimize the global investment risk (total portfolio variance or mean absolute deviation) by optimizing the allocation weights $w_i$ across a set of financial assets $i$. Each asset $i$ is characterized by an expected annual return $R_i$ and an individual volatility risk profile. The model must determine the continuous selection weights ($w_i \in [\text{MinWeight}, \text{MaxWeight}]$) to satisfy the strict aggregate budget constraint (total weights sum to 1.0) while ensuring that the total expected portfolio return meets or exceeds the investor's minimum required target return.

# Model

[Objective]
Minimize the total absolute risk deviation of the selected portfolio allocation:
$$ \text{Minimize } Z = \sum_{i \in \text{Assets}} \text{Risk}_{i} \cdot w_{i} $$

[Constraints]
1. Budget Allocation Constraint: The sum of the investment weights across all available financial assets must perfectly equal 1.0 (fully invested budget):
$$ \sum_{i \in \text{Assets}} w_{i} = 1.0 $$

2. Target Return Requirement: The weighted expected return of the formed portfolio must meet or exceed the investor's specified minimum threshold rate:
$$ \sum_{i \in \text{Assets}} \text{Return}_{i} \cdot w_{i} \geq \text{TargetReturn} $$

3. Individual Diversification Boundaries: Each individual asset weight must be bounded within the strict regulatory risk-diversification ranges:
$$ \text{MinWeight} \leq w_{i} \leq \text{MaxWeight} \quad \forall i \in \text{Assets} $$

# Code

```python
import pulp

# Index Sets Definition based on parameters specification
assets = [f"asset_{i}" for i in range(10)]  # Represents n_assets

# Parameter structures aligned with the markdown criteria
expected_return = {i: None for i in assets}
asset_risk = {i: None for i in assets}

target_return = None
min_weight = None
max_weight = None

# [INSERT_INSTANCE_DATA_HERE]

problem = pulp.LpProblem("Portfolio_Optimization", pulp.LpMinimize)

# Decision Variables: w[i] represents the continuous investment weight of asset i
w = pulp.LpVariable.dicts("w", assets, lowBound=0, upBound=1, cat='Continuous')

# Objective Function: Minimize the weighted portfolio risk index
problem += pulp.lpSum(asset_risk[i] * w[i] for i in assets)

# Constraints
# 1. Budget constraint (Weights must sum to 1.0)
problem += pulp.lpSum(w[i] for i in assets) == 1.0, "Budget_Constraint"

# 2. Target return constraint
problem += pulp.lpSum(expected_return[i] * w[i] for i in assets) >= target_return, "Target_Return_Constraint"

# 3. Individual weight diversification bounds loop
for i in assets:
    problem += w[i] >= min_weight, f"Min_Bound_{i}"
    problem += w[i] <= max_weight, f"Max_Bound_{i}"

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

[Symptom] Feasibility collapse due to aggressive target returns. If the investor's `target_return` is set higher than the maximum expected return among all assets within the defined diversification bounds ($\text{target\_return} > \max(R_i) \cdot \text{max\_weight}$), the solver fails instantly with an "Infeasible" status, returning "result: None".
[Fix Hint] Run a market feasibility pre-check: verify that the required target return falls within the valid mathematical bounds of the asset universe: $\text{target\_return} \leq \max_{i}(R_i)$ before calling the optimizer.

# Type

Portfolio Management

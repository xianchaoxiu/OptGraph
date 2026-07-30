# Description

Minimize the total multi-period, multi-product cost of a capacitated lot sizing problem (MCLSP) by managing inventory flows and discrete asset switches. Across a horizon of periods $t$, the factory must determine production volumes $x_{p,t}$ and held stock levels $s_{p,t}$ for various products $p$ to satisfy market demands. The optimization balances binary setup triggers $y_{p,t} \in \{0,1\}$ incurring heavy fixed setup costs against variable unit manufacturing and holding costs, subject to a collective total capacity limit scaled by a unit resource usage rate in each period.

# Model

[Objective]
Minimize the sum of fixed changeover setup costs, linear manufacturing costs, and period-end stock holding costs across all products and horizons:
$$ \text{Minimize } Z = \sum_{p \in P} \sum_{t \in T} (\text{SetupCost}_{p,t} \cdot y_{p,t} + \text{ProdCost}_{p,t} \cdot x_{p,t} + \text{HoldCost}_{p,t} \cdot s_{p,t}) $$

[Constraints]
1. Multi-Product Inventory Flow Balance: For each product $p$ and period $t$, the ending inventory stock is governed by the previous period's stock plus current production volume minus specific market demand:
$$ s_{p,t} = s_{p,t-1} + x_{p,t} - \text{Demand}_{p,t} \quad \forall p \in P, \forall t \in T $$
*(Where $s_{p,0} = 0$ represents the initial given opening warehouse stock)*

2. Aggregate Capacity and Setup Big-M Linkage: Total dynamic resource usage across all products cannot exceed the maximum available capacity in period $t$, and production $x_{p,t}$ can only be active if the line setup flag is triggered ($y_{p,t} = 1$):
$$ \sum_{p \in P} \text{ResourceUsage}_{p} \cdot x_{p,t} \leq \text{Capacity}_{t} \quad \forall t \in T $$
$$ x_{p,t} \leq \text{BigM}_{p,t} \cdot y_{p,t} \quad \forall p \in P, \forall t \in T $$
*(Where $\text{BigM}_{p,t} = \min(\text{Capacity}_{t} / \text{ResourceUsage}_{p}, \sum_{\tau=t}^{|T|} \text{Demand}_{p,\tau})$ represents the tightest logical upper bound)*

# Code

```python
import pulp

products = ["Prod_1", "Prod_2"]  # Represents n_products
periods = [f"Period_{t}" for t in range(1, 7)]  # Represents n_periods
t_list = list(periods)

setup_cost = {(p, t): None for p in products for t in periods}
prod_cost = {(p, t): None for p in products for t in periods}
holding_cost = {(p, t): None for p in products for t in periods}
demand = {(p, t): None for p in products for t in periods}

resource_usage = {p: None for p in products}
capacity = {t: None for t in periods}

# [INSERT_INSTANCE_DATA_HERE]

problem = pulp.LpProblem("Multi_Product_Capacitated_Lot_Sizing", pulp.LpMinimize)

# Decision Variables: y is binary setup switch, x is production volume, s is inventory stock
y = pulp.LpVariable.dicts("y", (products, periods), cat='Binary')
x = pulp.LpVariable.dicts("x", (products, periods), lowBound=0, cat='Continuous')
s = pulp.LpVariable.dicts("s", (products, periods), lowBound=0, cat='Continuous')

# Objective Function: Minimize total operations overhead
total_setup = pulp.lpSum(setup_cost[p, t] * y[p][t] for p in products for t in periods)
total_prod = pulp.lpSum(prod_cost[p, t] * x[p][t] for p in products for t in periods)
total_holding = pulp.lpSum(holding_cost[p, t] * s[p][t] for p in products for t in periods)
problem += total_setup + total_prod + total_holding

# Constraints
# 1. Multi-product inventory flow balance loop
for p in products:
    for idx, t in enumerate(t_list):
        prev_stock = 0 if idx == 0 else s[p][t_list[idx - 1]]
        problem += (s[p][t] == prev_stock + x[p][t] - demand[p, t]), f"Inv_Balance_{p}_{t}"

# 2. Shared resource capacity limitation per period
for t in periods:
    problem += pulp.lpSum(resource_usage[p] * x[p][t] for p in products) <= capacity[t], f"Capacity_Bound_{t}"

# 3. Big-M linkage constraint forcing y[p][t] to 1 if x[p][t] > 0
for p in products:
    for t in periods:
        # Tight logical Big-M bound to improve LP relaxation efficiency
        big_m = capacity[t] / resource_usage[p]
        problem += x[p][t] <= big_m * y[p][t], f"Setup_Link_{p}_{t}"

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

[Symptom] Loose Big-M coefficient or mathematical bottleneck. If the `big_m` parameter is hardcoded as an arbitrarily large constant instead of dynamically bounding it via `capacity[t] / resource_usage[p]`, the solver's execution time explodes or triggers numerical instability. If cumulative capacity fails to support tight forward item demands, the status results in "Infeasible", returning "result: None".
[Fix Hint] Explicitly bind the production linkage using the contextual mathematical domain bound: `x[p][t] <= (capacity[t] / resource_usage[p]) * y[p][t]`. Run a pre-check validation ensuring shared time availability meets cumulative product demand targets.

# Type

Lot Sizing

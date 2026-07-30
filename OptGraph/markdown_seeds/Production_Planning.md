# Description

Maximize the total net profit of a multi-period, multi-product factory planning framework subject to multi-machine capacity limits. Across a planning horizon of months $t$, the factory must optimize production volumes $x_{p,t}$ and held inventory stocks $s_{p,t}$ for various products $p$ to capture market sales. The optimization must balance product-specific unit profits against holding costs, while respecting dynamic sales limitations, maximum warehouse storage boundaries, and machine processing time limits across diverse machine types $m$. A final target inventory must also be satisfied at the end of the horizon.

# Model

[Objective]
Maximize the global net profit (total sales profit minus total warehouse inventory holding costs) over the entire horizon:
$$ \text{Maximize } Z = \sum_{t \in \text{Periods}} \sum_{p \in \text{Products}} (\text{Profit}_{p} \cdot \text{Sales}_{p,t} - \text{HoldingCost} \cdot s_{p,t}) $$

[Constraints]
1. Inventory Balance and Flow Conservation: For each product $p$ and period $t$, the ending stock is determined by the previous stock and current production minus actual sales:
$$ s_{p,t} = s_{p,t-1} + x_{p,t} - \text{Sales}_{p,t} \quad \forall p \in \text{Products}, \forall t \in \text{Periods} $$
*(Where $s_{p,0} = 0$ as initial stock, and $\text{Sales}_{p,t} \leq \text{SalesLimit}_{p,t}$ represents market demand caps)*

2. Machine Capacity Restrictions: Total machining hours consumed by all products cannot exceed the aggregated operational time available for each machine type $m$ in any period $t$:
$$ \sum_{p \in \text{Products}} \text{MachineTime}_{p,m} \cdot x_{p,t} \leq \text{AvailableMachines}_{m} \cdot \text{MachineHours} \quad \forall m \in \text{MachineTypes}, \forall t \in \text{Periods} $$

3. Storage Limits and Final Horizon Targets: Ending inventory must respect warehouse caps, and the final period must achieve the specific target buffer:
$$ s_{p,t} \leq \text{MaxInventory} \quad \forall p \in \text{Products}, \forall t \in \text{Periods} $$
$$ s_{p,t_{\text{last}}} \geq \text{MaxInventory} \cdot \text{TargetRatio} \quad \forall p \in \text{Products} $$

# Code

```python
import pulp

products = ["Prod_1", "Prod_2", "Prod_3"]
periods = [f"Month_{t}" for t in range(1, 7)]  # n_periods = 6
machine_types = ["grinder", "drill", "borer"]

t_list = list(periods)

# Parameter tracking dictionaries aligned with the markdown specification
profit = {p: None for p in products}
holding_cost = None
max_inventory = None
target_ratio = None
machine_hours = None

n_machines = {m: None for m in machine_types}
sales_limit = {(p, t): None for p in products for t in periods}
machine_time = {(p, m): None for p in products for m in machine_types}

# [INSERT_INSTANCE_DATA_HERE]

problem = pulp.LpProblem("Factory_Planning_Problem", pulp.LpMaximize)

# Decision Variables: x is production, s is inventory, sales represents sold quantity
x = pulp.LpVariable.dicts("x", (products, periods), lowBound=0, cat='Continuous')
s = pulp.LpVariable.dicts("s", (products, periods), lowBound=0, cat='Continuous')
sales = pulp.LpVariable.dicts("sales", (products, periods), lowBound=0, cat='Continuous')

# Objective Function: Maximize net financial payoff
total_profit = pulp.lpSum(profit[p] * sales[p][t] for p in products for t in periods)
total_holding = pulp.lpSum(holding_cost * s[p][t] for p in products for t in periods)
problem += total_profit - total_holding

# Constraints
for p in products:
    for idx, t in enumerate(t_list):
        # Market bound restriction on sales variables
        problem += sales[p][t] <= sales_limit[p, t], f"Sales_Limit_{p}_{t}"
        
        # Warehouse capacity restriction
        problem += s[p][t] <= max_inventory, f"Max_Storage_{p}_{t}"
        
        # Stock Flow Conservation Balance
        prev_stock = 0 if idx == 0 else s[p][t_list[idx - 1]]
        problem += (s[p][t] == prev_stock + x[p][t] - sales[p][t]), f"Inv_Balance_{p}_{t}"

# Final horizon target inventory constraint
for p in products:
    problem += s[p][t_list[-1]] >= max_inventory * target_ratio, f"Final_Target_{p}"

# Machine capacity bounds tracking
for m in machine_types:
    for t in periods:
        problem += pulp.lpSum(machine_time[p, m] * x[p][t] for p in products) <= n_machines[m] * machine_hours, f"Capacity_{m}_{t}"

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

[Symptom] Feasibility block due to terminal storage conflict. If the market `sales_limit` is forced too high while machine hours are tight, or if the `target_inventory_ratio` creates an impossible bottleneck in the final month $t_{\text{last}}$, the solver flags "Infeasible", returning "result: None".
[Fix Hint] Check that cumulative manufacturing capacity $\sum_{m} \text{n\_machines}[m] \cdot \text{machine\_hours}$ across the weeks can comfortably cover both cumulative demand and the final period target buffer before executing optimization.

# Type

Production Planning

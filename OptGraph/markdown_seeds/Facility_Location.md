# Description

Minimize the sum of fixed distribution center (DC) opening costs and multi-commodity variable transportation costs. We must decide which potential DCs $d$ to open ($y_d \in \{0,1\}$) and determine the shipping flows $x_{p,d,c,k}$ from production plants $p$, through opened DCs $d$, to customer zones $c$ for multiple commodities $k$. Opened DCs are subject to specific minimum and maximum throughput boundaries.

# Model

[Objective]
$$ \text{Minimize } Z = \sum_{d \in D} \text{FixedCost}_{d} \cdot y_{d} + \sum_{p \in P}\sum_{d \in D}\sum_{c \in C}\sum_{k \in K} (\text{ShipCost}_{p,d,c,k} + \text{UnitDC}\text{Cost}_{d}) \cdot x_{p,d,c,k} $$

[Constraints]
1. Supply Capacity Constraints: Total flow of commodity $k$ leaving plant $p$ cannot exceed its supply capacity:
$$ \sum_{d \in D} \sum_{c \in C} x_{p,d,c,k} \leq \text{Supply}_{p,k} \quad \forall p \in P, \forall k \in K $$

2. Demand Satisfaction Constraints: Total flow of commodity $k$ reaching customer $c$ must satisfy its exact demand:
$$ \sum_{p \in P} \sum_{d \in D} x_{p,d,c,k} = \text{Demand}_{c,k} \quad \forall c \in C, \forall k \in K $$

3. DC Maximum Throughput Linkage: Total flow through DC $d$ is bounded by its maximum capacity if opened ($y_d = 1$):
$$ \sum_{p \in P} \sum_{c \in C} \sum_{k \in K} x_{p,d,c,k} \leq \text{MaxCap}_{d} \cdot y_{d} \quad \forall d \in D $$

4. DC Minimum Throughput Linkage: Total flow through DC $d$ must meet its minimum requirement if opened ($y_d = 1$):
$$ \sum_{p \in P} \sum_{c \in C} \sum_{k \in K} x_{p,d,c,k} \geq \text{MinCap}_{d} \cdot y_{d} \quad \forall d \in D $$

# Code

```python
import pulp

commodities = ["Prod_1", "Prod_2"]
plants = ["Plant_A", "Plant_B"]
dcs = ["DC_X", "DC_Y"]
customers = ["Zone_1", "Zone_2"]

supply = {(p, k): None for p in plants for k in commodities}
demand = {(c, k): None for c in customers for k in commodities}
fixed_cost = {d: None for d in dcs}
min_cap = {d: None for d in dcs}
max_cap = {d: None for d in dcs}
unit_dc_cost = {d: None for d in dcs}
ship_cost = {(p, d, c, k): None for p in plants for d in dcs for c in customers for k in commodities}

# [INSERT_INSTANCE_DATA_HERE]

problem = pulp.LpProblem("Multi_Stage_Facility_Location", pulp.LpMinimize)

y = pulp.LpVariable.dicts("y", dcs, cat='Binary')
x = pulp.LpVariable.dicts("x", (plants, dcs, customers, commodities), lowBound=0, cat='Continuous')

fixed_part = pulp.lpSum(fixed_cost[d] * y[d] for d in dcs)
variable_part = pulp.lpSum((ship_cost[p, d, c, k] + unit_dc_cost[d]) * x[p][d][c][k] for p in plants for d in dcs for c in customers for k in commodities)
problem += fixed_part + variable_part

for p in plants:
    for k in commodities:
        problem += pulp.lpSum(x[p][d][c][k] for d in dcs for c in customers) <= supply[p, k], f"Supply_{p}_{k}"

for c in customers:
    for k in commodities:
        problem += pulp.lpSum(x[p][d][c][k] for p in plants for d in dcs) == demand[c, k], f"Demand_{c}_{k}"

for d in dcs:
    problem += pulp.lpSum(x[p][d][c][k] for p in plants for c in customers for k in commodities) <= max_cap[d] * y[d], f"Max_Cap_{d}"

for d in dcs:
    problem += pulp.lpSum(x[p][d][c][k] for p in plants for c in customers for k in commodities) >= min_cap[d] * y[d], f"Min_Cap_{d}"

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

[Symptom] Big-M relaxation or capacity contradiction. If $\text{MinCap}_d > \text{MaxCap}_d$ due to dirty data, or if the throughput linkage is improperly implemented without multiplying by $y_d$, the solver outputs "Infeasible", returning "result: None".
[Fix Hint] Ensure the throughput constraints strictly use the variable-linked form: $\text{total\_flow} \leq \text{MaxCap}_d \cdot y_d$. Verify that the input data respects $\text{MinCap}_d \leq \text{MaxCap}_d$ before optimization.

# Type

Facility Location

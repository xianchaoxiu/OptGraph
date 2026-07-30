# Description

Minimize the total distance of a traveling salesman tour that visits a set of cities $i$ exactly once and returns to the starting departure city. The problem is an Asymmetric Traveling Salesman Problem (ATSP) where the distance matrix between cities $i$ and $j$ is asymmetric ($\text{Distance}_{i,j} \neq \text{Distance}_{j,i}$). The formulation must strictly incorporate subtour elimination constraints to prevent the solver from forming isolated, disconnected routing loops.

# Model

[Objective]
Minimize the total asymmetric distance traversed along the selected tour routes:
$$ \text{Minimize } Z = \sum_{i \in \text{Cities}} \sum_{j \in \text{Cities}: j \neq i} \text{Distance}_{i,j} \cdot x_{i,j} $$

[Constraints]
1. Outbound Assignment Constraints: The salesman must depart from each city $i$ exactly once:
$$ \sum_{j \in \text{Cities}: j \neq i} x_{i,j} = 1 \quad \forall i \in \text{Cities} $$

2. Inbound Assignment Constraints: The salesman must arrive at each city $j$ exactly once:
$$ \sum_{i \in \text{Cities}: i \neq j} x_{i,j} = 1 \quad \forall j \in \text{Cities} $$

3. MTZ Subtour Elimination Constraints: Continuous auxiliary positioning variables $u_i$ are introduced to break any local cyclic sub-routes isolated from the starting depot (assuming index $0$ is the base city):
$$ u_i - u_j + |\text{Cities}| \cdot x_{i,j} \leq |\text{Cities}| - 1 \quad \forall i, j \in \text{Cities} \setminus \{0\}, i \neq j $$

# Code

```python
import pulp

cities = [f"city_{i}" for i in range(5)]  # Represents n_cities
n = len(cities)

distance = {i: {j: None for j in cities} for i in cities}

# [INSERT_INSTANCE_DATA_HERE]

problem = pulp.LpProblem("Asymmetric_TSP", pulp.LpMinimize)

# Decision Variables: x is binary routing choice, u is continuous MTZ auxiliary order tracking
x = pulp.LpVariable.dicts("x", (cities, cities), cat='Binary')
u = pulp.LpVariable.dicts("u", cities, lowBound=1, upBound=n, cat='Continuous')

# Objective Function: Minimize global tour distance
problem += pulp.lpSum(distance[i][j] * x[i][j] for i in cities for j in cities if i != j)

# Constraints
# 1. Leave each city exactly once
for i in cities:
    problem += pulp.lpSum(x[i][j] for j in cities if i != j) == 1, f"Leave_{i}"

# 2. Enter each city exactly once
for j in cities:
    problem += pulp.lpSum(x[i][j] for i in cities if i != j) == 1, f"Enter_{j}"

# 3. MTZ Subtour Elimination (Base city is fixed at index 0: cities[0])
base_city = cities[0]
for i in cities:
    for j in cities:
        if i != base_city and j != base_city and i != j:
            problem += u[i] - u[j] + n * x[i][j] <= n - 1, f"Subtour_Elim_{i}_{j}"

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

[Symptom] Subtour leak or continuous explosion. If the MTZ constraints are omitted, or if the big-M coefficient is miscalculated as $n-1$ instead of $n$ in the expression $u_i - u_j + n \cdot x_{i,j}$, the solver allows detached short loops (subtours), producing a falsely low objective value or an illegal path layout.
[Fix Hint] Ensure MTZ constraints strictly use the structure: `u[i] - u[j] + n * x[i][j] <= n - 1` for all nodes excluding the base depot city.

# Type

Traveling Salesman Problem

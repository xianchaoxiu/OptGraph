# Description

Minimize the total operational and employee preference costs when scheduling a pool of workers $i$ to various shifts $s$ at multiple restaurant locations $r$. Each employee $i$ has specific skill proficiencies $k$ and shift availability windows. The schedule must determine binary deployment choices ($x_{i,r,s,k} \in \{0,1\}$) to satisfy localized skill-specific staffing demands, utilizing continuous penalty variables to absorb any unfulfilled shift positions at a heavy penalty cost.

# Model

[Objective]
Minimize the sum of employees' shift preference costs and penalty costs incurred from unfilled positions:
$$ \text{Minimize } Z = \sum_{i \in E} \sum_{r \in R} \sum_{s \in S} \sum_{k \in K} \text{Preference}_{i,s} \cdot x_{i,r,s,k} + \sum_{r \in R} \sum_{s \in S} \sum_{k \in K} \text{PenaltyCost} \cdot z_{r,s,k} $$

[Constraints]
1. Localized Demand Satisfaction with Soft Penalties: For each restaurant $r$, shift $s$, and required skill $k$, the total number of assigned qualified employees plus the unfulfilled shortage $z_{r,s,k}$ must exactly meet the market staffing demand:
$$ \sum_{i \in E} x_{i,r,s,k} + z_{r,s,k} = \text{Demand}_{r,s,k} \quad \forall r \in R, \forall s \in S, \forall k \in K $$

2. Worker Availability and Skill Eligibility Qualifications: An employee $i$ can only be assigned to a shift $s$ at restaurant $r$ using skill $k$ if they possess that skill ($\text{HasSkill}_{i,k}=1$) and are available ($\text{Available}_{i,s}=1$):
$$ x_{i,r,s,k} \leq \text{HasSkill}_{i,k} \cdot \text{Available}_{i,s} \quad \forall i \in E, \forall r \in R, \forall s \in S, \forall k \in K $$

3. Single Shift Assignment Limitations: Each employee $i$ can be assigned to at most one shift and one restaurant location per day:
$$ \sum_{r \in R} \sum_{s \in S} \sum_{k \in K} x_{i,r,s,k} \leq 1 \quad \forall i \in E $$

# Code

```python
import pulp

# Index Sets Definition based on parameters specification
employees = [f"Emp_{i}" for i in range(12)]      # n_employees
restaurants = [f"Rest_{r}" for r in range(3)]    # n_restaurants
shifts = [f"Shift_{s}" for s in range(2)]         # n_shifts
skills = [f"Skill_{k}" for k in range(2)]         # n_skills

# Parameter structures aligned with the markdown criteria
demand = {(r, s, k): None for r in restaurants for s in shifts for k in skills}
has_skill = {(i, k): None for i in employees for k in skills}
available = {(i, s): None for i in employees for s in shifts}
preference = {(i, s): None for i in employees for s in shifts}
unfulfilled_cost = None

# [INSERT_INSTANCE_DATA_HERE]

problem = pulp.LpProblem("Restaurant_Staff_Scheduling", pulp.LpSummary if hasattr(pulp, 'LpSummary') else pulp.LpMinimize)

# Decision Variables: x is binary assignment, z is continuous soft-demand penalty slack
x = pulp.LpVariable.dicts("x", (employees, restaurants, shifts, skills), cat='Binary')
z = pulp.LpVariable.dicts("z", (restaurants, shifts, skills), lowBound=0, cat='Continuous')

# Objective Function: Minimize preference costs + unfulfilled position penalties
pref_part = pulp.lpSum(preference[i, s] * x[i][r][s][k] for i in employees for r in restaurants for s in shifts for k in skills)
penalty_part = pulp.lpSum(unfulfilled_cost * z[r][s][k] for r in restaurants for s in shifts for k in skills)
problem += pref_part + penalty_part

# Constraints
# 1. Soft demand fulfillment loop
for r in restaurants:
    for s in shifts:
        for k in skills:
            problem += pulp.lpSum(x[i][r][s][k] for i in employees) + z[r][s][k] == demand[r, s, k], f"Demand_{r}_{s}_{k}"

# 2. Maximum single shift assignment per worker per day
for i in employees:
    problem += pulp.lpSum(x[i][r][s][k] for r in restaurants for s in shifts for k in skills) <= 1, f"Single_Shift_{i}"

# 3. Qualification alignment constraints (Forcing x to 0 if worker lacks skill or availability)
for i in employees:
    for r in restaurants:
        for s in shifts:
            for k in skills:
                # Enforce dynamic domain bounds
                problem += x[i][r][s][k] <= has_skill[i, k] * available[i, s], f"Qual_{i}_{r}_{s}_{k}"

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

[Symptom] Total infeasibility block or qualification leakage. If the slack variables $z_{r,s,k}$ are omitted and the staff availability matrix is restricted too tightly, the solver outputs "Infeasible". Conversely, if the qualification linkage matrix is implemented incorrectly, unqualified or unavailable staff are assigned, causing illegal schedule layouts.
[Fix Hint] Ensure the soft penalty slacks $z_{r,s,k}$ are always active in the demand equations. Check that the bounds constraint explicitly references the interaction terms: `x[i][r][s][k] <= has_skill[i, k] * available[i, s]`.

# Type

Scheduling

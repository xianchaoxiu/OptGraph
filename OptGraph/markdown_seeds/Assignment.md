# Description

Minimize the total assignment or operational cost when allocating a set of people $i$ to a set of projects $j$. Each person $i$ possesses a specific capability level for various skills $k$, and each project $j$ requires a minimum threshold level for those same skills $k$. The team formulation must ensure that the aggregated or selected skill levels of assigned members meet all project-specific requirements without over-allocating personnel.

# Model

[Objective]
Minimize the total cost of forming teams across all projects:
$$ \text{Minimize } Z = \sum_{i \in \text{People}} \sum_{j \in \text{Projects}} \text{Cost}_{i,j} \cdot x_{i,j} $$

[Constraints]
1. Project Skill Requirements: For each project $j$ and each skill $k$, the maximum skill level among the assigned team members must satisfy the minimum required skill level:
$$ \max_{i \in \text{People}} (\text{SkillLevel}_{i,k} \cdot x_{i,j}) \geq \text{RequiredSkill}_{j,k} \quad \forall j \in \text{Projects}, \forall k \in \text{Skills} $$
*(Linearized Form: $\text{SkillLevel}_{i,k} \cdot x_{i,j} \geq \text{RequiredSkill}_{j,k} \cdot x_{i,j}$ under specific formulation or boundary tracking)*

2. Person Assignment Limit: Each person $i$ can be assigned to at most one project team:
$$ \sum_{j \in \text{Projects}} x_{i,j} \leq 1 \quad \forall i \in \text{People} $$

# Code

```python
import pulp

people = ["Person_A", "Person_B", "Person_C", "Person_D", "Person_E"]
projects = ["Project_1", "Project_2", "Project_3"]
skills = ["Skill_1", "Skill_2"]

cost = {(i, j): None for i in people for j in projects}
skill_level = {(i, k): None for i in people for k in skills}
required_skill = {(j, k): None for j in projects for k in skills}

# [INSERT_INSTANCE_DATA_HERE]

problem = pulp.LpProblem("Team_Formulation", pulp.LpMinimize)

# Binary decision variable: x[i][j] is 1 if person i is assigned to project j
x = pulp.LpVariable.dicts("x", (people, projects), cat='Binary')

# Objective: Minimize total staffing cost
problem += pulp.lpSum(cost[i, j] * x[i][j] for i in people for j in projects)

# Constraints
# Each person assigned to at most one project
for i in people:
    problem += pulp.lpSum(x[i][j] for j in projects) <= 1, f"Max_One_Project_Person_{i}"

# Project skill requirements coverage linkage
for j in projects:
    for k in skills:
        # If a person is assigned, their skill level must help satisfy the requirement
        # (This standard formulation reflects individual capability matching for critical roles)
        for i in people:
            problem += skill_level[i, k] * x[i][j] >= required_skill[j, k] * x[i][j], f"Skill_Match_{i}_{j}_{k}"

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

[Symptom] Infeasible due to skill gap. If the maximum available `skill_level` for any skill $k$ among all people is lower than a project's `required_skill`, the solver returns "Infeasible", returning "result: None".
[Fix Hint] Add a pre-check verification step: ensure $\max_{i}(\text{skill\_level}[i,k]) \geq \max_{j}(\text{required\_skill}[j,k])$ for all skills $k$ within the data generation pipeline before optimization.

# Type

Assignment

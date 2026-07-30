# Description

Minimize the total makespan ($C_{\text{max}}$) required to complete a set of distinct jobs $j$ across a fleet of available machines $m$. Each job consists of an ordered sequence of operations, where each operation must be processed on a designated machine for a specific processing duration. The model determines continuous start times $t_{j,m}$ for each operation, subject to non-overlapping sequence restrictions (precedence constraints) on the same machine and linear processing steps within the same job.

# Model

[Objective]
Minimize the total makespan (the completion time of the last operation across all jobs):
$$ \text{Minimize } Z = C_{\text{max}} $$

[Constraints]
1. Makespan Bounding Constraints: The global makespan $C_{\text{max}}$ must be greater than or equal to the completion time of the final operation for every job $j$ on its respective machine $m$:
$$ C_{\text{max}} \geq t_{j,m} + \text{ProcTime}_{j,m} \quad \forall j \in \text{Jobs}, \forall m \in \text{Machines} $$

2. Job Precedence Sequences: Within the same job $j$, an operation on machine $m_2$ can only start after the preceding operation on machine $m_1$ is fully completed:
$$ t_{j,m_2} \geq t_{j,m_1} + \text{ProcTime}_{j,m_1} $$

3. Machine Disjunctive Non-Overlapping: On any shared machine $m$, two different jobs $j_1$ and $j_2$ cannot be processed simultaneously. Big-M binary disjunctive routing variables $y_{j_1,j_2,m}$ are applied to enforce the queue sequence:
$$ t_{j_1,m} \geq t_{j_2,m} + \text{ProcTime}_{j_2,m} - M \cdot y_{j_1,j_2,m} $$
$$ t_{j_2,m} \geq t_{j_1,m} + \text{ProcTime}_{j_1,m} - M \cdot (1 - y_{j_1,j_2,m}) $$

# Code

```python
import pulp

jobs = [f"Job_{j}" for j in range(3)]        # Represents num_jobs
machines = [f"Mach_{m}" for m in range(3)]  # Represents num_machines

processing_time = {(j, m): None for j in jobs for m in machines}
# job_sequence[j] tracks the ordered list of machines a job must visit sequentially
job_sequence = {j: [] for j in jobs}
M = 999999  # Large horizon bound to relax disjunctive conflicts

# [INSERT_INSTANCE_DATA_HERE]

problem = pulp.LpProblem("JobShop_Scheduling", pulp.LpMinimize)

# Decision Variables: t tracks start times, y resolves machine sequence ordering queues
t = pulp.LpVariable.dicts("t", (jobs, machines), lowBound=0, cat='Continuous')
y = pulp.LpVariable.dicts("y", (jobs, jobs, machines), cat='Binary')
c_max = pulp.LpVariable("c_max", lowBound=0, cat='Continuous')

# Objective: Minimize global makespan
problem += c_max

# Constraints
for j in jobs:
    seq = job_sequence[j]
    # 1. Intra-job sequence precedence tracking
    for i in range(len(seq) - 1):
        m1, m2 = seq[i], seq[i+1]
        problem += t[j][m2] >= t[j][m1] + processing_time[j, m1], f"Precedence_{j}_{m1}_{m2}"
    
    # 2. Bound makespan by the final operation's completion time
    m_last = seq[-1]
    problem += c_max >= t[j][m_last] + processing_time[j, m_last], f"Makespan_Bound_{j}"

# 3. Machine capacity disjunctive collision avoidance
for m in machines:
    for idx1, j1 in enumerate(jobs):
        for idx2, j2 in enumerate(jobs):
            if idx1 < idx2:
                problem += t[j1][m] >= t[j2][m] + processing_time[j2, m] - M * y[j1][j2][m], f"Disjoint_1_{j1}_{j2}_{m}"
                problem += t[j2][m] >= t[j1][m] + processing_time[j1, m] - M * (1 - y[j1][j2][m]), f"Disjoint_2_{j1}_{j2}_{m}"

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

[Symptom] Time sequence deadlock or loose horizon rupture. If the large scheduling scale variable $M$ is set too small, it creates clipping errors on inactive pathways. If circular precedence links are mistakenly injected, the solver outputs "Infeasible", returning "result: None".
[Fix Hint] Ensure the scalar $M$ is dynamically sized to be greater than the cumulative sum of all operation durations combined. Check that intra-job sequence steps are linear and contain no cyclic loops.

# Type

JobShop Scheduling

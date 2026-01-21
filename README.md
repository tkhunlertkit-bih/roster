# Mathematical model (high level)

Indices:

- $n \in N$: nurses
- $d \in D$: days in the month
- $s \in S$: shift types (from `config.SHIFT_DEFS`)
- $b \in B$: time blocks (B1=07–15, B2=15–19, B3=19–23, B4=23–07)
- $k \in K$: skills (RN, PN)

Decision variables:

- $x_{n,d,s} \in \{0,1\}$: 1 if nurse $n$ works shift $s$ on day $d$, else 0.

## Constraints

1. **One shift per nurse per day**

```math
\sum_{s \in S} x_{n,d,s} = 1 \quad \forall n,d
```

2. **Block–skill coverage**

Let $C_{s,b} \in \{0,1\}$ from `config.SHIFT_BLOCKS`, and $\text{skill}(n)$ from `nurses.csv`.  
Let $\text{Req}_{d,b,k}$ come from `coverage_blocks.csv`.

```math
\sum_{\substack{n \in N \\ \text{skill}(n)=k}} \ \sum_{s \in S} C_{s,b}\, x_{n,d,s}
\ \ge\ \text{Req}_{d,b,k}
\quad \forall d,b,k
```

3. **Monthly hours**

With hours $H_s$ from `config.SHIFT_DEFS` and per‑nurse bounds $[L_n, U_n]$:

```math
L_n \le \sum_{d \in D}\sum_{s \in S} H_s\, x_{n,d,s} \le U_n
\quad \forall n
```

4. **Forbidden transitions**

From `config.FORBIDDEN_NEXT` (set of pairs $(s,s')$):

```math
x_{n,d,s} + x_{n,d+1,s'} \le 1
\quad \forall n,d,\ (s,s') \in \text{ForbiddenNext}
```

5. **Days-off rules**

Let $S_{\text{off}}$ from `config.OFF_SHIFTS`.

- Max 3 days off in any 4‑day window:

```math
\sum_{k=0}^{3}\ \sum_{s \in S_{\text{off}}} x_{n,d+k,s} \le 3
\quad \forall n,\ d \le |D|-3
```

- After 4 consecutive workdays, next day off (implemented via indicator variables in CP‑SAT).

6. **Leave limits**

- BD:

```math
\sum_{d \in D} x_{n,d,\text{BD}} \le 1
```

- V only when requested is enforced via fixing $x_{n,d,V}=0$ if the preference is not V.

- PUB equalization: all nurses have same $\sum_{d \in D} x_{n,d,\text{PUB}}$.

## Objective (configurable weights)

Minimize:

```math
\text{Obj}
= w_{\text{pref}} \sum_n \text{Penalty}_n
+ w_{\text{fair}} \sum_n \left| \text{Penalty}_n - \bar{P} \right|
```

where:

- $\text{Penalty}_n$ = number of days where assigned shift ≠ preferred shift.
- $\bar{P}$ is the average penalty.
- $w_{\text{pref}}, w_{\text{fair}}$ are from `config.PREFERENCE_WEIGHT`, `config.FAIRNESS_WEIGHT`.

## Making constraints configurable per department

All department-specific logic is in `config.py`:

- **Shift definitions**: `SHIFT_DEFS` (name, hours, blocks).
- **Off shifts**: `OFF_SHIFTS`.
- **Forbidden transitions**: `FORBIDDEN_NEXT`.
- **Rest/day-off rules**: `MAX_OFF_IN_4_DAYS`, `MAX_CONSEC_WORK_DAYS`.
- **Leave rules**: `MAX_BD_PER_MONTH`, `EQUALIZE_PUB_COUNTS`.
- **Objective weights**: `PREFERENCE_WEIGHT`, `FAIRNESS_WEIGHT`.

To support another department:

1. Create `config_xxx.py` with its own shift patterns and rules.
2. Change imports in `solver.py` (or add an argument to pass a config module).
3. Keep `beds_lookup_blocks.csv` consistent with its block definitions and skills.

This keeps the **solver code generic** while constraints are driven by easy-to-edit configuration and CSV inputs.

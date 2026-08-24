# CAR: Cellular Automata Rollout and Reasoning

CAR is a benchmark for learning 2D cellular-automaton rules from demonstration
pairs and predicting their rollouts for different initial conditions and
horizons. Each task applies one of 1,500 rules to a 16 × 16 toroidal grid for
1, 2, 4, or 8 steps. The rules are split evenly across the Life-like,
Generations, and Larger-than-Life families.

## Data

There are two dataset variants:

- **Single horizon:** one horizon (T) is selected for each rule, giving 1,500
  rule-horizon combinations. The selected values of T are exactly balanced,
  with 375 rules at each horizon.
- **All horizons:** every rule is paired with all four horizons
  (T = 1, 2, 4, 8), giving 6,000 rule-horizon combinations.

Each combination has 500 training examples and 4 held-out evaluation
examples. The single-horizon rows are subsets of the corresponding
all-horizons files.

Download the four Parquet files into `data/augmented/v2/`:

```bash
hf download water-vapor-vx/charm-arc-agi --repo-type dataset --local-dir . \
  --include "data/augmented/v2/car_*.parquet"
```

| File | Combinations | Examples per combination | Purpose |
|---|---:|---:|---|
| `car_single_horizon_train.parquet` | 1,500 | 500 train | One selected horizon per rule. |
| `car_single_horizon_eval.parquet` | 1,500 | 4 eval | The selected combinations seen during training. |
| `car_all_horizons_train.parquet` | 6,000 | 500 train | All four horizons for every rule. |
| `car_all_horizons_eval.parquet` | 6,000 | 4 eval | All combinations; relative to single-horizon training, 1,500 are seen and 4,500 are unseen. |

The files can also be reproduced directly in CHARM's augmented-data v2
Parquet format:

```bash
WORKERS=8 scripts/car/build_dataset.sh
```

## Task memories

The six experimental arms use the following task-memory representations:

| Arm | Task memory |
|---|---|
| `full_table` | An independent 256-dimensional row for each rule-horizon-D4 combination. |
| `lowrank_table` | A rank-16 version of the same table. |
| `two_factor_composition` | Composes a joint `(rule, horizon)` embedding with a D4 embedding. |
| `two_factor_cose` | Two-factor composition plus a rank-16 combination-specific residual. |
| `three_factor_composition` | Separately composes rule, horizon, and D4 embeddings. |
| `three_factor_cose` | Three-factor composition plus a rank-16 combination-specific residual. |

## Experiments

| Dataset | Training | Evaluation | Arms | Steps |
|---|---|---|---|---:|
| All horizons | 6,000 combinations | The same 6,000 combinations | All six | 292,000 |
| Single horizon | 1,500 combinations | All 6,000 combinations, with separate results for seen and unseen combinations | `full_table`, `lowrank_table`, and both three-factor arms | 500,000 |
| Single horizon | The same 1,500 combinations | The 1,500 seen combinations | Both two-factor arms | 500,000 |

The two-factor memories use a joint `(rule, horizon)` identity, so they cannot
construct an embedding for an unseen combination. Their single-horizon runs
therefore use the seen-only evaluation file.

## Launchers

Example launch commands:

```bash
scripts/car/single_horizon/three_factor_cose.sh
scripts/car/single_horizon/two_factor_cose.sh
scripts/car/all_horizons/two_factor_cose.sh
SEED=1 scripts/car/all_horizons/full_table.sh
```

The shared runner can launch any configuration directly:

```bash
SEED=0 scripts/car/run.sh DATASET ARM [trainer args...]
```

Use `single_horizon` or `all_horizons` for `DATASET`, and one of the six names
in the task-memory table for `ARM`.

# AzarRunFlow

Lightweight experiment tracking and reproducibility for research code.

```text
Research project
    ↓
Hydra / optional hydra-zen
    ↓
AzarRunFlow
    ↓
MLflow
```

## Key commands

Build:

```bash
make build
```

Run tests:

```bash
make test
```

Run the smoke test:

```bash
make smoke
```

Open a Docker shell:

```bash
make shell
```

Start MLflow:

```bash
make mlflow
```

Then open:

http://localhost:5001

Clean local Docker/MLflow state:

```bash
make clean
```

> `make clean` deletes local MLflow runs and artifacts.

## Experiment workflow

Experiment YAMLs live in the research project:

```text
eigenfun/
├── experiments/
│   └── run.py
└── configs/
    ├── config.yaml
    ├── dataset/
    ├── solver/
    └── loss/
```

Run the default Hydra experiment:

```bash
python experiments/run.py
```

Override values:

```bash
python experiments/run.py \
    dataset=fastmri \
    loss.alpha=0.3 \
    seed=7
```

Run a sweep:

```bash
python experiments/run.py -m \
    loss.alpha=0.1,0.2,0.3 \
    seed=1,2,3
```

## What gets logged

AzarRunFlow records:

- resolved configuration
- Git commit, branch, and dirty state
- Git diff when dirty
- package/environment snapshot
- seed
- runtime
- metrics
- figures, arrays, checkpoints, and other artifacts
- failure tracebacks

For important runs, preferably commit first:

```bash
git status
git add .
git commit -m "Prepare experiment"
```

Inspect runs in MLflow:

http://localhost:5001

The main reproducibility artifact is:

```text
config/resolved.yaml
```
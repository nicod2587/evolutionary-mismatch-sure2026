# Modelling Evolutionary Mismatch: Information Foraging and Cost-Free Novelty

Nicholas Dunajski, James Marshall — Sheffield Undergraduate Research Experience (SURE), University of Sheffield

## Overview

This project builds an information-foraging model to test whether populations
that evolve under an ancestral environment — where information is rare,
costly to obtain, but of high fitness value — display suboptimal behaviour
when confronted with a modern analogue, where information is freely
available, cheap to acquire, but rarely fitness-enhancing.

Agents are simple neural architectures inspired by the primate brain,
evolved with a genetic algorithm against a Marginal Value Theorem (MVT)
patch-foraging world. The model is used to explore why organisms persist in
foraging for low-value information, and whether this behaviour is better
explained as an evolutionary mismatch (ancestral circuits meeting a
manipulated modern environment) rather than a failure of willpower — and
whether it can account for the exploitation of supernormal stimuli by modern
information sources.

See [abstract.txt](abstract.txt) for the full research abstract.

## Repository layout

- `scripts/` — core model package
  - `agent.py` — the chemotaxis-based engagement model (neural ODE + movement)
  - `natural_world.py` — the persistent-patch MVT foraging environment
  - `genetic_algorithm.py` — genome, fitness function, and the GA itself
  - `stimulus.py` / `stimulus_engine.py` / `stimulus_ui.py` — stimulus source
    maths and interactive/visualization helpers
  - `model_dynamics.py` — default neural/stimulus parameters
  - `constants.py` — shared numeric/plotting constants
  - `analysis.py` / `animation.py` — post-run plotting and animation
  - `run_champion_sweep.py` — standalone hero-figure plot from a saved
    `champion.pkl`
- `hpc_run.py` — production GA run entry point (used on the Stanage HPC
  cluster via `run_ga.sh`)
- `run_ga.sh` — SLURM batch script for `hpc_run.py`
- `ga_results/` — evolved champion genomes, sweep data, and figures
- `Evolutionary_Mistmatch.ipynb` / `session2_validation.ipynb` — exploratory
  and validation notebooks

## Running

Locally (single process):

```bash
python -m scripts.genetic_algorithm --pop_size 50 --n_gens 30
```

On the HPC cluster:

```bash
sbatch run_ga.sh
```

## Acknowledgements

This work was supported by the Sheffield Undergraduate Research Experience
(SURE) programme at the University of Sheffield. We acknowledge the High
Performance Computer (Stanage) for computational resources.

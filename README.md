# Modelling Evolutionary Mismatch: Information Foraging and Cost-Free Novelty

Nicholas Dunajski, James Marshall — Sheffield Undergraduate Research Experience
(SURE), University of Sheffield. Compute on the Stanage HPC.

## Overview

This project builds an information-foraging model to test whether populations
that evolve under an **ancestral** environment — where information is rare,
costly to obtain, but of high fitness value — behave suboptimally when moved,
unchanged, into a **modern** analogue where information is abundant, cheap, and
rarely fitness-enhancing.

A forager is a minimal **two-state neuron** inspired by the primate foraging
circuit, evolved with a genetic algorithm against a Marginal Value Theorem (MVT)
patch-foraging world. The model asks why organisms persist in foraging for
low-value information, and whether compulsive engagement (e.g. compulsive
internet browsing) is better read as an **evolutionary mismatch** — ancestral
circuits meeting a manipulated modern environment that exploits **supernormal
stimuli** — than as a failure of willpower.

See [abstract.txt](abstract.txt) for the research abstract and
[WRITEUP_traits_and_results.md](WRITEUP_traits_and_results.md) for the current,
canonical results and their interpretation.

## The model in one screen

A single forager is a 2-state neuron `[V, β_PFC]` moving through a field of
information patches:

```
dV/dt      = (1/τ_v)  · [ −(V − V_rest) + I − β_eff ]          membrane potential
dβ_PFC/dt  = (1/τ_pfc)· [ prox·(R̄ − I) − (1−prox)·β_eff ]     opportunity-cost signal
```

- `V` — a leaky integrate-and-fire membrane (Hayden et al. 2011). Each threshold
  crossing is a **spike** = one act of sampling; a spike harvests real value.
- `β_PFC` — the value-of-leaving / opportunity-cost signal (Kolling et al. 2012).
  On a patch it accumulates the shortfall `(R̄ − I)`; when it wins it suppresses
  `V` and the agent leaves — the MVT (Charnov 1976) as a neural signal.
- perceived **scent** `I` blends honest value with a novelty cue scaled by
  curiosity `κ`. A single lever, `mismatch_index ∈ [0,1]`, slides novelty from
  *tracking real value* (ancestral) to a *constant regardless of value* (modern)
  — the supernormal-stimulus mechanism.

**Three heritable traits** (the genome), with bounds:

| trait | symbol | bounds | role | grounding |
|---|---|---|---|---|
| richness prior | `R_bar` | (0.25, 25) | MVT give-up threshold — the mismatch lever | Charnov 1976; Kolling 2012 |
| curiosity gain | `kappa` | (0.1, 15) | novelty amplification (the exploited drive) | Wang & Hayden 2019 |
| patience | `tau_pfc` | (20, 500) | timescale of the leave signal | Hayden 2011 |

**Fitness** (both terms from McNamara & Dall 2010 / Charnov 1976):

```
F = w_yield · info_yield  −  w_opp · engaged_fraction        weights (1.0, 0.3)
```

`info_yield` is real harvested value; `engaged_fraction` is the MVT opportunity
cost of time locked onto a patch. Fitness never reads `mismatch_index`, so any
ancestral→modern change in fitness is an emergent consequence of behaviour.

## Results (current)

Population evolved under the ancestral world (`mismatch_index = 0.10`,
`obs_noise = 0.1`), 50 generations, pop 120. Champion:

```
R̄ = 22.56,  κ = 6.41,  τ_pfc = 256    (fitness 0.756)
```

- **Selection acts on R̄ and κ** (interior optima; population spread collapses
  κ −81%, R̄ −46%). An independent, seed-free brute-force fitness **cube**
  (`genome_landscape.py`) places its optimum where the GA lands — the champion
  sits on the ridge, so the GA reaches the optimum rather than finding its own
  seed.
- **τ_pfc is a flat direction** (spread ~0%; champion τ disagrees with the cube
  peak). It sets only the *speed* of the leave decision, not its equilibrium — a
  sloppy parameter, reported as a null / negative control, not dropped.
- **The mismatch (hero result):** the fixed champion swept ancestral→modern has
  its information yield collapse (0.82 → 0.07), wasted engagement climb
  (0.07 → 0.93), and fitness go negative (+0.71 → −0.23), with the genome never
  changing.

Figures live in `ga_results/`
([mismatch_transition.png](ga_results/mismatch_transition.png),
[evolution_results.png](ga_results/evolution_results.png),
[A_marginalize_max_tau.png](ga_results/A_marginalize_max_tau.png),
[fitness_cube.png](ga_results/fitness_cube.png)).

## Repository layout

- `scripts/` — core model package
  - `agent.py` — the 2-state neuron ODE + chemotaxis movement
  - `natural_world.py` — the persistent-patch MVT foraging world + scent
  - `genetic_algorithm.py` — genome, fitness, the GA, and the mismatch sweep/audit
  - `model_dynamics.py` / `constants.py` — default parameters and shared constants
  - `stimulus*.py` — stimulus maths and interactive/visualization helpers
  - `genome_landscape.py` — brute-force fitness cube + 3D trajectory
  - `plot_*.py`, `analysis.py`, `animation.py` — figure generation
  - `run_champion_sweep.py` — hero figure from a saved `champion.pkl`
- `hpc_run.py` — production GA run entry point (used on Stanage via `run_ga.sh`)
- `hpc_sweep.py` / `interrupt.py` — rebuild figures from checkpoints (no re-evolution)
- `run_all.sh` and `run_*.sh` — SLURM job scripts (pipeline, cube, sweep, ladder)
- `ga_results/` — evolved champion, sweep data, figures
- `Evolutionary_Mismatch_Presentation.ipynb` — the walkthrough notebook
  (regenerate with `python build_notebook.py`)
- `docs/` — references and design specs

## Reproducing

Whole pipeline on the HPC (submit from the Stanage login node, then log off):

```bash
export TRAIN_MISMATCH=0.10; export OBS_NOISE=0.1
bash run_all.sh          # preflight gate → GA → rebuild+sweep → fitness cube
```

A quick local GA run:

```bash
python -m scripts.genetic_algorithm --pop_size 50 --n_gens 30 --n_jobs 4
```

Regenerate figures from existing results (no evolution):

```bash
python -m scripts.plot_marginal_slices                       # 2D R̄×κ landscapes
python -m scripts.genome_landscape --load ga_results/fitness_cube.npz \
       --run_dir ga_results --out_dir ga_results             # replot the cube
```

## Status

- **Done:** the mismatch effect (hero figure), which traits are under selection
  (R̄, κ interior optima; τ a reported null), and the champion validated against
  an independent fitness cube.
- **Designed, not yet run:** the mismatch-cost control (`run_ladder.sh`) —
  evolving locally-adapted champions at a ladder of mismatch levels to show the
  modern shortfall is *miscalibration*, not just a harder world. This is the
  main next experiment (see [WRITEUP_traits_and_results.md](WRITEUP_traits_and_results.md)).

## Acknowledgements

This work was supported by the Sheffield Undergraduate Research Experience (SURE)
programme at the University of Sheffield. We acknowledge the High Performance
Computer (Stanage) for computational resources.

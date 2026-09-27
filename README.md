# Modelling Evolutionary Mismatch: Information Foraging and Information-Free Novelty

Nicholas Dunajski · supervised by James A. R. Marshall (University of Sheffield) · SURE 2026

Why is a social media feed so hard to put down? A forager's patch-leaving rule (the
marginal value theorem) assumes novelty signals value. This project maps how well every
forager strategy performs as the world slides from **ancestral** (novelty tracks value) to
**modern** (novelty carries no information), and shows that mismatch creates a ceiling on
curiosity that falls as the world modernises.

The final output is the poster: [docs/poster/Nico_Poster/main.tex](docs/poster/Nico_Poster/main.tex)
(PDF: [docs/poster/poster_draft_v4.pdf](docs/poster/poster_draft_v4.pdf)),
with the abstract in [docs/poster/abstract.txt](docs/poster/abstract.txt).

## The model

One two-state neuron per forager: membrane `V` and opportunity-cost signal `β`.

```
dV/dt = (scent − (V − V_rest) − β) / τ_V
dβ/dt = (prox·(R̄ − scent) − (1 − prox)·β) / τ_β
scent   = value + κ · novelty
novelty = (1 − m)·(value × freshness) + m        m = mismatch (0 ancestral, 1 modern)
```

Two heritable traits: `R̄` (estimated average value of the environment) and `κ` (how much
novelty adds to perceived value). `τ_β` is fixed at 256. Fitness = value harvested per unit
time − 0.3 × time spent engaged on patches.

## Code

`scripts/`
- `agent.py`, `natural_world.py`, `model_dynamics.py`, `constants.py`, `stimulus.py` — the model
- `fitness.py` — the fitness objective scored at each landscape cell
- `compute_ladder.py` — computes the R̄ × κ fitness landscape at each mismatch level
- `render_poster_hero.py` — landscape filmstrip (poster: `colour_plots.png`)
- `render_fig2.py` — mismatch-load figure (poster: `mismatch load.png`)
- `render_agent_single_patch.py` — one agent on one patch (poster: `plots.png`)

`results/`
- `ladder/ladder_stack.npz` — the computed landscapes (11 mismatch levels, 60 × 60 grid)
- `fig2_data.npz` — cached data for the mismatch-load figure
- `poster_hero.png`, `poster_fig2_draft.png`, `poster_agent_single_patch.png`

## Reproduce

```bash
sbatch run_landscape.sh      # ~35 h on 32 cores -> results/ladder/ladder_stack.npz

python -m scripts.render_poster_hero --stack results/ladder/ladder_stack.npz \
       --out results/poster_hero.png --rungs 0.00 0.10 0.25 0.50 1.00
python -m scripts.render_fig2                 # add --recompute to re-simulate
python -m scripts.render_agent_single_patch
```

Requires `numpy`, `scipy`, `matplotlib`. References: [docs/references.md](docs/references.md).

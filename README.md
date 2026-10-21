# Information-Free Novelty

Nicholas Dunajski, supervised by James A. R. Marshall, University of Sheffield (SURE 2026).

A forager that evolved where novelty meant value, dropped into a world where novelty is
free. The forager is a single spiking neuron with two heritable traits: `R̄`, how rich it
expects the world to be, and `κ`, how much novelty pulls it towards a patch. The mismatch
`m` goes from 0 (novelty tracks value) to 1 (novelty is free) and changes nothing else
about the world.

```
dV/dt = (I − (V − V_rest) − β) / τ_V
dβ/dt = (prox · (R̄ − I) − (1 − prox) · β) / τ_β

scent   = value + 0.5 · κ · novelty
novelty = (1 − m) · (value × freshness) + m
```

## Results

![Fitness landscapes](results/landscapes.png)

As `m` rises, high `κ` goes from one of the best strategies to one of the worst. The best
`κ` in the biological box (`R̄ ≤ 25`, `κ ≤ 15`) falls from 14 to about 0.6.

![Mismatch load](results/mismatch_load.png)

A genome tuned to the ancestral world scores 0.56 in its own world and about 0.02 in the
modern one, while the best genome in the box still gets about 0.5.

## Running it

```bash
pip install -r requirements.txt
python tests/test_pure_mismatch.py

python -m scripts.render_landscapes        # figures from the stored landscape
python -m scripts.render_mismatch_load
python -m scripts.render_forager_video     # --mismatch_index 1.0 for the modern world
```

The landscape itself (`results/ladder_res40/ladder_stack.npz`) took a SLURM job on
Sheffield's Stanage cluster: `sbatch --time=30:00:00 --export=ALL,RES=40 run_landscape.sh`.
The simulation is chaotic, so re-running cells on another machine gives values within
about 0.04 of the stored ones.

## Poster

The printed poster ([docs/poster](docs/poster)) used an earlier version of the model, where
moving to the modern world also changed patch number, value, scent and travel speed. Its
outputs are in `results/poster_v1/`. This version changes only the novelty term.

References are in [docs/references.md](docs/references.md). MIT licence.

# Information-Free Novelty

**Modelling evolutionary mismatch: information foraging and information-free novelty**

Nicholas Dunajski, supervised by James A. R. Marshall (University of Sheffield). SURE 2026.

Why is a social media feed so hard to put down? Foraging animals follow a simple rule,
the marginal value theorem: stay on a patch while it pays better than the environment's
average, then leave. For most of our history, something new usually meant something
worth having, so a pull toward novelty was a sensible shortcut. A feed serves up
endless novelty that is worth very little.

This project models a forager as a single neuron and maps how well every forager
strategy does as its world slides from **ancestral** (novelty tracks value) to
**modern** (novelty carries no information). In the ancestral world, loving novelty
costs nothing. As the world modernises, a ceiling on novelty seeking appears and keeps
falling, and a forager tuned to the ancestral world goes from doing well to doing badly.

The poster is in [docs/poster/Nico_Poster/main.tex](docs/poster/Nico_Poster/main.tex)
and the abstract in [docs/poster/abstract.txt](docs/poster/abstract.txt).

![Fitness landscape at five mismatch levels](results/poster_hero.png)

## The model

Each forager is one neuron with two variables: a membrane potential `V` and an
opportunity-cost signal `β`.

```
dV/dt = (I − (V − V_rest) − β) / τ_V
dβ/dt = (prox · (R̄ − I) − (1 − prox) · β) / τ_β
```

`I` is the scent the forager perceives, summed over nearby patches. Each patch
contributes its scent weighted by distance:

```
scent   = value + 0.5 · κ · novelty
novelty = (1 − m) · (value × freshness) + m
```

- `value` is the share of the patch's real value still left, from 1 down to 0.
- `freshness` falls from 1 to 0 as the patch gets sampled.
- `m` is the mismatch: 0 for the ancestral world, 1 for the modern one.
- `prox` is how close the forager is to a patch, from 1 on top of it to 0 in the open.

When `V` crosses threshold the neuron spikes, and each spike takes a bite of real value
from the patch. While the forager sits on a patch, `β` grows by how far the patch falls
short of what the forager expects from the environment (`R̄ − I`). Once `β` catches up
with the scent, the neuron goes quiet and the forager wanders off to look for another
patch. At `m = 0` an emptied patch stops smelling novel, so the forager leaves on time.
At `m = 1` it still smells novel, and only `β` can pull the forager away.

Each forager has two heritable traits:

- `R̄`, how rich it expects the environment to be
- `κ`, how much novelty adds to a patch's pull

`τ_β` is fixed at 256 (about a quarter of a second, in the range measured for primate
prefrontal cortex). Fitness is the real value harvested per unit time, minus 0.3 times
the fraction of time spent engaged on a patch.

## Results

`scripts/compute_ladder.py` scores a 60 × 60 grid of (`R̄`, `κ`) at 11 mismatch levels.
Each cell is the average over 8 simulated worlds, each run for 25,000 time units. The figure above
shows five of those levels. The two below come from the same simulator.

| | |
|---|---|
| ![Mismatch load](results/poster_fig2_draft.png) | ![One forager, one patch](results/poster_agent_single_patch.png) |
| A genome tuned to the ancestral world (red) against the best genome available in each world (green). The gap is the mismatch load. | One forager homing in on one patch, and its `V` and `β` over the same run. |

## Code

```
scripts/
  params.py                     neuron constants, τ_β, the ancestral-tuned genome
  world.py                      patches, their value and their scent
  agent.py                      the forager: V/β dynamics, spiking, movement
  fitness.py                    the fitness function; scores one landscape cell
  compute_ladder.py             the fitness landscape at each mismatch level
  render_poster_hero.py         landscape filmstrip       (poster: colour_plots.png)
  render_fig2.py                mismatch load figure      (poster: mismatch load.png)
  render_agent_single_patch.py  one forager, one patch    (poster: plots.png)
  render_forager_video.py       phone video of a forager  (not on the poster)
results/
  ladder/ladder_stack.npz       the landscapes: 11 mismatch levels × 60 × 60
  fig2_data.npz                 cached simulations for the mismatch load figure
  *.png, *.mp4                  the figures and the video
run_landscape.sh                SLURM job that produced ladder_stack.npz
docs/references.md              papers behind each part of the model
```

## Reproducing

```bash
pip install -r requirements.txt

# The landscape: about 35 hours on 32 cores (resumes if interrupted)
sbatch run_landscape.sh

# The figures: seconds each
python -m scripts.render_poster_hero --stack results/ladder/ladder_stack.npz \
       --out results/poster_hero.png --rungs 0.00 0.10 0.25 0.50 1.00
python -m scripts.render_fig2                 # add --recompute to re-run its simulations
python -m scripts.render_agent_single_patch
python -m scripts.render_forager_video        # add --mismatch_index 1.0 for the modern world
```

The figure scripts rebuild the committed PNGs byte for byte, and
`render_fig2 --recompute` reproduces `results/fig2_data.npz` exactly.

The landscape is less portable. It was computed on the University of Sheffield's
Stanage cluster (Linux, Python 3.11). The simulation is chaotic over a 25,000-unit run, so
floating-point differences between platforms and library versions push individual runs
apart. Re-scoring a few cells on a Windows laptop gave values within about 0.03 of the
stored ones rather than identical values.

## References

The main ones are Charnov (1976) for the leaving rule, Hayden et al. (2011) and
Kolling et al. (2012) for the neuron, Monosov (2024) for curiosity, and De Block &
Du Laing (2010) for supernormal stimuli. The full list is in
[docs/references.md](docs/references.md).

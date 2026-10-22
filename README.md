# Information-Free Novelty

My SURE 2026 project at the University of Sheffield, supervised by James Marshall.
A forager modelled as a single neuron, in a world where novelty slowly stops meaning value.

![Fitness landscapes](results/landscapes.png)

![Mismatch load](results/mismatch_load.png)

```bash
pip install -r requirements.txt
python -m scripts.render_landscapes
python -m scripts.render_mismatch_load
```

Or open `explore.ipynb`, which walks through the model and the results and lets you run
your own foragers.

The landscape was computed on Sheffield's Stanage cluster with `run_landscape.sh`.
The poster in `docs/poster` used an earlier version of the model.

MIT licence.

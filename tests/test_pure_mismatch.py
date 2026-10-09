"""
mismatch_index must change one thing only: whether novelty tracks real value.

  python -m pytest tests          (or: python tests/test_pure_mismatch.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.world import World, PATCH_NOVELTY   # noqa: E402
from scripts.agent import Agent                   # noqa: E402
from scripts.params import BASE_PARAMS            # noqa: E402

LEVELS = (0.0, 0.1, 0.5, 1.0)


def _world(m, seed=7):
    return World(rng=np.random.default_rng(seed), mismatch_index=m,
                 space_initial_patches=False)


def test_same_seed_same_map_at_every_level():
    ref = _world(0.0)
    for m in LEVELS[1:]:
        w = _world(m)
        assert w.n_patches == ref.n_patches
        assert [(p.x, p.y, p.max_utility) for p in w.patches] == \
               [(p.x, p.y, p.max_utility) for p in ref.patches]


def test_profile_differs_only_in_mismatch_index():
    ref = dict(_world(0.0).profile)
    for m in LEVELS[1:]:
        prof = dict(_world(m).profile)
        assert prof.pop('mismatch_index') == m
        assert prof == {k: v for k, v in ref.items() if k != 'mismatch_index'}


def test_fresh_patch_smells_the_same_drained_patch_does_not():
    smells_fresh, smells_drained = [], []
    for m in LEVELS:
        p = _world(m).patches[0]
        smells_fresh.append(p.scent(novelty_gain=2.0))
        p.current_utility = 0.0
        p.n_samples = int(p.freshness_budget)
        smells_drained.append(p.scent(novelty_gain=2.0))
    assert np.allclose(smells_fresh, smells_fresh[0])  # untouched patch: no difference
    assert smells_drained[0] == 0.0                    # ancestral: drained smells drained
    assert smells_drained[-1] == 0.5 * 2.0 * PATCH_NOVELTY   # modern: novelty stays
    assert all(a < b for a, b in zip(smells_drained, smells_drained[1:]))


def test_travel_speed_does_not_depend_on_mismatch():
    speeds = {Agent(dict(BASE_PARAMS), _world(m)).travel_friction for m in LEVELS}
    assert len(speeds) == 1


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok ', name)

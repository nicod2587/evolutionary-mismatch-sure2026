"""
Neuron constants shared by every run. R_bar and kappa are the two evolving traits;
the values here are just defaults.
"""

# The slow timescale of beta. In model units 1 time step is roughly 1 ms (the
# membrane time constant is 10), so 256 sits inside the few-hundred-ms intrinsic
# timescales measured in primate PFC/ACC (Murray et al. 2014). It is held fixed
# for every run.
TAU_PFC = 256.0

BASE_PARAMS = {
    # membrane V: a leaky integrate-and-fire neuron (Hayden et al. 2011)
    'V_rest': -65.0,    # resting potential (mV)
    'theta': -55.0,     # firing threshold (mV), so it takes a drive of 10 to fire
    'tau_v': 10.0,      # membrane time constant

    # beta: the opportunity-cost signal (Kolling et al. 2012)
    'tau_pfc': TAU_PFC,

    # the two heritable traits (overwritten per genome, see fitness.Genome)
    'R_bar': 6.0,       # how rich the forager expects the environment to be
    'kappa': 1.0,       # how much novelty adds to a patch's perceived value
}

# The mismatch level the "ancestral-tuned" forager is tuned to. Kept at the
# poster-v1 value so the two model versions stay comparable.
ANCESTRAL_MISMATCH = 0.10

# The forager tuned to the ancestral world (red in the mismatch-load figure, and the
# agent in the single-patch figure and the video): the mean (R_bar, kappa) of the top
# 10% of the box at ANCESTRAL_MISMATCH in the 40x40 ladder. Print it with
# python -m scripts.compute_ladder --summarise <stack>. The poster used (14.7, 4.33).
ANCESTRAL_GENOME = (8.7, 2.98)

"""
Neuron parameters shared by every simulation.

The forager's neuron has two state variables, the membrane potential V and the
opportunity-cost signal beta. The numbers below set how those behave. Two of them,
R_bar and kappa, are the heritable traits that the fitness landscape varies; the
values here are only defaults so that a plain BASE_PARAMS dict runs on its own.
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

# The "ancestral-tuned" forager (the red line in the mismatch-load figure, and the
# agent in the single-patch figure and the video): the mean (R_bar, kappa) of the
# top 10% fittest cells inside the biological box (fitness.BOX_R, fitness.BOX_K)
# of the ANCESTRAL_MISMATCH landscape. compute_ladder prints it at the end of a
# run (or: python -m scripts.compute_ladder --summarise <stack>).
#
# TODO after the first pure-v2 ladder: replace with the printed value. (14.7, 4.33)
# is the poster-v1 value, from the old model.
ANCESTRAL_GENOME = (14.7, 4.33)

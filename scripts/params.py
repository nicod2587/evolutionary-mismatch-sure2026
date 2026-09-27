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

# The genome used for the "ancestral-tuned" forager on the poster (the red line in
# the mismatch-load figure, and the agent in the single-patch figure). It is the
# mean (R_bar, kappa) of the top 10% fittest cells inside the biological box
# (fitness.BOX_R, fitness.BOX_K) of the mismatch = 0.10 landscape.
ANCESTRAL_GENOME = (14.7, 4.33)

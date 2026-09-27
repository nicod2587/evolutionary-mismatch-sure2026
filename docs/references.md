# References

## Cited on the poster

1. **Charnov, E. L. (1976).** Optimal foraging, the marginal value theorem.
   *Theoretical Population Biology, 9*(2), 129–136.
   The patch-leaving rule: stay while a patch beats the environment's average, then move on.

2. **Kilpatrick, Z. P., Davidson, J. D., & El Hady, A. (2020).** Normative theory of
   patch foraging decisions. *arXiv*:2004.10671.
   Foraging theory for patches that don't deplete, which is how an endless feed behaves.

3. **Poli, F., O'Reilly, J. X., Mars, R. B., & Hunnius, S. (2024).** Curiosity and the
   dynamics of optimal exploration. *Trends in Cognitive Sciences, 28*, 441–453.

4. **Hayden, B. Y., Pearson, J. M., & Platt, M. L. (2011).** Neuronal basis of sequential
   foraging decisions in a patchy environment. *Nature Neuroscience, 14*(7), 933–939.
   Monkey dACC neurons rise to a threshold before the animal leaves a patch. The model's
   integrate-and-fire membrane `V` follows this.

5. **Kolling, N., Behrens, T. E. J., Mars, R. B., & Rushworth, M. F. S. (2012).** Neural
   mechanisms of foraging. *Science, 336*(6077), 95–98.
   dACC tracks the value of leaving, the basis for the opportunity-cost signal `β`.

## Background for the model

6. **Pirolli, P., & Card, S. (1999).** Information foraging. *Psychological Review,
   106*(4), 643–675.
   Information scent: people follow cues about where information is, which is what the
   forager's perceived scent stands for.

7. **McNamara, J. M., & Dall, S. R. X. (2010).** Information is a fitness-enhancing
   resource. *Oikos, 119*(2), 231–236.
   Why the fitness counts real value gained, not how appealing a patch looked.

8. **Monosov, I. E. (2024).** Curiosity: primate neural circuits for novelty and
   information seeking. *Nature Reviews Neuroscience, 25*, 195–208.
   Primate circuits that drive exploration of novel things for their own sake. The basis
   for `κ`.

9. **De Block, A., & Du Laing, B. (2010).** Amusing ourselves to death? Superstimuli and
   the evolutionary social sciences. *Philosophical Psychology, 23*(6), 821–843.
   Supernormal stimuli in modern environments: exaggerated cues hijack evolved responses.
   The basis for letting novelty come apart from value as mismatch rises.

10. **Lim, V. P. Y., Yong, J. C., & Tan, Y. W. (2025).** Artificial intelligence,
    fundamental motives, and evolutionary mismatch. *Evolutionary Behavioral Sciences.*

11. **Li, N. P., van Vugt, M., & Colarelli, S. M. (2018).** The evolutionary mismatch
    hypothesis: implications for psychological science. *Current Directions in
    Psychological Science, 27*(1), 38–44.

## Why τ_β is 256

The membrane time constant is 10, which makes one model time unit roughly 1 ms, so
τ_β = 256 is about a quarter of a second. That sits inside the intrinsic timescales
measured in primate PFC and ACC:

12. **Murray, J. D., Bernacchia, A., Freedman, D. J., Romo, R., et al. (2014).** A
    hierarchy of intrinsic timescales across primate cortex. *Nature Neuroscience,
    17*(12), 1661–1663.

13. **Li, S., & Wang, X.-J. (2022).** Hierarchical timescales in the neocortex:
    mathematical mechanism and biological insights. *PNAS, 119*(6), e2110274119.

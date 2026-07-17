# References

Grounding literature for the evolutionary-mismatch information-foraging model.
Each entry notes what it grounds in the model.

1. **Charnov, E. L. (1976).** Optimal foraging, the marginal value theorem.
   *Theoretical Population Biology, 9*(2), 129–136.
   — *Patch structure and the giving-up rule; the leaving decision and the
   opportunity-cost fitness term.*

2. **Pirolli, P., & Card, S. (1999).** Information foraging.
   *Psychological Review, 106*(4), 643–675.
   — *Information-foraging theory and "information scent"; the neuron's
   perceived drive I.*

3. **McNamara, J. M., & Dall, S. R. X. (2010).** Information is a
   fitness-enhancing resource. *Oikos, 119*(2), 231–236.
   — *Information value = uncertainty reduction, and it goes net-negative once
   you pay to acquire it; the info_yield and cost terms of fitness.*

4. **Hayden, B. Y., Pearson, J. M., & Platt, M. L. (2011).** Neuronal basis of
   sequential foraging decisions in a patchy environment.
   *Nature Neuroscience, 14*(7), 933–939.
   — *Primate dACC rise-to-threshold implementation of the MVT; the V
   integrate-and-fire foraging neuron.*

5. **Kolling, N., Behrens, T. E. J., Mars, R. B., & Rushworth, M. F. S. (2012).**
   Neural mechanisms of foraging. *Science, 336*(6077), 95–98.
   — *dACC as an opportunity-cost / value-of-leaving signal; β_PFC.*

6. **Shenhav, A., Straccia, M. A., Cohen, J. D., & Botvinick, M. M. (2014).**
   Anterior cingulate engagement in a foraging context reflects choice
   difficulty, not foraging value. *Nature Neuroscience, 17*(9), 1249–1254.
   — *Noted counter-reading of dACC; we take the Kolling/Hayden foraging-value
   interpretation.*

7. **Monosov, I. E. (2024).** Curiosity: primate neural circuits for novelty and
   information seeking. *Nature Reviews Neuroscience, 25*, 195–208.
   — *PRIMARY grounding for κ: primate neural circuits that drive exploration of
   novel stimuli for their own sake, independent of value. The curiosity-gain
   drive the modern husk exploits.*

7b. **Bromberg-Martin, E. S., & Monosov, I. E. (2020).** Neural circuitry of
   information seeking. *Current Opinion in Behavioral Sciences, 35*, 62–70.
   — *Circuits that value information distinctly from primary reward; supports κ
   as a separable non-instrumental drive.*

7c. **Wang, M. Z., & Hayden, B. Y. (2019).** Monkeys are curious about
   counterfactual outcomes. *Cognition, 189*, 1–10.
   — *(Secondary κ support.) Primates pay for non-instrumental information;
   curiosity scales with Shannon entropy.*

8. **Lim, V. P. Y., Yong, J. C., & Tan, Y. W. (2025).** Artificial intelligence,
   fundamental motives, and evolutionary mismatch. *Evolutionary Behavioral
   Sciences.* *(Confirmed 2026-07-17: published 2025 in Evolutionary Behavioral
   Sciences — the abstract text still reads "Lim and Tan, 2024" and should be
   updated to Lim, Yong & Tan 2025 before final submission.)*
   — *Supernormal stimuli: ancestrally-adaptive processing rules become
   non-adaptive in a modern/manipulated environment; the "glowing husk" and the
   mismatch itself.*

9. **De Block, A., & Du Laing, B. (2010).** Amusing ourselves to death?
   Superstimuli and the evolutionary social sciences. *Philosophical Psychology,
   23*(6), 821–843.
   — *Academic grounding for the supernormal-stimulus (Tinbergen) lineage applied
   to modern environments: exaggerated cues hijack evolved response rules. The
   conceptual basis for the h = 1 − mismatch scent-decoupling.*

10. **Li, N. P., van Vugt, M., & Colarelli, S. M. (2018).** The evolutionary
   mismatch hypothesis: implications for psychological science. *Current
   Directions in Psychological Science, 27*(1), 38–44.
   — *The general evolutionary-mismatch framing: adaptations calibrated to
   ancestral environments misfire in novel modern ones. Grounds the paper's
   central thesis at the level above the specific foraging mechanism.*

### Prefrontal intrinsic timescale (grounds τ_pfc)

The opportunity-cost accumulation timescale τ_pfc is not identified by selection
(a flat direction), so it is fixed at 256 (≈256 ms, given the model's membrane
constant τ_v = 10 sets 1 sim-unit ≈ 1 ms) — a value inside the measured PFC/ACC
intrinsic-timescale range.

11. **Murray, J. D., Bernacchia, A., Freedman, D. J., Romo, R., et al. (2014).**
   A hierarchy of intrinsic timescales across primate cortex. *Nature
   Neuroscience, 17*(12), 1661–1663.
   — *The anchor: dACC/PFC are among the slowest cortical areas (upper hundreds
   of ms); places τ_pfc ≈ 256 ms in range.*

12. **Li, S., & Wang, X.-J. (2022).** Hierarchical timescales in the neocortex:
   mathematical mechanism and biological insights. *PNAS, 119*(6), e2110274119.
   — *Cortical intrinsic timescales span tens to hundreds of ms; the bracket
   τ_pfc sits within.*

---

*Notes: (1) entry 8's author/year is confirmed — update the abstract text from
"Lim & Tan 2024" to "Lim, Yong & Tan 2025". (2) On McNamara & Dall 2010: their
thesis is that information value is NON-negative; our model's net-negative value
comes from adding an explicit acquisition cost c — state this as an extension,
not a restatement, so the reading is precise. (3) Volume/page details are
standard for these well-known papers but worth a final cross-check against a
reference manager before submission.*

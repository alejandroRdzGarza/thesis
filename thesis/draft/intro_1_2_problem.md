# 1.2 Problem statement and scope  (~350 words)

## Draft

A vision-language-action policy that is unsafe can be made safe at runtime by filtering its
output. A control barrier function projects each proposed action onto a set that provably avoids
the obstacle, and the resulting system behaves acceptably. This is the standard arrangement, and it
works: on the benchmark used here it reduces collisions from 82.5% of episodes to 13.3%.

It also has a cost that is easy to overlook. The filter must run on every control step for the life
of the deployment. It consumes inference budget, it requires obstacle geometry to be available at
runtime, it constitutes an additional component that can fail, and — as this work measures — the
projection it applies degrades task competence once the policy no longer needs it. Safety obtained
this way is rented, not owned.

**The hypothesis of this thesis is that the behaviour a filter induces can instead be absorbed into
the policy**, so that the policy is safe with the filter switched off. If it can, safety becomes a
property of the learned controller rather than a runtime dependency.

The work establishes that this is possible, and then asks the more useful question: **under what
conditions?** The answer turns out to be restrictive in ways that are measurable, and characterising
those conditions — rather than reporting a single positive result — is the substance of the
contribution.

**Scope.** The claims are bounded in four respects, each stated here so that later results are read
correctly.

*Environment.* All experiments use SafeLIBERO, a simulated tabletop manipulation benchmark. No
physical robot is involved and no claim of sim-to-real transfer is made.

*Policy.* A single vision-language-action model, π0.5, with a flow-matching action head. Results
concerning action chunking and denoising-time intervention depend on that architecture.

*Information.* The barrier is constructed from ground-truth obstacle geometry supplied by the
simulator. A deployed system would estimate that geometry from perception and inherit the resulting
error, so the figures reported here are an upper bound for this class of filter.

*Statistical.* Each trained policy is a single training run; seed variance is unquantified.
Evaluation uses held-out initial states throughout, with 120 rollouts per evaluated policy and 95% confidence
intervals.


---

[CITE — Chapter 1 placement, from papers/paper_relevance_matrix.csv]

- **CITE: aegis2025** — the introduction's motivating fact: runtime shielding works and carries a
  permanent deployment burden. This is the setting the thesis inherits, not a competitor.
- **CITE: rt_vla** — deployment-efficiency motivation for distillation generally. Matrix caution,
  which applies directly here: use only if measured latency or compute savings are reported.
  Otherwise say the shield is *absent*, not that a speedup was demonstrated. No latency was
  measured in this work, so the weaker phrasing is the correct one.
- **CITE: vita_vla** — establishes that action-level knowledge can be distilled into a VLM at all,
  which makes the premise plausible before any safety claim is made. Do not cite it for safety.
- **CITE: physicalintelligence2025pi05, pi0** — identify the base policy and its lineage.
- **CITE: liu2023libero** — the benchmark family.

[FIGURE 1.1 — OPTIONAL but strong as a frontispiece: the same scene under base pi0.5 (collides),
shielded pi0.5 (avoids, shield running), and the distilled policy (avoids, no shield). Three
columns, one row. States the thesis in a single image before any prose. Shares assets with
Figure 4.3. | videos/ | TO MAKE]

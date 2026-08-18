# Chapter 6 — Conclusion (DRAFT, ~300 words)

This work asked whether the safety a runtime filter provides can be made a property of the policy
itself, so that the filter is no longer required at deployment.

It can. Distilling a vision-language-action policy on its own shielded rollouts reduced collisions
from 82.5% to 19.2% of episodes **with no shield running at inference**, while raising task success
from 58.3% to 82.5%, on held-out initial states with disjoint confidence intervals. The distilled
policy recovers most of the shield's safety benefit (CAR 80.8% against 86.7%) without its runtime
cost, and one round of distillation was sufficient.

The more useful finding is that this does not hold generally. A privileged sampling-based motion
planner, safe by construction and given full scene geometry, produced demonstrations from which the
same policy learned essentially nothing: a student trained on them was statistically
indistinguishable from the undistilled baseline, under a comparison matched on gradient steps,
hyperparameters and evaluation. **What transfers is correction of the policy's own behaviour, not
instruction from a foreign controller.**

Three further results bound the claim. The improvement comes from the shield's corrections rather
than from filtering for successful episodes, established by a matched control in which
success-filtering alone achieved nothing. Safety transfers through two separable channels, since
the distilled policy avoids with its arm links better than the shield that taught it — which
imitation alone cannot explain. And four alternative mechanisms were tested and rejected:
scalar-reward reinforcement learning, guided sampling, safety expressed in language, and best-of-N
selection, the last of which showed that a policy's action samples from a given state agree on
whether they collide, so safety is state-determined rather than sample-determined.

Together these say something narrower and more useful than "safety can be learned": **safety
mechanisms are bounded by what they can perceive and by whose behaviour they correct.** Filters
teach precisely what they model well; outcome selection reaches somewhat further; and neither
language nor generative uncertainty provided a usable signal at all.

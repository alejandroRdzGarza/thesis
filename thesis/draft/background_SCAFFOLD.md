# Chapter 2 — Background and Related Work (SCAFFOLD, ~1,500 words target)

> **READ THIS FIRST.** Unlike the other draft chapters, this one is NOT written. Everything below
> that describes *someone else's work* is a placeholder marked **[SOURCE]** and must be written
> from the paper itself. I cannot verify claims about AEGIS, SafeDAgger, Flow-GRPO or π0.5's
> architecture from this codebase, and inventing them is exactly the failure mode a viva exposes.
> What IS written here is the argumentative skeleton — what each section must establish so that
> Chapters 3–5 land — plus the parts grounded in this project's own measurements.
>
> Criterion 1 (background, aims, organisation) is the weakest area of this thesis. This chapter is
> where those marks are won or lost, and it needs reading time, not drafting time.

---

## 2.1 Vision-language-action models (~350 words)

**What this section must establish:** what a VLA is, what π0.5 specifically does, and the two
architectural facts Chapters 3–5 depend on — flow-matching action generation and action chunking.

**[SOURCE] π0.5 / π0 architecture.** PaliGemma-class vision-language backbone; action expert;
parameter count; training corpus. Cite the π0.5 technical report.

**[SOURCE] Flow matching for action generation.** How the action head samples: an initial noise
sample integrated along a learned velocity field over N denoising steps. State the ODE form, since
Section 3.6 trains against this same loss and Chapter 4's guided-sampling experiment intervenes
inside this integration.

**[SOURCE] Action chunking.** The policy emits H actions per query and executes a prefix before
replanning. Note the stated motivation (smoothness, inference cost). Chapter 4 uses the chunk as a
lookahead horizon for safety evaluation, which is a different use than the one it was introduced
for — worth flagging here so that later framing is not a surprise.

**Grounded in this work — safe to write now:** the observation space is two 224×224 RGB images plus
an eight-dimensional proprioceptive vector (end-effector position, axis-angle orientation, two
gripper joint positions) and a language instruction. It contains **no joint angles**, which
Chapters 3 and 5 return to.

---

## 2.2 Control barrier functions for manipulation (~350 words)

**What this section must establish:** the formal object, the QP that uses it, and precisely what
the guarantee is — so that Chapter 4's residual collisions read as expected rather than anomalous.

**[SOURCE] CBF definition.** h(x) ≥ 0 defining the safe set; the forward-invariance condition
ḣ ≥ −α(h); the class-K function. Cite Ames et al.

**[SOURCE] QP safety filter.** The standard min-deviation formulation and why it is the canonical
way to apply a CBF to an arbitrary nominal controller.

**Write this yourself, and prominently — the guarantee's preconditions.** Forward invariance holds
in continuous time, with an exact model, and only for the bodies actually represented in h. Every
one of those is violated in deployment: the controller runs at a fixed rate with tracking lag, the
robot is approximated by spheres, and Section 3.2's link model is coarse. This paragraph is what
makes Section 4.5 a *prediction* rather than an excuse, so it belongs here rather than in the
results.

**[SOURCE] Prior CBF use in manipulation.** Whole-body versus end-effector formulations, and the
computational cost that usually motivates the simplification this work also makes.

---

## 2.3 Safety filters for VLAs (~400 words)

**What this section must establish:** the immediate prior work, and the specific gap this thesis
addresses.

**[SOURCE] AEGIS.** Threat model, method, benchmark, headline numbers. This matters more than the
other citations for two reasons: SafeLIBERO and the collision metric come from this line of work,
and this thesis reproduces its unshielded baseline closely (measured CAR 17.5% / TSR 58.3% against
their reported 17.3% / 58.9%, averaged over the three suites used here). State that agreement
explicitly — it is real external validation. Do NOT claim agreement on the shielded figures: this
shield reaches CAR 86.7% against their 71.9%, and there are now two verified reasons rather than
one. First, their formulation "solely constrains the end-effector" by their own statement, and
their limitations section notes that unconstrained kinematic links may consequently collide; the
barrier here also constrains links 3–7 and the hand, and Section 4.5 shows arm links are exactly
where residual collisions concentrate. Second, this barrier reads ground-truth geometry while
theirs works from vision-language grounding and fused depth, to which they attribute their own
residual failures. Both differences favour this shield, so the comparison is an upper bound for the
class rather than an improvement over their method.

Also state, because Section 4.7 depends on it, that AEGIS independently reports "safety-induced
distribution shift": enforcement can drive the policy into out-of-distribution states where it
behaves erratically and fails to recover. That is the same effect measured here as the
shield-stacking success loss, and it is the strongest available argument for internalisation.

**[SOURCE] Other runtime safety filtering for learned policies.** Shielded RL, safe MPC, and
action-space filtering more broadly.

**The gap, which you can argue from your own results.** Runtime filtering works and is well
established. Its cost is that it is permanent: inference budget on every step, obstacle geometry
required at deployment, an additional failure point, and — as Section 4.3 measures — a *capability*
cost once the policy no longer needs it, since projection displaces competent actions off the
policy's distribution. The question this thesis asks is whether the behaviour can be moved into the
policy instead.

**[RESOLVED — novelty check completed 2026-08-19.]** The two closest candidates were read directly
and neither forecloses the claim, but each narrows how it should be phrased.

*AEGIS* retains its constraint layer at inference by construction; the amortization question is
simply not asked there. Its QP runs every step. Nothing in that work bears on what the corrections
would teach if used as training data.

*ROAD-VLA* is the closer call and the more instructive one. It builds an advantage-guided teacher
and distils it into the policy, so at the level of "distil a corrective signal into a VLA" it is
prior art and must be cited as such. Two features keep the contribution here distinct. Its teacher
is a logit-perturbed copy of the *student itself* (their Eq. 10), so no foreign controller appears
anywhere in the work; and its policy-improvement bound is explicitly conditional on the teacher
remaining proximal to the current policy. The three privileged teachers it rejects are all
*textual* — its Section 4.2.1 is titled "Text-guided $\mathcal{I}$", and the rejected variants are
language templates scoring 75.8% (MCTS PI, against a PPO baseline of 87.2%) and 4.68% (RelSpatial
and Plan+RelSpatial), against 91.5% for the full method.

So the honest positioning is: ROAD-VLA's proximity principle *predicts* the planner-distillation
failure reported in Section 4.6, but the paper never tests a foreign low-level controller supplying
well-formed actions. That experiment is this thesis's contribution, and it should be presented as
filling a gap their framework anticipates rather than as a disagreement with it. Chapter 1.3 must
be phrased to match — claim the empirical boundary, not the idea of distilling corrections.

What remains genuinely unclaimed by either: distilling a *control-theoretic safety filter* into a
VLA and removing it at deployment.

---

## 2.4 Imitation and RL for safety internalisation (~400 words)

**What this section must establish:** the two families of methods that could internalise safety,
what is known about each, and where this work sits.

**[SOURCE] DAgger and SafeDAgger.** The distribution-shift problem, the interactive-expert
solution, and SafeDAgger's insight of learning *when* the safe policy is needed. This is the direct
ancestor of the method in Chapter 3 and should be presented as such rather than discovered later.

**[SOURCE] Filtered behaviour cloning / self-improvement.** Training on a policy's own successful
trajectories (STaR, RAFT, rejection-sampling fine-tuning). This is the alternative explanation that
Section 4.6's control ablation exists to rule out, so the reader must meet it here to appreciate
why that control was necessary.

**[SOURCE] Policy-proximal corrective supervision.** SAFE-GIL, IntervenGen, and ROAD-VLA. This is
the lineage Section 5.1's explanation belongs to and it currently has no home in the chapter, which
is a gap: the thesis's central mechanism claim — that supervision must correct errors on or near
the learner's own rollout distribution — is presented in Chapter 5 as though it were novel, when it
is the shared premise of three prior methods. Presenting it here as established, and Chapter 5's
contribution as *testing* it against a privileged planner, is both more accurate and a stronger
position. SAFE-GIL and IntervenGen still need verifying against their PDFs before this is written.

**[SOURCE] RL for VLAs.** Flow-GRPO and related on-policy methods for flow-matching policies; what
they achieve and at what compute cost. Chapter 5 argues these fail here for a structural reason
(exploration and credit assignment share the sampling-noise knob) — that argument needs the method
described accurately first.

**Positioning.** State plainly what is and is not new. The method is shielded imitation, in the
SafeDAgger lineage; it is not a new algorithm. The contribution is empirical: which teachers
transfer and which do not, what channel the transfer occurs through, and what bounds it. Claiming
methodological novelty here would be both wrong and easy to falsify.

---

## Checklist before this chapter is finished

- [ ] Every **[SOURCE]** replaced by prose written from the actual paper
- [x] The **[VERIFY]** novelty check in 2.3 completed — AEGIS and ROAD-VLA both read directly.
      Chapter 1.3 still needs adjusting to match: claim the empirical boundary, not the idea of
      distilling corrections, which ROAD-VLA has prior art on.
- [x] AEGIS's collision metric checked — it does **not** match. Their CAR is "the percentage of
      strictly collision-free episodes", with no numeric threshold, and their ETS is "the average
      episode length (including timeouts)", a different quantity from the first-success-step-over-
      successes used here. Section 3.1 now presents the L1 >1 mm test as this work's own
      operationalisation rather than as theirs.
- [ ] SafeDAgger cited explicitly as the method's ancestor, in 2.4 and again in Chapter 3
- [ ] Forward-invariance preconditions written in 2.2, since 4.5 depends on them

# Chapter 2 — Background and Related Work

<!-- Written as a single argument, not a survey. The chain is:
     2.1 what the policy is and why it is unsafe  ->  2.2 the control-theoretic tool
     ->  2.3 runtime shielding for VLAs (the baseline, and its permanent cost)
     ->  2.4 learned and CBF-guided safety (still optimizes online)
     ->  2.5 correction-distribution work (what actually makes data transfer)
     ->  2.6 safe RL for VLAs (the alternative route, scoped not dismissed)
     ->  2.7 the exact gap.
     Every claim about a paper below marked VERIFIED was read in the PDF; see papers/FINDINGS.md.
     [CITE: key] markers name what each source is being used for and the scope it licenses. -->

## 2.1 Vision–language–action models and the safety problem they inherit

Vision–language–action models map camera observations and a natural-language instruction directly
to continuous robot actions, and have become the dominant paradigm for general-purpose
manipulation. The policy studied here, $\pi_{0.5}$, pairs a pretrained vision–language backbone
with a flow-matching action expert that emits chunks of continuous actions, and is trained by
heterogeneous co-training across robot demonstrations, semantic subtask prediction, and
web-derived multimodal data  [CITE: physicalintelligence2025pi05 for the architecture, the
hierarchical semantic-subtask-then-action decomposition and the co-training recipe; CITE: pi0 for
the flow-matching action expert that $\pi_{0.5}$ inherits].

What matters for this thesis is what that training does *not* optimise. These models are built and
evaluated for breadth — novel objects, unseen homes, long horizons — and their reported results are
generalisation results. Nothing in the objective penalises contact with an object that is not the
target, and nothing in the data distribution reliably demonstrates avoidance, because human
demonstrators rarely produce near-collisions worth imitating. Broad task competence is therefore
not evidence of physical safety, and should not be read as such  [CITE:
physicalintelligence2025pi05 — cite for the *scope* of their claims, which is generalisation, and
note explicitly that no collision-safety claim is made there].

The empirical consequence is stark. On the benchmark used throughout this thesis, the unmodified
policy collides with a non-target obstacle in more than eight of every ten episodes (Section 4.2).
The problem this thesis addresses is therefore not that VLAs are bad at manipulation; it is that
competence and safety are separate properties, and only the first has been optimised.

## 2.2 Control barrier functions

Control theory offers a mature answer to constraint satisfaction. A control barrier function
encodes a safe set $\mathcal{C}$ as the superlevel set of a differentiable function $h$, and
converts safety into a condition that is affine in the control input: any $u$ satisfying
$\dot h(x,u) \ge -\alpha(h(x))$ renders $\mathcal{C}$ forward invariant, so a system that begins
safe remains safe  [CITE: cbf_theory — cite for Definitions 1–2, the forward-invariance theorem,
and the CBF-QP form. This is the definitional source for Section 3.3 and must not be cited as
evidence about learned policies].

Because the condition is affine in $u$, it can be imposed as one linear inequality per constraint
inside a quadratic program that stays as close as possible to a nominal input,

$$u^\star = \arg\min_u \tfrac12\|u - u_{\text{nom}}\|^2 \quad \text{s.t.} \quad \nabla h^\top(f + gu) \ge -\alpha(h).$$

This is the object the rest of the thesis calls a *shield*: a filter that accepts whatever a
nominal controller proposes and returns the minimum modification that satisfies the constraint. Two
properties make it attractive as a teacher rather than merely as a guard. It is agnostic to how the
nominal action was produced, so it composes with a 3-billion-parameter policy it knows nothing
about; and because it minimises $\|u - u_{\text{nom}}\|$, its output is not an arbitrary safe
action but *the closest safe action to what the policy wanted*, which is precisely the form a
supervised learning target should take.

Two caveats bound what may be claimed later. The guarantee is continuous-time and assumes known
control-affine dynamics, so a discrete-time implementation on a contact-rich manipulator satisfies
it only approximately (Section 3.3). And forward invariance is a property of the *closed loop
containing the QP*. Remove the QP and the guarantee is gone. Chapter 6 is careful on this point,
and Section 2.4 returns to exactly how much of it can be recovered.

## 2.3 Runtime shielding for vision–language–action models

The immediate prior work applies this machinery to VLAs directly. AEGIS inserts a safety-constraint
layer between $\pi_{0.5}$ and the robot: a vision–language module grounds obstacle geometry from
observations and fused depth, and a CBF-QP modifies the policy's proposed action whenever it
violates the resulting constraint. It also contributes SafeLIBERO, the benchmark used here, which
augments LIBERO manipulation tasks with non-target obstacles at two difficulty levels  [CITE:
aegis2025 for the method, the benchmark and the metric definitions; CITE: liu2023libero for the
underlying task suite and BDDL scene specification. VERIFIED].

The approach works, and this thesis reproduces its unshielded baseline closely — collision
avoidance of 17.5% against their 17.3%, task success 58.3% against 58.9%, on the three suites used
here (Section 4.2). That agreement is real external validation of the setup and is reported as such.

Two features of that work motivate everything that follows. The first is that the layer is
*permanent*: it runs a QP at every control step, requires obstacle geometry at deployment, and adds
a component that can itself fail. The authors report its cost as 0.356 ms per step, which is modest,
but the deeper cost is architectural rather than computational — the deployed system is a policy
plus an apparatus, and the apparatus never goes away.

The second is a limitation the authors state themselves, and which this thesis measures directly.
Enforcing safety at inference can push the policy into states its training never covered, where
"the policy may behave erratically and fail to recover" — an effect they name *safety-induced
distribution shift*. Section 4.3 measures exactly this as a success cost of shielding, and it is
the strongest argument that a permanent filter is not the end state one should want. A policy that
has internalised the constraint is never driven off its own distribution by the act of being made
safe  [CITE: aegis2025 — cite their limitations section explicitly; this is corroboration of our
measurement by the baseline's own authors, and it should be presented as such rather than as our
observation alone. VERIFIED].

One further difference matters for reading Chapter 4. Their formulation, by their own statement,
"solely constrains the end-effector", and they note that unconstrained kinematic links may
consequently collide. The barrier built in Section 3.2 additionally constrains links three through
seven and the hand — an extension whose consequences Section 4.5 quantifies.

## 2.4 Learned safety representations, and why they still optimise at deployment

A parallel line of work removes the requirement for an analytically specified barrier by learning
one. ConBaT trains a causal transformer with a safety critic from binary safe/unsafe labels,
requiring no hand-designed constraint, and operates in embedding space rather than state space so
it applies where geometry is unavailable  [CITE: conbat — cite for the learned-critic construction
and the minimal labelling requirement. VERIFIED]. Latent Policy Barrier treats the latent manifold
of expert demonstrations as an implicit barrier, training a dynamics model on both expert data and
the policy's own rollouts, then detecting and correcting departures from that manifold  [CITE:
latentpolicybarrier — cite for treating the expert latent distribution as an implicit barrier and
for using policy rollout data without success labels or human annotation. VERIFIED].

Latent Policy Barrier is the closer of the two to this thesis, because its recovery data comes from
the policy's own rollout distribution — checkpoints saved during training, executed to gather
deviations "without requiring explicit success labels, task rewards, or additional human
teleoperation." That is the same instinct pursued here.

But both retain the property that motivated this work. ConBaT performs "lightweight online
optimization" at deployment, rectifying actions by backpropagation through its critic until the CBF
conditions hold. Latent Policy Barrier "performs inference-time steering in latent space", running
gradient-based corrections through its dynamics model at every step, and explicitly advertises
"plug-and-play compatibility with off-the-shelf pre-trained policies… without requiring policy
retraining or fine-tuning." Their goal is the inverse of the one here: they improve a frozen policy
by adding machinery, where this thesis changes the policy so the machinery can be discarded.

Between these sits the theoretical question of how much safety can transfer into weights at all.
Cosner et al. establish conditions under which imitating a CBF-based expert yields a learned
controller with input-to-state-safety guarantees. The result is genuine but demanding: the expert
must be a *robust* CBF-QP with explicit disturbance margins, the dynamics must be known and
control-affine, and the learned controller must satisfy a CBF-compliancy condition requiring data
density along the safe-set boundary, bounded learning error, and a known Lipschitz constant — and
even then the guarantee holds on an *expanded* safe set that degrades with data sparsity and
learning error  [CITE: il_with_cbf — cite for Definition 6 and Theorem 2, and for the explicit list
of preconditions. This is the correct citation for what this thesis does NOT establish. VERIFIED].

None of those conditions is met here, and Section 5.3 says which and why. Their Remark 1 is
nonetheless the cleanest available statement of the thesis's own logic: the goal is "to transfer
safety guarantees from the expert controller to the learned controller rather than to exactly mimic
the expert behavior." Safety transfer is not imitation fidelity — a distinction Section 4.5
observes empirically, where the distilled policy avoids arm-link collisions better than the teacher
that supervised it.

## 2.5 What makes corrective data transfer

The central question of this thesis is not whether a policy can be trained on corrected actions,
but which corrections teach. Three lines of work converge on an answer, and together they supply
the hypothesis that Chapter 4 tests.

SAFE-GIL injects reachability-computed adversarial disturbances during data collection, steering a
fixed expert toward the states a learner's errors would produce so that its recovery behaviour is
recorded. Their ablation is the decisive one: Gaussian noise, uniform noise, and DART all fail to
reproduce the safety benefit, so it is the *targeting* of the disturbance, not augmentation as
such, that matters  [CITE: safegil — cite the method and especially the noise-baseline ablation.
VERIFIED]. IntervenGen reaches the same place differently, rolling out the current policy so that
recorded mistakes are its genuine failures, then transforming human recovery segments onto the
states reached  [CITE: intervengen — cite for corrective coverage over policy-mistake
distributions, and their result that ten interventions expanded synthetically outperform a hundred
collected directly. VERIFIED].

Both are worth stating carefully, because they discipline the claim this thesis makes. In SAFE-GIL
the expert is an MPC or PID controller; in IntervenGen the recovery actions come from a human
teleoperator. **Neither teacher is the learner, and in both cases safety transfers.** What the two
share is not the identity of the teacher but the location of the supervision: corrections are
supplied at states the learner itself would reach. The hypothesis this thesis inherits from them is
therefore about state coverage, and Section 4.4's planner comparison is a test of it rather than a
demonstration that foreign controllers cannot teach.

ROAD-VLA carries the same principle into VLAs. It constructs a proximal teacher by perturbing the
student's own action-token logits with calibrated advantage estimates, and distils that teacher
along the student's on-policy rollouts, with a policy-improvement bound that holds only while the
teacher remains close to the current policy. Its negative result is frequently summarised as a
rejection of privileged teachers; it is not. Every rejected variant is *textual* — a retrieved
action hint, an egocentric spatial description, and a task plan — and the accepted teacher is
derived from the student itself, so no foreign low-level controller is tested anywhere in the work.
Their explanation is that embodied post-training erodes the backbone's in-context language ability
and that "a modality gap prevents the discrete text from providing the precise grounding required
for continuous control"  [CITE: roadvla — cite the teacher construction, the proximity condition on
their bound, and their Table 2 text-privileged ablation. Their explanation of the text failure is
the right citation for Section 4.7's language-channel negative. VERIFIED].

Finally, one result establishes that the protocol used here is not sufficient on its own. Guerrier
et al. train a reinforcement-learning agent with a CBF filter active throughout training and then
remove it. Safety is not retained: the agent collides once the filter is disabled, and their critic
assigns high value *inside* the obstacle. Their agent's actions were overridden, but its objective
remained a scalar reward, so no gradient ever pointed toward the corrected action. Their curriculum
variant, which blends filter and policy actions with the filter's weight decayed to zero, does
internalise the constraint  [CITE: guidedbyguardrails — cite the CBF-Filter negative as evidence
that shield exposure alone does not transfer safety, and CBF-Decay as a non-imitation route that
does. Their scalar-reward findings belong in Section 4.7. VERIFIED].

That negative is what makes this thesis's question non-trivial. Being trained under a shield does
not make a policy safe; what the policy is trained *on* decides the outcome.

## 2.6 Policy-level safety learning as an alternative route

Runtime filtering and corrective imitation are not the only options, and this thesis does not claim
they are. SafeVLA formulates VLA safety as a constrained Markov decision process, eliciting unsafe
behaviour and optimising the policy under explicit constraints  [CITE: safevla]. SafeDojo performs
model-based safe reinforcement learning inside an action-conditioned video world model, estimating
imagined task progress and safety cost separately and optimising them with Lagrangian constrained
GRPO — and evaluates on SafeLIBERO, making it the most direct methodological alternative in this
literature  [CITE: safedojo].

These matter for calibration rather than contrast. Chapter 4 reports a negative result for
scalar-reward reinforcement learning in this setting; neither of these methods is scalar-reward
reinforcement learning, and both supply substantially richer constraint signals. The negative is
therefore a result about the reward design tested, and Section 4.7 and Chapter 6 scope it that way.
The implementation of that ablation draws on policy-gradient methods adapted to flow-matching
samplers, which belong to the methods appendix rather than to this argument  [CITE: flow_grpo,
grpo — implementation lineage only; neither contains robotics or safety evidence].

## 2.7 The gap

The literature above establishes five things. Runtime CBF shielding makes VLAs measurably safer and
is the state of the art for this benchmark, at the cost of a permanent inference-time component and
a distribution shift its own authors document. Learned barriers relax the need for analytic
constraints but continue to optimise at deployment. Imitation of a CBF-safe expert can in principle
transfer formal guarantees, under conditions no large visuomotor policy currently satisfies.
Corrective data transfer when they cover the states the learner reaches, irrespective of who
produced them. And exposure to a shield during training does not, by itself, internalise anything.

What has not been asked is the amortization question: whether the corrections a control-theoretic
shield produces can be compiled into a VLA's weights so that the shield can be **removed** at
deployment, and what bounds that transfer.

The claim of novelty must be stated precisely, because two neighbouring claims would be false.
Learning safety from CBF-related supervision is not new — Cosner et al. treat it theoretically, and
learning from CBF demonstrations in simulation predates that work. Distilling a corrective signal
into a VLA is not new either; ROAD-VLA does exactly that. What is new is the conjunction: a
control-theoretic safety filter, distilled from a VLA's own shielded rollouts, evaluated with the
filter removed, in the setting where runtime shielding is currently the state of the art — together
with the boundary condition that a geometry-privileged planner, safe by construction and supplying
well-formed low-level actions, does not transfer under a matched comparison. ROAD-VLA's proximity
principle predicts that boundary; nothing in the literature tests it, because no work in this line
uses a foreign low-level controller as teacher.

Chapter 3 describes the shield, the teachers, and the distillation procedure. Chapter 4 reports what
transfers, what does not, and where the residual failures live.

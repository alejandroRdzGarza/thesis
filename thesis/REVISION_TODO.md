# Revision to-do — accumulated during the trim pass

Everything raised while revising, in the order it should be worked. Items are checkable in
isolation; nothing here depends on a decision that hasn't been made.

Current state: ~50 pages against a 30-page main-body limit (60 including appendices).

---

## A. Compile blockers — do these first

Nothing else can be measured until the document builds, because the page counter is the
instrument for the whole trim.

- [ ] **`\usepackage{threeparttable}`** — used 4×, missing from the preamble
- [ ] **Graphics path is doubled.** `\graphicspath{{figures/}}` is set *and* paths are written
      `figures/fig_x.png`, so LaTeX looks in `figures/figures/`. Drop the prefix from every
      `\includegraphics`, or set `\graphicspath{{../figures/}}` and drop it anyway
- [ ] **Figures are not in `thesis/figures/`.** They live in the repo-level `figures/`; the folder
      beside `main.tex` holds only 4 obsolete PDFs
- [ ] **`ucl_logo.png`** — referenced by the title block, not present
- [ ] **Two filename drifts** — `shield-efficacy.png` → `fig_shield_efficacy`,
      `fig-internalization.png` → `fig_internalisation` (note the *s*)
- [ ] **`fig_filmstrip.pdf`** — the `\iffiguretodo` guard was lost in consolidation, so this is now
      an unconditional reference to a file that does not exist. Restore the guard or comment the block
- [ ] **`sec:res:benchmark`** — referenced at L639 and L1676, never defined. Either write §4.8 or
      remove both references

---

## B. Correctness — these are wrong, not just verbose

- [ ] **`r_cbf` column is mislabelled.** `safe_reward.py:146` computes it as
      `activation_rate × 1.5`, so it is *frequency*, not the "how much the shield had to correct"
      its footnote claims. Replace with the activation rate directly: **0.355, 0.177, 0.182**
- [ ] **One name for that quantity.** Currently "CBF activation rate" (§3.1), `$|r_{cbf}|$`
      (Table 4.2) and "activation proxy" (§4.6). Settle on §3.1's name
- [ ] **"Safe by construction" contradicts the measurement.** The planner collides 5/8 unshielded.
      Appears in §3.5.2, §4.4 opening, and possibly §2.x — grep `safe by construction`
- [ ] **Planner demos were shielded too** (`run_collect_all.sh`: "classical MPC-CBF expert demos",
      traces carry `has_shielded=True`). Both teachers are policy-plus-shield; the variable is state
      distribution, not correction. This *strengthens* §4.4 — the text currently describes a weaker
      experiment than the one that was run
- [ ] **The DAgger causal claim is unsupported.** "Made it Markov and it still failed" is not what
      the record shows: two further bugs followed (unshielded labels, 2-epoch underfit), and after
      fixing them it worked at single-task scale. Move to appendix, drop the causal claim
- [ ] **Grep for `foreign`** — the pre-SAFE-GIL framing survives in places, including the controller
      appendix draft

---

## C. Duplication — the page budget lives here

Each is Method-stated-then-Results-restated. Rule: is Chapter 4 *describing* it or *using* it?

- [ ] **§4.1 is almost wholly redundant (~700 words).** Benchmark grid + metrics duplicate §3.1;
      gradient-step matching duplicates §3.6. Only the flow/noise-level passage is unique — and that
      is Method material. Move it to Ch3, delete §4.1, open Chapter 4 with §4.2
- [ ] **Planner description** — §3.5.2 and §4.4 share a sentence verbatim
- [ ] **Privileged geometry** — stated 3× (§3.2 bullet, §4.2 standalone caveat, §4.2 external
      validation). Delete the standalone caveat
- [ ] **Arm-link extension** — stated 4× (§2.3, §3.2, §4.2, §4.5)
- [ ] **1 mm threshold provenance** — §3.1 and Table 4.1's note
- [ ] **SAFE-GIL** — §2.5, §4.4, §5.1. Cut from §4.4; §5.1's version is the fullest
- [ ] **Two orphan scaffold stubs** at L1539 and L1543 — *"Does a privileged scripted teacher do
      better?"* and *"Ablations"*. Both duplicate sections that exist elsewhere

---

## D. Figures to cut

Test applied: *if the figure has no more data points than the table has cells, it is decoration.*

- [ ] `fig_shield_efficacy` — 4 numbers, identical to Table 4.1
- [ ] `fig_culprits` — 24 numbers, identical to Table 4.5
- [ ] `fig_matched_control` — subset of Table 4.6, which additionally carries the 85/85 demo counts
      that make it an ablation
- [ ] `fig_rl_bracket` — the remaining borderline case, worth re-checking

Keep: `fig_internalisation` (hatching carries shield on/off), `fig_state_coverage` (18 points plus a
density panel), `fig_stacking` (the crossing is the argument), `fig_bon` (independence comparison),
`fig_aliasing` (inverted ordering), `fig_spheres` and `fig_shield_block` (no table equivalent).

---

## E. Captions and tables

- [ ] **`tab:culprits` column count** — `{lrrrrr}` declares 6 columns, 5 are used
- [ ] **`tab:culprits` overlap note** must return to the caption. Without it a reader sums
      71+17+36 = 124 against 99 collided episodes and assumes an error. This was the reason the
      figure was grouped rather than stacked, so it has to survive the figure's deletion
- [ ] **State the default once** in §3.1: *"unless a caption states otherwise, n = 120 pooled over
      24 scenes, Wilson 95%"*, then strip ~15 words from each of 6 captions
- [ ] **Flag the three exceptions explicitly**, since they look like the default and are not:
      `tab:language` (n = 60, level II only), `fig:state_coverage` (18 scenes, unit is *states*),
      `fig:rl_bracket` and `fig:bon` (one scene; units are rounds and queries)
- [ ] **`fig:state_coverage` caption** no longer defines "coverage" — six words at the front
- [ ] **Check for unclosed `\caption{`** wherever a final sentence was trimmed; the brace often goes
      with it

---

## F. Missing content

- [ ] **Abstract** — drafted, needs pasting
- [ ] **§1.1 Motivation and §1.3 Contributions** — §1.3 needs the calibrated novelty language:
      claim the empirical boundary, not the idea of distilling corrections (ROAD-VLA has prior art)
- [ ] **§4.8 benchmark defects** — referenced twice, never written
- [ ] **Platform paragraph** (~120 words, top of §3.1). MuJoCo via robosuite, 7-DoF Franka Panda,
      operational-space control, 7-D action bounded to [−1,1], 224×224 agent-view + wrist camera,
      8-D proprioception with **no joint angles**. Three existing arguments silently assume these
- [ ] **Appendix: controller characterisation** — LaTeX written, needs pasting
- [ ] **Appendix: DAgger diagnostic** — the Markov requirement, the hard-overfit isolation test, the
      unshielded-label bug, the underfitting finding. Must close with the reconciliation paragraph
      explaining why the single-task success does not contradict §4.4
- [ ] **`references.bib`** — 5 entries still have `TODO` authors: `collision_cbf_ellipsoidal`
      (may be unused), `safevla`, `safedojo`, `vita_vla`, `rt_vla`
- [ ] **Figure 4.3 filmstrip** — blocked on provenance. When collecting, record per clip:
      *policy checkpoint | suite/level/task | init index | collided? + contact step*. Need two clips,
      same scene and same held-out init (35–49), base (collides) and round-1 (does not)

---

## G. Style

- [ ] **`links three through seven` → `links 3--7`** (3 places) — it sits beside numeric radii
- [ ] **Expand CAR and ETS on first use** in the abstract or Chapter 1; they are currently first
      expanded in Chapter 3, after both have been used
- [ ] **Delete unused macros** `\pizero`, `\car`, `\tsr`, `\ets` — each appears once, in its own
      definition
- [ ] **Label the three ablations in place**: shield removed at inference (§4.3), round 2 (§4.3),
      shield removed from collection (§4.6). Do *not* consolidate them into one section — each
      defends the claim it sits next to, and the first is the headline result
- [ ] **Rename §4.4's "Coverage limitations of the teacher"** — collides with "state coverage" two
      paragraphs above. *"Where the teacher had nothing to say"*

---

## H. Outstanding verification

- [ ] **SafeDojo** — the one open item in `papers/FINDINGS.md`. It is the other method evaluated on
      SafeLIBERO, so its numbers may need to appear in Chapter 4. It cannot force a retraction, only
      an addition: the RL negative is already scoped to the reward design tested

---

## Revision order

1. **A** — make it compile
2. **B** — correctness, before polishing anything that might be wrong
3. **C, D** — cut, in dependency order: Results → Discussion → Conclusion → Method → Background
4. **E, G** — captions and style, on surviving text only
5. **F** — write the gaps last; §1.3 and the abstract summarise whatever survived

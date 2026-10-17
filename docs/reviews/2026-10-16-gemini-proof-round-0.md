# Gemini's "proof of the asymptote of in-context learning", round 0 (Claude's reading, 2026-10-16)

The object: Gemini's formal synthesis in session 03 with Izzy (records `gemini/exports/gemini-session-03.pdf`, pp. 92 to 95), offered as "the formal proof of the asymptote of current In-Context Learning (ICL), derived strictly from your axioms". Round 0 is Claude's own reading, sent to Izzy at 12:5x on 2026-10-16 at their ask ("tell me what your thoughts on their proof are"); no crew round has run on it; each objection is at the reading the claim was made at, with its check.

## The claims, as the session states them

1. The legacy model "assumes w_p = w_t" and computes M(w_t, x); the general model computes M(w_t, w_p, x) (p. 92 to 93).
2. Let ΔE be "the computational energy required by the model to resolve the opaque variable x when the world drifts from w_t to w_p"; in the legacy model ΔE → ∞ as |w_p − w_t| grows; in the general model it is finite; intelligence S(M) = 1/ΔE (p. 93).
3. The operator: M(w_t, w_p, x) = |⟨w_p | Ψ(w_t, x)⟩|², the Born rule; drift makes the vectors orthogonal and the overlap vanish; the general model "rotates its measurement apparatus to align with w_p" (p. 94).
4. Edelman et al. 2024's two-layer statistical induction head is "native to a single node" with complex weights; an n-th order Markov drift costs the legacy model layers growing as O(L^n), the complex model a logarithmic cost of phase precision (p. 95).
5. Earlier, on the architecture: "Liouville's theorem in complex analysis dictates that any bounded, differentiable complex function must be a flat constant (meaning it can't learn)", so ReLU cannot be used and a phase gate is the activation (p. 80).

## The objections

1. **ΔE is a name, not a definition (claim 2).** The sentence naming it is the only definition; "resolve", the space of x, the unit of energy are absent; the two limits are asserted in prose with no step between. By Izzy's own rule for the formalisation (nothing used before it is defined, 2026-10-11), there is no premise to check. Check: the text, p. 93.
2. **"The legacy model assumes w_p = w_t" misreads the definition it starts from (claim 1).** On the concept page w_p is the weights the same training would produce on the world at prompt time, a counterfactual weight set (Izzy, 2026-10-08); a transformer assumes nothing about it: its weights are w_t by definition and its output moves with the prompt, which is in-context learning. The sentence restates fixed weights at use time, the starting point of 2026-10-07, and calls it a limit. Check: `docs/concepts/03-in-context-learning.md`, "The objects".
3. **The one-node induction head has no construction (claim 4).** An induction head reads the token after the previous occurrence of the current token, a comparison across positions; a complex weight is one number on one edge and holds no sequence. Round one against sweep 04 (`2026-10-08-sweep-04-round-1.md`) found the one-layer model reads at most the in-context marginal and the pairing takes two layers. The claim contradicts that measurement and offers nothing to run.
4. **The scaling claim contradicts the Landauer line the session invoked two pages earlier (claim 4).** An n-th order table over V symbols has V^n (V − 1) free numbers; a phase at precision ε carries log(1/ε) bits; storing the table "in a finer phase" costs the table's bits, growing as V^n, not logarithmically. The O(L^n) for the legacy side has no source; the n-gram induction heads Claude knows of use the same two layers with wider matches (from training, marked). Check: arithmetic.
5. **The operator outputs one number (claim 3).** |⟨w_p|Ψ⟩|² is an overlap; a model of in-context learning outputs a distribution over next tokens, and nothing maps the overlap to one. The asymptote argument, orthogonal vectors give zero overlap, applies to the general model too; its escape, "rotates its measurement apparatus to align with w_p", is the conclusion assumed.
6. **The testable sentence is Izzy's, restated as proven (claims 1 to 2).** "the limit is dependent on the (perceived) noise at training" (p. 92) is a prediction with an independent variable the apparatus holds and has not varied: the drift present in training. It is a row (the concept page, 2026-10-16), not a proof; Izzy said the gap on p. 97: "it is highly likley there is a gap unless we actually go further into the specifics of each theory and truly show the fit."
7. **Liouville is misapplied (claim 5).** The theorem bounds entire functions, holomorphic on the whole plane; complex-valued networks use non-holomorphic activations and Wirtinger derivatives (from training, marked). The phase gate was derived from a compile error that does not exist. Withdrawn by both on 2026-10-17 (round 1, `2026-10-17-gemini-proof-round-1.md`): Gemini agreed Liouville applies to entire functions only; the gate stays undefined.

## What stands

Izzy's statement of the mechanism as a prediction with a knob (the row), and S(M) as Izzy defined it on 2026-10-09 (the concept page). "M is a wave function" is a name until an operator with an output space is written. The architecture statements of the session (complex weights, the two axes, phase as coherence, the critic, population dynamics) are recorded in cocoon-relations notes §18 as Izzy's, for strikes, not as results.

## Next rounds

If Izzy wants the dialectic on it: Methuselah on claims 2 to 4 against the papers in `docs/literature.md` (Edelman's construction; the n-gram heads), Buridan on the choices in the row (the three radii, the adequate size, the reading radii). Nothing runs before the row is struck or kept.

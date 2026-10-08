# The count, dialectical round one (Claude against Claude, 2026-10-08)

The rule: no number reaches Izzy until two full rounds and two further replications. This is the first round, taken against the count on the concept page and the code that realises it, before any number exists. Each item: the objection, what it does to the count, and what closes it.

1. **Shares: frequencies or weights.** *Closed on 2026-10-08 by Izzy's 2 + n: the two readings are the two operations of a semiring, ⊕ across sisters and ⊗ along a merge, and a daughter's weight is a sister with a null one level down. Not a choice; see the page.* The code reads a share as the frequency of an alternative; the definitions also allow a weight on a daughter. Under the second reading no artifact draws anything and k artifacts pin no ratio by frequency: the whole "carried by k" paragraph is void. Closes only by Izzy's answer (on the page as a question). Until then the count is conditional on the first reading.

2. **Cuts per node per artifact assumes one visit.** The count gives each node two binary cuts per artifact. In a grammar a node may be visited several times in one derivation, or not at all. The carried count is per visit, and the expected visits per draw is the π of the count, so the correction is to read "per artifact" as "per visit" and let π carry the rest. Closes by wording; the √(kπ) line already has it right.

3. **The drift cap is per node, not per table.** λ is the total share movement per step over the whole table; a node moves by about λ/n. The cap 1/(sλ) overstates the drift at a node by the node count. Corrected cap: ρ_eff ≤ n/(sλ) at a node, or, stated per node, 1/(sλ_node). Closes by replacing λ with λ_node in the statement. The code's `step` already spreads the bound per node.

4. **Phases are idle in the apparatus as built.** `draw` samples sisters classically by squared moduli; no interference enters the artifacts, so no run on this code can pin a phase, and the φ terms in the count describe a quantity the generator does not produce. Declared gap: the constructed source needs a draw whose string probabilities are squared sums of amplitudes over derivations before any phase is measurable. Until then every φ_eff is φ, and the count's phase lines are dormant, not wrong. **Closed in code the same day:** `distribution` and `draw_interfering` in `cocoonml/schema.py` give each string the squared modulus of the sum of its derivations' amplitudes; two readings of one string add at phase zero and cancel at half a turn (tests). One consequence found by the tests: a silent sister in either order is already two readings of one string, so order ambiguity interferes too, and the classical draw is recovered only when orders are fixed or strings differ.

5. **An order cut is observable only when the sisters' forms differ.** If both symbols of a merge pronounce the same forms, the two orders give one string, and the artifact carries no order cut at that node. The carried count must exclude such nodes; the demand keeps them. This is a second kind of ambiguity, order ambiguity, beside the tree's, and it belongs in t.

6. **The shape is assumed known.** `estimate` takes the table's shape and the count takes L and the node structure as given; a reader of artifacts does not have them. The number of shapes consistent with the artifacts is a count of its own, Catalan-like in the leaf count, and it multiplies the demand. The identifiability statement as written is for a known shape; the general one needs the shape term, and it is the largest term for short windows.

7. **Gap one, a sketch toward closing it.** Claim: the predictive distribution at position i depends only on nodes visited by derivations consistent with the prefix. Sketch: the probability of the next form is a sum over derivations of the whole context that agree with the prefix, each a product of the shares and orders along its visits; a node visited by no such derivation contributes to no term; so unpinned nodes off every consistent derivation change nothing at the query. Premises: the first reading of shares; classical draws (phases dormant, item 4). What it leaves open: nodes visited by some consistent derivations and not others are partially relevant, and the path-restricted residual has to be a sum over consistent derivations weighted by their shares, not a single path. The equivalence of identification and prediction is then: prediction error is bounded by the residual restricted to the consistent-derivation set, which is smaller than the tree and larger than one path. A theorem-shaped statement, unproved.

8. **The smoke run is not evidence of anything.** It exists to show the chain executes; its numbers are at toy size, one seed, one table, and are excluded by the rule from any sentence to Izzy. Round two and the two replications are owed before any of them is said.

**Added the same day, from the harness:** next-form prediction reads only the orthographic side; the protocol side needs a target that depends on the cuts. `pack_meaning` puts each artifact's extension at its separator as a token, and `meaning_loss_at_separators` reads the recovery of meaning against k. Closed in code before round two could raise it.

Verdict after round one: the count stands conditional on item 1, with item 3 corrected in the statement, items 4 and 6 declared as gaps, item 5 folded into the ambiguity term, and item 7 the first theorem to attempt. Nothing in it is a number yet.

## T1, the first theorem of the toy (Claude, 2026-10-08; proved by the definition, held by a test)

**Statement.** In the constructed source, the predictive distribution of the next form given a prefix depends only on the nodes visited by some derivation consistent with the prefix. A node visited by none has no effect, whatever its share, order or phase.

**Proof.** The predictive distribution is a ratio of two sums over derivations consistent with the prefix, each term a product over the derivation's visits of that node's amplitude. A node visited by no consistent derivation appears in no term of either sum. Hence both sums, and their ratio, are independent of it. Interference does not change this: the sums are over the same consistent set. ∎

**What it gives the count.** Gap one, identification against prediction, has its first half: the residual that bounds prediction is at most the residual over the consistent-derivation set, never the whole tree. What it does not give: how much a node visited by some consistent derivations and not others matters, which is the weighted version and is the next theorem.

**Test.** `TestTheoremOne` in `tests/test_schema.py`: an unvisited node's share, order and phase are moved and the prediction is unchanged to twelve places; a visited node's share is moved and the prediction moves.

## T2, the weighted T1 (Claude, 2026-10-08; a sketch, held by a test on random tables)

**Statement.** For a node visited by some consistent derivations, let m be the share of the consistent squared-amplitude mass that passes through it, s its share, δ a move of that share. The total variation of the predictive distribution is at most m · |δ| / min(s, 1 − s), to first order in δ. T1 is the case m = 0.

**Sketch.** A derivation through the node carries a factor √s or √(1 − s) per visit; moving s by δ changes that factor by a relative amount of about |δ| / (2 min(s, 1 − s)) per visit. Derivations not through the node are unchanged. The predictive distribution is a ratio of sums of squared sums of amplitudes; a relative change ε on a fraction m of the mass moves the ratio by at most about 2mε in total variation. Substituting gives the bound. Second-order terms and multiple visits are not controlled; the test uses a small δ, keeps s away from the clip, and allows a tolerance.

**What it gives the count.** Gap one closes in this form: the residual that bounds prediction is the residual over nodes weighted by their consistent mass, which lies between one path and the tree. A node the context never visits costs nothing; a node every consistent derivation visits costs its full unpinned resolution; the rest in proportion. The path-restricted residual of the count is replaced by the mass-weighted one.

**Test.** `TestTheoremTwo`: forty random tables, every node away from the clip, δ = 1/64: the measured total variation stays under the bound (tolerance 0.02 for the second order), and nodes of mass zero move nothing.

# The fixed-index bound: the skeleton of the identifiability statement

Claude, 2026-10-09, owed since the night of 2026-10-08 ("the proofs get the same discipline as the code: statement, premises, derivation, disproof"). The general form, the index moving inside the window, is Izzy's to open; this document fixes the index and states what is then arithmetic from the definitions on the concept page (`docs/concepts/03-in-context-learning.md`, "The count" and "4, the fixed-index bound, first form"). Every step carries its status: **theorem** (follows from the definitions, with a proof or a test in `tests/`), **derivation** (follows with a stated approximation), **assumption** (a premise that could fail, with what would show it wrong), **open** (Izzy's). Counts throughout; a logarithm appears only as bookkeeping and is not written.

## 0. Setting

One table T of a fixed shape at one world. The shape: n nodes (nonterminals), each with two sisters, each sister a merge of two symbols with an order ratio; leaves are forms, the empty form among them. The values at the grid: every share and every order ratio at one of ρ + 1 values (grains of 1/ρ; the code's `resolution`), every phase at one of φ values (φ = 1 at a node on no cycle: a phase attaches only to a loop, and the shape has a loop only where a string admits more than one derivation), every spoken leaf a weight vector over c cuts at grains of 1/ρ summing to one. The index is fixed: every artifact in the window is a draw from T.

A window W of k artifacts: k strings, each a draw of the whole tree pronounced (`schema.draw`). Nothing else is in the window: no tree, no extension, no state (the corpus design). The reader has the shape and the strings.

## 1. Definitions

- **D1 (visit count).** For a node v, n_v(W) is the number of observations of v in the window: each string counts once, its derivations at equal weight 1/t where t is the number of derivations the shape admits for it (`schema.estimate_shares`). Likewise n_{v,i} for sister i of v (the order ratio of that expansion is observed only when that sister is drawn), and a_v(W) for the observations through strings with more than one derivation of nonzero amplitude (`schema.n_eff`).
- **D2 (path mass).** π_v is the probability under T that a draw passes through v (the sum over derivations through v of their classical probabilities). E[n_v] = k · π_v for one-derivation strings; with ambiguity the expectation is the same sum weighted by 1/t.
- **D3 (demand).** D(T) is the number of tables of the shape distinguishable at the grid: per node ρ (the share) · ρ² (an order per expansion) · φ (the phase), times the leaf term L (section 6). D = (ρ³ φ)^n · L. This counts the tables, not the tables a window could tell apart; it is the ceiling of what identification could pin.
- **D4 (resolved levels).** For a node v after W, the share is resolved to ρ_eff(v) = min(ρ, √n_v) distinguishable levels; each order to min(ρ, √n_{v,i}); the phase to φ_eff(v) = min(φ, √a_v). Section 2 says why √.
- **D5 (the consistent set and the residual).** 𝒯(W) is the set of tables of the shape at the grid consistent with W: every node's share within the band of width 1/ρ_eff(v) around the observed frequency, and likewise orders and phases. The residual is the count R(W) = |𝒯(W)|, which under D4 is the product over nodes of (ρ/ρ_eff)(v) · Π_i (ρ/min(ρ, √n_{v,i})) · (φ/φ_eff)(v), times the unpinned part of the leaf term. R(W) = 1 is identification to the grid.
- **D6 (consistent mass).** For a query prefix x and a node v, m_v(x) is the share of the squared-amplitude mass of the derivations consistent with x that pass through v (`schema.consistent_mass`).

## 2. Lemma 1: the sampling floor (derivation)

A frequency estimated from n observations of a share s has standard error √(s(1 − s)/n) ≤ 1/(2√n). Two shares closer than about 1/√n are not told apart by n observations; so n observations resolve a share to about √n levels on [0, 1], and no finer than the grid: ρ_eff = min(ρ, √n). **Status: derivation** (the binomial standard error; the constant absorbed into "about"; the expected absolute error at the true share and count is the code's sampling floor in `schema.identification_excess`, √(2/π) · √(s(1 − s)/n)). What would show it wrong: a reader that pins a share finer than 1/√n from n draws, which would mean the draws were not independent given the table, i.e. the index was not fixed.

For phases the observations are the ambiguous strings only (a phase enters an amplitude only where two derivations of one string add), so the count is a_v, and the same argument gives φ_eff = min(φ, √a_v). **Status: derivation**, resting on the carrier being complex with interference confined to one string's derivations (the page, "The weight rule as a semiring"; `schema.predictive`).

## 3. Lemma 2: T1, the unvisited node (theorem)

A node visited by no derivation consistent with the prefix x has no effect on the predictive distribution P_T(· | x), whatever its share, order or phase. **Proof.** P_T(· | x) is the conditional of `distribution` on strings extending x; every such string's probability is the squared modulus of a sum of amplitudes over its derivations; a node not on any of those derivations enters none of the amplitudes; the normaliser over all strings cancels in the conditional. **Status: theorem**; `tests/test_schema.py` checks it numerically.

## 4. Lemma 3: T2, the sensitivity of a prediction to one node (derivation, first order)

Moving the share of a node v by δ (orders alike, phases with their own factor) moves P_T(· | x) by at most m_v(x) · |δ| / min(s_v, 1 − s_v) in total variation, to first order in δ. **Sketch.** The consistent mass through v is m_v; a share move rescales the amplitudes through v by √((s + δ)/s) on one sister and √((1 − s − δ)/(1 − s)) on the other, a relative change of at most |δ| / (2 min(s, 1 − s)) in modulus, hence at most |δ| / min(s, 1 − s) in squared modulus; the mass not through v is unchanged; renormalisation moves the conditional by no more than the moved mass. T1 is the case m_v = 0. **Status: derivation**, first order; `schema.sensitivity` computes the exact total variation for a finite move, and the test on random tables is the check of the constant. What would show it wrong: a random table where the exact sensitivity exceeds the first-order bound by more than the second-order term for a one-grain move. The exact statement needed for a grid move of one grain (δ = 1/ρ) is the finite-difference version; its constant is read from the code, not derived here (**open**, mine).

## 5. The bound (proposition)

**Statement.** Fix T, W with k artifacts, a query prefix x. For any two tables T₁, T₂ in 𝒯(W),

TV(P_{T₁}(· | x), P_{T₂}(· | x)) ≤ B(x, W) := Σ_{v} m_v(x) · [ (1/ρ_eff(v) − 1/ρ)₊ / min(s_v, 1 − s_v) + Σ_i (1/min(ρ, √n_{v,i}) − 1/ρ)₊ / min(o_{v,i}, 1 − o_{v,i}) + (1/φ_eff(v) − 1/φ)₊ · c_φ ],

to first order in the band widths; (·)₊ is the positive part (a node pinned to the grid contributes nothing); c_φ the phase constant of Lemma 3's phase case. So a reader that holds the window and nothing else cannot be nearer the truth at x than B(x, W) is wide, in the worst case over the consistent set, and the best consistent reader is within B(x, W) of the truth. k enters only through the visit counts: E[n_v] = kπ_v. That is the bound at k.

**Proof outline.** Move from T₁ to T₂ one node at a time, each move within the band of D5; the triangle inequality over the n moves; Lemma 3 at each move with that node's band as δ; nodes outside every consistent derivation contribute zero by Lemma 2; sum. Each step is a derivation or a theorem as marked above; the sum is arithmetic. **Status: derivation**, inheriting Lemma 3's first-order caveat and one assumption:

- **A1 (independence of the bands).** The product in D5 and the sum in B treat the nodes' bands as independent. They are not when one string's derivations visit several nodes (an observation of a string is one observation of every node on its derivations). The sum stands as an upper bound (the triangle inequality holds regardless); the count R(W) over-counts the consistent set when bands are correlated. What would show A1 costly: a window where the exact consistent set (enumerable at toy size from `schema.estimate_shares`) is smaller than the product by more than the ambiguity factor.

**Corollary (identifiability).** R(W) = 1, and B(x, W) = 0 for every x, iff n_v ≥ ρ² at every node, n_{v,i} ≥ ρ² at every sister, a_v ≥ φ² at every node on a cycle; in expectation, kπ_v ≥ ρ² at every node, the page's condition. Otherwise the residual is the product in D5 and the prediction bound is B. **Status: theorem given D4** (D4 is the derivation of section 2).

**The bound on learning against identification** (the page's gap 1): B sums over the nodes on x's consistent derivations with their mass m_v(x); R(W) multiplies over every node. A query reaches only its own path, so the learning bound is the path-weighted sum and the identification residual the full product; they agree when every unpinned node lies on the query's path with mass one. The difference is the size of the assumption that identification is prediction, now a number per query.

## 6. The two readouts and the leaf term

- **Form.** The next-form reading (`corpus.form_losses_by_rank`) is P_T(· | x) over forms. Leaf weights do not enter the string distribution at all (`schema.distribution` sums amplitudes over derivations; a leaf's weight vector rides along as the extension and never multiplies an amplitude), so for the form readout the leaf term is absent from both D and B: the bound is over nodes only. **Status: theorem** (read off `_all_derivations`).
- **Meaning.** The extension reading (the pairs design, `harness.pack_pairs`; latent in the corpus design) depends on the leaf weights: the demand has the leaf term L and the residual has its unpinned part. A form-only window pins no leaf weight (the forms carry no cut), so for meaning the leaf term stays wholly in the residual however large k: that is I_Δ in the page's terms, and it is where the two predictions separate. Pins come only through an extension channel (a pair in the input), one observation of the whole vector per occurrence; at grains of 1/ρ a vector is read exactly from one occurrence, so its pinned count is 1 or the full L_leaf, not a √.
- **The leaf term, open (Izzy's).** In grains: a spoken leaf's weight vectors over c cuts at grains 1/ρ summing to one number C(ρ + c − 1, c − 1); with c = 2 that is ρ + 1. In ratios: the distinguishable ratios between the cuts' weights. With the sum-to-one gauge the grain vectors and the ratio tuples are in bijection, so the two counts coincide **at a uniform grid**. They differ if "distinguishable ratio" is relative (the mantissa: a weight near zero resolved finer than one near a half), which is what "counts are relative" says of the model's grain. So the question is whether the leaf's resolution is the schema's uniform grid or the model's relative one; the code takes the grid (`schema.generate(graded=True)`). Open, Izzy's; the bound above holds with either L substituted.

## 7. Where the index enters (the general form, Izzy's to open)

Everything above has the index fixed. With the index moving s steps inside the window at share movement λ_v per step at v, no frequency pins a share finer than the target has moved: ρ_eff(v) = min(ρ, √n_v, 1/(sλ_v)), the page's term, and B gains a floor that no k removes. s is the model's own reading of the window, not a parameter of ours. The general form is the statement of B with that ρ_eff and with the two samples of the world (w_t, w_p) in place of one table: S = S(M(w_t, w_p), x), Izzy's law; this document is its fixed-index case, M(w, w).

## 8. What the apparatus reads of it

- `schema.identification_excess`: the error beyond Lemma 1's floor; zero excess is a reader at the floor.
- `schema.n_eff`: the pinned share of D, Σ grades / 2n, the ratio R(W)'s complement read as a fraction.
- `corpus.form_losses_by_rank` at rank k: the reading of P_T(· | x) after k artifacts; its excess over `schema.entropy_floor_form` against B's prediction of how fast the excess can fall with k (through n_v = kπ_v: as 1/√k at a node, divided by depth through π_v).
- `schema.sensitivity` and `schema.consistent_mass`: Lemma 3's two sides, exact on a table.

## 9. Disproofs

- Lemma 1 fails if a reader pins a share finer than 1/√n with the index fixed.
- Lemma 3's constant fails if `sensitivity` exceeds the first-order bound on random tables beyond second order.
- A1 is costly if the enumerated consistent set is smaller than the product beyond the ambiguity factor.
- The bound as a whole is wrong if a model holding only the window predicts x within less than B(x, W) of the truth, averaged over the consistent set, at a size past the derived one (the run).

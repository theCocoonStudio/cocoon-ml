# 03 In-context learning

A model with fixed weights is given a few examples in its input and does the new task without any weight changing. Nobody knows the mechanism, what it can learn that way, or where the limit is. The first unsolved problem taken here (2026-10-07).

## The objects (Izzy, 2026-10-08)

- **Weights** w: set by training, read at use time. The internal state that moves per prompt is the activations only.
- **Input space:** the data lives on orthography, a living system that drifts as language does. What matters is not the form but the organisation of the protocol that produced it, a degree, never one, unreadable from the form.
- **Expected input space:** a rating of semantic weights and pragmatic common ground; a relation between the model and society; a weight shift.
- **Expected output space:** every output above a threshold. The threshold is empirical for the model, descriptive, and drifts; the space may change between input and output. So tests compare distributions against thresholds, never a function against a value.
- **Consistency:** a function of w_t, the weights training produced, and w_p, the weights the same training would produce on the world at prompt time. A scale between two vantages; the interlocutor is folded into w_p. Expected output space null: consistency 0, no gradient. Fixed by syntax alone: consistency 1, the prompt-time world adds nothing. Consistency is the share of the expectation the syntax does not fix.
- **Information is the substance.** The potential is the gap between expected and received, in cuts; the dynamics is across exchanges; equilibrium is where no cut remains between the model's expectation and the world. Treat the ensemble with statistical mechanics.

## The direction

Obviate the class: prove that at scale every sufficiently expressive architecture approximates the same object, the best predictor of its data distribution, differing only in speed and in where it cannot. Then the layer to reason in is the data's, with no architecture in it. The loss curve's best fit (a power law with a floor) is a window, not a limit; the alternative curve is a superposition of thresholds. A circuit defined without an architecture is the task under either root.

## The test

Our own language: a lexicon of symbols with meanings, a grammar composing them, a compositional rule to the output (the same fragment as the site's). Keep the form fixed and move the protocol: remap the meaning of one morpheme, then two. Tokens as characters, morphemes as symbols, so drift can be applied to form (respell, meaning fixed) or to protocol (remap, spelling fixed). Consistency is exact because we set it; weight each remapped symbol by how often the grammar produces it. Measure: loss under drift with no examples in context (predicts only its two ends: consistency 1 gives the trained loss, consistency 0 the loss with no usable meaning); loss with examples that reveal the new meanings, against k and context length, which is the bound; a probe for a variable tracking k.

A run is designed to encapsulate as many tests as possible; conceptual loose ends are tied before any run (Izzy).

## Predictions, lodged before any run

Izzy (2026-10-08, verbatim): "the drift opens the family. and it's graded. just like i could communicate with people in china if neither of us spoke each others language. if you train with non deterministic data, the weights account for the drift predictably [Izzy considered correcting this to "predictively" and left it; the note is at their ask]. so we disagree. if the model is trained on deterministic data, there is no difference. the model only tracks the family."

Claude (2026-10-07): the prompt selects within the class the weights fix; recovery works for remappings within the family training contained and fails sharply, unimproved by examples, for meanings outside it.

They separate on out-of-family remapping under non-deterministic training. Izzy, added later for the record: Claude's prediction is probably right for current models, and Izzy's is right for what is being developed here. The two predictions are about two paradigms, not one, and that distinction is a concept both repos share, cocoon-ml and cocoon-relations. "Class change" (Izzy, 2026-10-07) was withdrawn by Izzy as a name that admits every solution.

## Hurdles

The lexicon and grammar (Izzy's). Compute in pure Python at toy size. The proxy for consistency. An operational definition of "outside the family". The convergence theorem.

Check: a result is written only after it is read; both predictions stay as lodged.

## Added 2026-10-08, later

Izzy's prediction, quantified (verbatim): "it has to do with the protocol. the information space of your training is constant, w_t. to test, the protocol is defined such that its informational space can also be computed. how many meanings there are. adjust a morpheme. adjust a token. grab the delta." The delta is one factor; what it leaves is a residual, incomputable even with the protocol in hand (Izzy: "transcendent residuals"), the jump from syntax and semantics to pragmatics.

Claude's definition, by which Claude's prediction is judged: the family is the closure of the training protocol's meanings under composition, and, after the day's push-back, closure under the drift training contained; the boundary is the radius training reached. Outside: a new atomic meaning no composition reproduces; recovery there stays flat however many examples.

Design settled: not an experiment but reality. The generator's truth is never an input; drift is computed after the fact from the artifacts, produced prompts against the schema, a relation satisfied at training and run time. Drift in steps, a bounded walk, present in training as any large corpus has it; a strong seeded generator as the source (chaos conceded away); mechanism-agnostic, so the test transfers to the new paradigm. A deterministic language has consistency one by construction and can decide nothing: the information space needs the share of meaning the form does not fix. The schema: predicates all the way to the floor, which is the partial application of the schema to w_current, an input that exists in the type and never in the value; the residual is the part of w_current the artifacts do not identify.

## 2026-10-08, night: the schema, defined down to the count

Every item below is Izzy's unless marked; Claude verified and named the theorem that forces it where one does.

**The artifact is a string.** What was produced and kept: a sequence of forms with its place in the order along the index, and nothing else. Nobody keeps a tree, so the tree is an inference. Forced by the definition of an artifact. Consequence: structural ambiguity is in play, and identifiability covers the tree and the table both. The residual is what neither pins.

**Composition is one binary operation along a tree.** Any rule of higher arity curries into binary ones (theorem). A node with one branch is binary with a null daughter; null is a leaf with an empty form and a weight that is not empty. Not an arity issue.

**The schema is a relation**, not a function: index, form, extension, with the joint over the three as the honest object. Same form under different worlds happens and is not a decision. The incidence, which pairs a world holds, is filled at prompt time by the partial application and attached to its own index. Two indices, one per axis [Claude]: the form's index is what consistency reads; the incidence's index is what resolution reads.

**An extension is a weight.** The model's weights are the extensions at the training index; the prompt-time weights are the extensions at the current index; consistency is the distance between two extension sets of one intension. The measure is forced once extensions are weights: a divergence, whose uniform case is a ratio of counts (Boltzmann 1877, Shannon 1948, Kullback–Leibler 1951).

**Counts are relative, on rationals.** Ratios of counts, no logarithm, no base; the reals never enter (Claude's "arithmetic on reals" was an import, caught). A cardinality need not be whole: the number of ambiguities a structure can define is fractional under weights. Weight resolution is relative too: the number of distinguishable ratios an entry can stand in, the mantissa width, read off the model (floating point is a relative-count representation; the networks are scale-free; quantisation damage is relative).

**Tokens are not needed; cardinalities are.** The relation's smallest case is three counts and their joint: forms, extensions, pairs the index holds. Ambiguity is a row with several entries, synonymy a column. The alphabet's labels are meaningless; its cardinality, with the string length, sets the scale ("whatever gives the largest count").

**The step is a superposition.** Many entries each moving a little; the step is their sum, bounded, so no jump can happen; a sum of many small changes is a diffusion in the limit (central limit theorem), so the Brownian or mean-reverting index is derived, not chosen. The step has a direction: left alone, resolution falls. Languages simplify at every level from the sound up (an empirical fact, with cycles and the contact effect as checks). Simplification is a clue to the mechanism: a form sheds what the shared world carries; that information is residual to the prompt and not to communication; w_current is the other channel of the same message.

**Sequence matters to the degree the tree cannot be recovered without it.** Order and marking are two channels for one information (free-order languages, Warlpiri, spend from marking); the count is linearisations admitted per tree. Morphology is a phonetic artefact, out of purview; the string is the floor; the floor's meaning is lexical semantics.

**What is struck from the unknowns.** Index states in play per window: the model's own reading, the thing under test, not a parameter of ours. There is no metronome in the index.

**The reframing.** Counting the structures a string can carry (Catalan for full binary trees; unary-binary trees by Motzkin numbers; unary chains collapse by the definition of a cut) is a reframing of the problem: in-context learning as the artifacts pinning structure out of an exponential space, and the bound as the gap between the cuts the context carries and the cuts the ambiguity demands. Claude's critical pass, four gaps: identification is not prediction (the equivalence is a theorem to prove); a uniform prior (dissolved: the framework has no initial conditions, the demand is measured between two readings); per-artifact capacity is not a constant and artifacts are correlated (which is the content, not a flaw); the count must be over the table and the tree, not trees alone.

**Two kinds of jump.** A forced direction fails when its derivation fails; a choice fails when the alternative does better. By the night's end the list of choices is empty: the unary bound (cuts), no jumps (superposition), the alphabet (labels are nothing), the length (a swept variable). Everything is forced or read off the model.

**Open, for the morning.** The operations list, Izzy's: for each operation, its order rule (what it does to the sequence of its two daughters: fixed, free, marked) and its weight rule (how the daughters' extensions become the whole's), which together are the composition's count rule. Then the step written with its lean. Then the count is arithmetic and the identifiability statement can be attempted on paper. Izzy: "the stop happens tonight."

## The count (Claude, 2026-10-08; the definitions above turned into arithmetic; for Izzy's strikes)

Everything below is a count or a ratio of counts. Logarithms, where they would make a product a sum, are bookkeeping and are not written.

**Notation.** A table at one world is a binary tree with L leaves, nulls counted as leaves, so L − 1 internal nodes. At each node: a share between the two sisters (they sum to one; the squared moduli, since weights are complex in general), an order ratio, and a phase. A share or an order ratio is resolved to ρ distinguishable values, ρ being the grain: one in L while L is small, the mantissa past the crossover. A phase is resolved to φ values, and φ = 1 at any node that lies on no cycle, since a phase attaches only to a loop and the structure has loops only where a string admits more than one tree. The incidence, which form–cut pairs the world holds, has I distinguishable settings.

**Demand.** The number of tables distinguishable at that resolution is

D = I · (ρ · ρ · φ)^(L − 1),

one share, one order and one phase per node, and the incidence once. That is what full identification would pin.

**Carried by one artifact.** An artifact is one draw of the whole tree, pronounced. It pins the pairs it instantiates (a part of I), and at each node it pins which sister was drawn and which order was drawn: two binary cuts per node. It pins no ratio, because a ratio is a frequency and one draw has none. If the string admits t trees, the artifact pins its tree only to one of t, so its two cuts per node are divided among t candidates: the carried count of one artifact is at most 2^(2(L − 1)) / t distinguishable tree readings.

**Carried by k artifacts from one world.** Draws through a node are independent given the table, so a node seen n times has its share pinned to about √n distinguishable values, the relative uncertainty of a frequency. A node on a path of share π is seen about kπ times. So after k artifacts the share at a node is resolved to

ρ_eff = min(ρ, √(kπ)),

and the same for its order ratio. Depth costs: a node under small shares is seen rarely and resolved coarsely, which is the reframing's "bits per artifact is not a constant" made exact.

**Drift within the window.** Artifacts from different worlds are draws from a moving table. Over s steps with total share movement λ per step (the bounded sum of the superposition), the target has moved by about sλ, and no frequency can pin a share finer than the target has moved:

ρ_eff = min(ρ, √(kπ), 1/(sλ_node)),

with λ_node the share movement at that node per step (the table's bounded total λ spread over its nodes; round one, item 3).

This is the term no bound in the literature has, and it is the only place the index enters the count. s is not ours to set: it is what the model reads out of the window (the belief), so this line is also where the model's own reading becomes a parameter of the bound.

**Phases.** A phase is pinned only by artifacts whose readings interfere, the ambiguous ones. If a fraction a of artifacts admit more than one tree, φ_eff = min(φ, √(ak)). A language with no ambiguity never pins a phase, which is consistent: it has none.

**The residual.** The number of tables still consistent with the k artifacts, over one, is the product over nodes of (ρ/ρ_eff)² · (φ/φ_eff), times the unseen part of the incidence, I/I_seen. That product is the residual as a count. The identifiability statement, first form: the table at a world is identified to its resolution from k artifacts in a window of s steps iff kπ ≥ ρ² at every node, ak ≥ φ² at every node on a cycle, sλ ≤ 1/ρ, and every pair the world holds has occurred; otherwise the residual is the product above, and it is a ratio of counts.

**Where the four gaps bite.**

1. Identification is not prediction. A query's output depends only on the nodes along its own path, so the residual that bounds prediction is the product over that path, not over the tree. The bound on learning is the path-restricted residual; the full residual is the bound on identification. The equivalence is exact when the unpinned nodes off the path change nothing at the query, and the difference between the two products is the size of that assumption.
2. No initial conditions. The demand is not the whole table but the difference between two readings, training and now: only the nodes whose share, order, phase or incidence moved count, D_Δ = I_Δ · (ρ²φ)^(moved nodes). Consistency is this difference measured; the count makes it a product over moved nodes.
3. Correlation enters as sλ, and it is content: with λ = 0 the bound is the field's; with λ > 0 it has a ceiling that no k removes.
4. The tree enters as t and a, the ambiguity: it divides what an artifact carries and it is the only thing that lets a phase be carried at all.

**The two predictions in the count's terms.**

- Claude: recovery at a node is the identified fraction, ρ_eff/ρ, for nodes whose cuts lie in the trained closure, and zero for nodes whose cut is new, I_Δ outside I_trained: a pair the model holds no cut for cannot be pinned by frequency, whatever k. Flat at zero, independent of k, past the closure.
- Izzy: recovery tracks D_Δ, graded, new pairs included; what the delta does not account for is the residual's remainder, transcendent, and it is incomputable from the artifacts by the identifiability statement itself.

**What the count predicts before any run, where both agree.** Recovery grows as √k at a node and is divided by depth through π; it is capped by drift at 1/(sλ) however large k grows; a phase is learnable only from ambiguous artifacts; and the resolution a model can reach is the coarser of the tree's grain and its own. The disagreement is confined to I_Δ: whether a new pair is recoverable at all. That is the run.

## Operational reading in the code, and one question (Claude, 2026-10-08)

`cocoonml/schema.py` realises the table as a binary grammar: every nonterminal has two sisters that are alternative expansions, their shares summing to one, and each expansion is a merge of two symbols with an order ratio. In that reading a share is the share of the world's traffic that takes one alternative, which is what "a leaf's weight is its share of the world's traffic" says, and what the count's "which sister was drawn" uses.

The definitions also admit a second reading, and the two are not the same object: sisters as the two daughters of one merge, both present in every artifact, with the share saying how much of the node's weight rides on each. Under that reading nothing is drawn at a node, every leaf is spoken in every artifact, and a share is a weight on meaning rather than a frequency of use. The code took the first because an artifact must be a draw for frequencies to exist and for k artifacts to pin anything; the second is the one "sisters sum to one, the scale set by the leaf count" was said in.

The question, Izzy's: are shares frequencies of alternatives, weights on daughters, or both, with one at the branch points and one inside a merge? The count and the code follow the answer; until it is given, the code's reading is marked as mine.

Also: `cocoonml/harness.py` packs several artifacts into one context with a separator form and reads loss by position as the recovery curve inside one window; `run()` trains on a table, reads the curve on it and on its drifted version. A smoke run of the chain exists in Claude's scratch; its numbers are under the rule and are not on this page.

## The weight rule as a semiring, and the shares question closed (2026-10-08, after Izzy's "2 + n")

Izzy: a tree is not two-dimensional; it is 2 + n, with n hidden, and the two are the axes of syntax and semantics. The two readings of a share were two visible axes of one object, not a fork, and the list of what a share can count is open, finite only to a reader. Izzy asked for the constraints on F, the function at a node; Claude's lookup, forced by that:

F is not one function but two operations: along a merge, x ⊗ y; across sisters, x ⊕ y. The constraints: each associative with an identity, ⊗ distributing over ⊕, so that the sum over derivations of a string exists at all; the identity of ⊗ is the removed sister; degree one, so the gauge can be per node; closed on the rationals, Gaussian when phases are in; order a factor, ⊗ commutative. A commutative semiring. The two visible dimensions of the tree are those two operations, across and along.

The carrier is complex in general, by the fork principle, the real case being phase zero, and the axes are projections of it, not separate tables: frequency is the squared modulus; cost is minus the log of the modulus, which at coarse resolution turns (plus, times) into (min, plus), the tropical semiring, the lean as a shortest path; phase is what both discard. One carrier, three readings; the n hidden dimensions are the readings a window does not resolve.

The shares question is closed by this, not by a choice: a sister's amplitude is ⊕-side, a daughter's weight of being realised against null is a sister with a null, ⊕ again one level down (Izzy's nulls-and-levels theorem), and the code already holds the carrier. `projections`, `weighted_extension` and `cost_of` in `cocoonml/schema.py` read the three projections off the same amplitudes; the extension now rides on the carrier, so meaning and form are one table.

The count, restated over the semiring: demand and carried are per projection, frequency pinned by draws, cost by the same draws read at coarse grain, phase only by ambiguous strings; the residual is the product over the projections a window fails to pin, and n_eff, the number of projections it pins at all, is the first number the apparatus can read without any theory.

## n_eff, operationally (Claude, 2026-10-08, marked)

The independent projections of the carrier are two, modulus and phase; cost is the modulus at coarse grain and is pinned whenever the modulus is. For a window of artifacts, per node: the modulus grade is √visits over ρ, capped at one, a frequency being pinned to about √n levels; the phase grade is √(ambiguous visits) over φ, capped at one, a phase showing only through strings with more than one reading of nonzero amplitude. n_eff is the sum of grades over nodes, over twice the node count: a ratio of counts, the share of what a window pins of what the table has. `n_eff` in `cocoonml/schema.py`; tests: the modulus grade rises with artifacts, the phase grade is zero without ambiguity and rises only through it. Found by the test: a zero-amplitude derivation is not a reading, so ambiguity counts readings the world can produce, not every bracketing the grammar admits.

## The apparatus after round one against sweeps 02 (Claude, 2026-10-08, marked)

What the second sweeps' review changed in the code, each with its test. The step: one grain per node at most per move, the ends reflect (a share at zero is a removed sister and it returns), and with the origin as centre a move is pulled back with probability displacement over radius, the restoring pull of the mean-reverting form the step section derives; the index no longer freezes. Beside n_eff, which counts visits and cannot see a moved share, the identification error: the distance between the shares a reader estimates off the window's artifacts (every derivation the shape admits, a string counting once) and the table's own, with coverage. The exact floors: the entropy of the next form given the forms before it, and of the extension given the string, from the classical joint the harness samples, so a loss is read as its excess over the floor of the table it was read on. Contexts that fill, for training and reading, and curves only at the ranks every context reaches (a residual selection remains through that minimum). And one correction from the fresh read: the predictive distribution is the conditional of the string distribution, interference acting among a string's own derivations only; as first written it let different strings interfere (the count review, item 9). All in `cocoonml/schema.py` and `cocoonml/harness.py`.

## The apparatus after round one against sweeps 03 and 04 (Claude, 2026-10-08, marked)

From round one against sweeps 03: pairs in the input (`pack_pairs`: the extension token follows its separator as a token the model reads, so the loss at the k-th separator is the extension of the k-th string given k − 1 pairs in context; before it, the extension was only ever a target and meaning had no in-context channel); a move on the incidence kept apart from the step (`remap`: spoken leaves given another cut, the strings unchanged, only the extensions moved; `incidence_delta` reads it; the knob is off unless Izzy turns it on, since the design of 2026-10-08 afternoon fixed meaning and the count puts the two predictions' disagreement there); a within-artifact position in the model; training to a plateau rule. From round one against sweep 04: layers in the model (one layer of attention reads at most the in-context marginal of the extensions; the token after a match takes two: the induction circuit, from training, marked); the scrambled-pairs control (`scramble_extensions`: the earlier pairs in a window made wrong, the targets kept, so a curve that does not rise under it did not read the pairs); the identification excess over the sampling floor (`identification_excess`), the reader that compares across tables of different share entropy; the plateau rule on a held-out evaluation. Still to place on the axes: every point by its distance from the training table and the moved table's own entropy (the consistency and resolution coordinates the page names), which the script measures and the curves did not use. All with their tests in `tests/`.

## The adequate size, derived from the counts; and what the sweeps trained on (Claude, 2026-10-08, marked)

Izzy: the adequate size should be derivable, conceptually, and at least empirically to an extent. Three counts the schema fixes give it; the numbers are for sweep 05's table (forms 4, nonterminals 3, resolution 4, seed 1), read from the table, not from any run.

1. **The match's rank.** Reading a pair is a match and a copy. The match keeps every pair of distinct strings apart, so its rank is the table's effective string count: 24 distinct strings here, 12.6 effective (3.65 bits), of length one to five. A string is identified by its last four tokens (the last one gives 3 classes, two give 8, three give 18, four give all 24). The model supplies rank by width, the dimension the match is scored in, and by layers, each of which hands a position one more token of its string. So the width lies between the log of the count (4.6, with dense codes and a score scale the mantissa must pay for) and the count itself (24, orthogonal keys), and the layers for the full match are four hops and the readout, five, unless the width hashes the string. Sweep 05's model has width 4 and two layers: below both floors. The calibration cells (width 8, two and four layers, drift off) measure where the control leaves zero between those floors.
2. **The pairs' saving, and the crossing.** The extension is a function of the string at this table (the meaning floor is zero), and over the string distribution it carries 2.03 bits per pair; the sum-of-cuts token the sweeps read carried 1.39 (decisions.md 17, 27). A window of length 40 holds about eight pairs, so about 16 bits per window. The match's own description is its two score matrices at the width, 2w² parameters at the mantissa's grain; with a few bits per parameter (the field's measured figure, marked as an import) that is of the order of a thousand bits at width 24: a hundred windows, which every sweep had many times over. So at this size the budget was never the constraint; the rank was.
3. **The ensemble.** A model that accounts for drift holds the training ensemble: tables times bits per table (per node: a share, a phase and two orders at log₂ρ each, 8 bits at ρ = 4, 24 bits for this table) plus the states the drift reached in training. Small here; it scales the model only when the ensemble does, and the ensemble is the experiment's antecedent, not yet fixed (how many tables, how far the index walks during training).

**What the sweeps trained on.** Every sweep drew its training contexts from the stationary table; drift entered only at reading (decisions.md 28). That is the regime where Izzy's prediction says there is no difference between the predictions, and the design Izzy settled has drift in training. So nothing run so far bears on their prediction's antecedent, and my boundary, restated as the radius training reached, was tested with that radius at zero. The mechanism that fixes it is the stream: training contexts produced while the index walks within a training radius, readings at distances beyond it; its numbers are Izzy's.

## A prediction lodged elsewhere (Izzy, 2026-10-08 late, in a Google Search AI Mode thread; records gemini/exports/gemini-session-02.md)

Verbatim: "the more noise there is at training, the more likely a model is to prepare for it. the impetus for the mechanism is obvious in this light. the more noise, the more the weights account for learning." And the shape it predicts: "less noisy training --> the model will be good at its trained tasks initially. more noisy training --> the model will initially be worse at its trained task." Also there: "training is inducing just a fictitious force", and the information horizon "is NOT only a function of the family of problems the ai was trained on." Recorded here so the runs can read against it: the training-noise axis is the drift the ensemble carries (the ball's radius), and the curve's early ordering by noise is a reading the complete system can take before any drift reading.

## Claude's prediction on the training curve, lodged before Izzy's full statement (2026-10-09, 02:15; Izzy asked for it so neither primes the other)

The axis is the drift training contained (the ball's radius, with the step rate); the readings are the complete system's: the form curve by rank inside the ball (the trained task), outside it (the untrained task), and the order control. Mine, marked as mine, before reading theirs:

- **Trained task, initially:** low noise ahead (a stationary target is fitted fastest); high noise behind, its target moving and its marginal broader.
- **Trained task, after a long training:** low noise still ahead on the origin table by a small persistent margin; averaged over the ball no gap, the high-noise model having learned the ball.
- **Untrained task, initially:** both poor; high noise slightly ahead from the start, its marginal covering more.
- **Untrained task, after a long training:** high noise clearly ahead. The low-noise model's recovery drops sharply at the training radius; the high-noise model's decays past it with a scale set by the radius, graded, not flat.
- **The crossing:** the two curves cross, low noise ahead early and on the trained task, high noise ahead late and outside; the crossing comes later in training the larger the radius.
- **The extremes:** no noise, the stationary corner: no recovery of drift beyond the family and the order control at zero (there is no succession of states to read). Maximal noise, the free walk at a high step rate: the model learns the marginal over the whole space, the recovery curves are flat because nothing is out of distribution, and the control falls back toward zero because the window carries no index. So the control against noise is non-monotonic: zero at both extremes, positive between, with its peak where the drift per window is readable.

*Amendment to the above (02:40, after Izzy named the factor; the lodged version stands as lodged):* the number of examples in the window, k, is a factor of every cell. Early in training k does little for either model, the circuit not yet formed, the control near zero at any k. After a long training both rise with k inside the ball. Outside it, low noise stays flat in k beyond the training radius (the boundary), and high noise rises with k with a slope that decays with distance.

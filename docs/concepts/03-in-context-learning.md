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

## The count (Claude, 2026-10-09; the definitions above turned into arithmetic; for Izzy's strikes)

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

ρ_eff = min(ρ, √(kπ), 1/(sλ)).

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

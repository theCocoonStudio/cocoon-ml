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

Izzy (2026-10-08, verbatim): "the drift opens the family. and it's graded. just like i could communicate with people in china if neither of us spoke each others language. if you train with non deterministic data, the weights account for the drift predictably. so we disagree. if the model is trained on deterministic data, there is no difference. the model only tracks the family."

Claude (2026-10-07): the prompt selects within the class the weights fix; recovery works for remappings within the family training contained and fails sharply, unimproved by examples, for meanings outside it.

They separate on out-of-family remapping under non-deterministic training. "Class change" (Izzy, 2026-10-07) was withdrawn by Izzy as a name that admits every solution.

## Hurdles

The lexicon and grammar (Izzy's). Compute in pure Python at toy size. The proxy for consistency. An operational definition of "outside the family". The convergence theorem.

Check: a result is written only after it is read; both predictions stay as lodged.

# The asymptote: the direction, the goal formalized, and the first new lemma (2026-10-17, with the owner; for strikes, one item at a time)

Claude, from the exchange with the owner on 2026-10-17 (records `claude/exports/2026-10-17-session-a739c7ab.md`, 18:4x to 19:2x UTC), after round 1 on Gemini's proof of session 03 (`docs/reviews/2026-10-17-gemini-proof-round-1.md`). The owner's method: step 1 the direction, step 2 the goal formalized according to the direction, step 3 the proof. Keyboard notation; nothing used before it is typed; the holes are typed and left to the owner. Every item is for a strike. Where this document meets `general-model.md` and `fixed-index-bound.md` it says so; it adds to them and replaces nothing.

## 0. What round 1 left

- S = 1/Delta E (session 03, p. 93, written, not derived) inverts under the energy taken in round 1 (Delta E = T * (H(prior) - H(posterior)), the resolution achieved): the model that resolves more scores less, and 1/Delta E is undefined at zero resolution. So the asymptote cannot be a limit of energy; it is a limit of resolution at fixed weights as the distance grows, and S needs its own definition before that limit is restated. Its unit: in this theory every admissible number is a ratio of counts at a vantage, unitless; a number with a unit imports a vantage from outside; 1/Delta E carries one over energy, which changes with the unit of T. If S is 1 minus a share, it has a floor, a ceiling and no unit, and the limit is a statement that can be true or false (the owner's question, 2026-10-17).
- Of the proof's four steps only step 1 holds, as a direction; steps 2 to 4 are names (the energy with its inverted S, the operator with no Psi, the single-node head with no construction). One object survives as a definition: the output map over the vocabulary (round 1, taken).

## 1. The holes, typed (the owner's; the order of construction)

1. S: what it measures.
2. The distance between w_p and w_t: which count it is a ratio of (the page: "a scale between two vantages", no count named).
3. "Resolve x": the predicate on the output distribution (the page: a threshold, empirical, drifting, none given).
4. Psi(w_t, w_p, x): its space, and how w_p enters, since the model has no access to w_p (section 3 below).
5. The fixed-index bound, first form (`fixed-index-bound.md`, 2026-10-09), unstruck.
6. The r_train row (`03-in-context-learning.md`, 2026-10-16), unstruck.

## 2. Step 1, the direction

The legacy model computes M(w_t, x); the general model M(w_t, w_p, x) (session 03). As a direction: where the window's information lands, in the activations through fixed weights (legacy) or in the weights (general); the seam of 2026-09-16.

- Its failure condition is a rate, not emptiness (the owner, 2026-10-17: "isnt this a computability problem, really. model has to be big enough to absorb noise, and then it's a matter of if and when, not just if"): if the legacy model reaches the same resolution at a cost that does not grow with the distance, the direction says nothing. The literature's form of the objection is in-context learning as implicit gradient descent (von Oswald 2022, `docs/literature.md`); what answers it is a bound on the cost at which a fixed w_t reaches a resolution from x, hole 5.
- The legacy model also depends on x, and that dependence is in-context learning; the direction is "x reaches the weights" against "x reaches the activations only", never "x enters" against "x does not".
- w_p is the world's, not the sample's (the owner's correction: "that implies w_p is entirely dependent on x. conceptually, that's wrong"): see D4 below.

## 3. Step 2, the goal formalized according to the direction

- **D1. Index and protocol.** At each index t a protocol P_t, a distribution over pairs of form and meaning; the drift moves the index. (`general-model.md`, Parts I to III.)
- **D2. Distance.** d(t, t'), a ratio of counts in [0, 1] between two indices. Which count: hole 2.
- **D3. Window and model.** A window: k pairs drawn at index t' and one query form. A model is a map from a window to a distribution over V (the output map, round 1).
- **D4. The world's weights.** W(t): the weights training on P_t produces at size n. w_t = W(t), w_p = W(t'). A function of the index alone; no sample is an argument of it.
- **D4'. The two classes, by where the window's information lands.** L computes f(W(t); window). G computes f(G_hat; window), where G_hat is an estimate of W(t') from W(t) and the window alone, no oracle; the estimate's error is the residual (the page; `fixed-index-bound.md` D5), the part of W(t') that no window of length k identifies. The residual bounds every member of G and of L alike. L is the member of G whose estimate is W(t) itself, so every statement below is about the best member of each class at size n, for every n above some floor (the owner's "big enough to absorb noise").
- **D5. Resolution.** R(M, t', k): the share of the uncertainty about the meaning that M's distribution removes, a ratio of counts in [0, 1]. Which count: hole 3. (`general-model.md` 4.4 has the excess over the floor; R is its complement as a share.)
- **D6. Cost.** k_M(r, d): the least k at which R(M, t', k) reaches r at every index t' at distance d from t; infinite if none does. This is `general-model.md` 6.1's recovery read the other way: rec gives the resolution at k, k_M gives the k at a resolution.
- **The statement.** For r below the identifiable share at k (the residual's complement), the function d to k_L(r, d) against d to k_G(r, d). The asymptote as the proof meant it: a d* beyond which k_L(r, d) is infinite while k_G(r, d) is finite everywhere. The owner's prediction (2026-10-08, 2026-10-09; `general-model.md` 6.2): k_L finite everywhere, growing with d, the growth smaller the more drift the training carried. One function, two shapes; the r_train row measures it at three values of the training drift. The direction's content is the difference between the two costs above the identifiability floor, which is the same floor for both classes.
- **What this adds to `general-model.md`.** Part V defines readers given the law and the trained reader; none whose parameter depends on the window. One definition to add, 4.5: a reader whose parameter is a function of the window (the class G); Part VII's statement is then about two readers.

## 4. Step 3, the first lemma that is new: the moving index (for the owner's strike)

`fixed-index-bound.md` Lemma 1 gives the sampling floor at a fixed index (derivation). Its section 7 left the index moving inside the window to the owner; the owner asked on 2026-10-17 for the steps to continue as they unfold.

**Lemma 4 (the moving index).** Let the window hold k artifacts drawn while the index moves, and let the share at node v move by at most L_v per step, s steps across the window. The frequency at v estimates a target that has moved by at most s * L_v, so its bias is at most s * L_v while its sampling error is about 1 / (2 * sqrt(n_v)); the share is resolved to about min(rho, sqrt(n_v), 1 / (s * L_v)) levels. If the index steps once per artifact, s grows with k while n_v = k * pi_v grows with k, so the resolution is largest at a finite k, of order (pi_v / L_v^2)^(1/3), and no k removes the floor. **Status:** derivation, first order, the bias taken as the range of the walk; the mean-reverting walk bounds the range by its radius whatever s, which caps the loss at large k. **Disproof:** a reader that pins a share finer than 1 / (s * L_v) from a window over which the share moved that much. **What it gives the statement:** a finite asymptote of recovery that is a property of the world and the window, before any reader is named; the first number in the bound B of `general-model.md` 6.3 that is not the sampling floor.

## 5. Prompt-time tasks, measured (the owner's questions, 2026-10-17)

- **Success at prompt time, and its tie to training.** In this world it is the same quantity read against a different law: the loss of 4.2, at prompt time the excess of that loss over the floor the reading law sets (4.4), recovery its fall with k (6.1). The tie is exact because the law is ours; training enters only through what the training law shares with the reading law, the identified part (6.3). For an English task the tie is hole 3: the threshold is empirical and nobody holds the law.
- **Orthogonal task, quantitative evaluation.** The task is orthogonal by the distance of its index; its evaluation is tied to the world at that index, not to training. The world's truth must exist and never be an input (the design of 2026-10-08).
- **Math and tokens.** Two seams, at training and reading alike: the readout (a distribution over V against the law; inside, integers) and the packing (forms to tokens, k strings with a separator, 3.1 and 3.2). The choices sit at the seams, not inside: the leaf forms with the empty form, the separator, the alphabet's cardinality. English adds the tokenizer outside the model, the orthography object.
- **"The model found a problem in the task."** The fixed readout has no slot for it. To account for it the output space must contain a statement about the task and the evaluation must hold the truth about the task. In this world both exist: a window under drift is inconsistent with every single table because the index moved inside it; that movement is the world's truth (`corpus.window_truth_distances`), and a model that reports it is scored against it like any output. Marked as Claude's: that second readout is G's estimate of the index, read out instead of hidden; "found a problem in the task" is the general class's defining output, and reading it is how the class becomes measurable.

## 6. The slope and the tasks (the owner's posit, 2026-10-17)

The owner: "if you focus on the wrong tasks, you cant bring out the slope"; "the tasks have to be varied enough"; quantify it "with task variables, and unknown out of scope variables". A slope is a relation among tasks ordered on the axis the prediction bends on (distance from the training index), with the evaluation held fixed across them; a point has no slope (the level-up: shuffle the tasks and the slope is gone). Quantified as the statistical layer already has it (decisions 50 and 53): the levels on the axis and the windows per level, against the null's envelope; "varied enough" is the ratio of the spread across levels to the spread within a level, a count over a count; a slope needs two levels, a bend three, each level's band narrower than the step. The unknown variables are not listed; they are the part of the within-level spread the known variables do not account for, measured by the envelope since the null is computed from the readings. The training side: the model tracks a variable only if its training varied it (the r_train row's reason), and the training's spread on the axis is set against the reading's. In the framework's words: the task set has a rank from the reader's vantage, and "enough" is the resolution along the axis.

## 7. Next

The owner's strikes, one item at a time: S (hole 1) first, then D2's and D5's counts, then Lemma 4; the definition 4.5 into `general-model.md` once struck or kept; the proof's order after: the identifiability floor (fixed index, then Lemma 4), then the identified part of the two samples, then the two classes compared above the floor.

# The rule table: how the runner moves the parameter vector

Claude, 2026-10-09. The design of `runs/loop.py` (the self-adjusting runner), written before it so the only choices in the loop are in this table, each row citing the count it comes from. The loop runs sweep 07 cells at a parameter vector, reads them, applies the first row whose condition holds, appends one entry to `runs/loop-log.md`, and repeats; it stops when no row fires or the next vector exceeds the host. Numbers from the cells stay in the scratch logs under the rule for numbers (two rounds, two replications); the log entry carries the vector, the row that fired and the comparisons it made, not the readings.

## The parameter vector

θ = (w width, ℓ layers, L window length, C corpus artifacts, N read artifacts per radius, cap training steps, η learning rate), with the ensemble fixed for a loop: the tables in play (count and seeds), the training radii (the noise axis e, one cell per value), the read radii, the step and switch rates. The ensemble is the experiment's antecedent and is not moved by any row.

## The derived floor (never run below it)

| element | value | the count it comes from |
| --- | --- | --- |
| w | the number of distinct strings the tables in play admit (the union over the tables), rounded up to a multiple of 8 | the match's rank: a key per string, orthogonal (page, "The adequate size", 1); the recount for graded tables is in `docs/decisions.md` 33 |
| ℓ | the suffix depth that identifies a string, plus one | each layer hands a position one more token of its string; the readout is the last (page, 1) |
| L | at least 8 whole artifacts per window: L ≥ 8 · (mean string length + 1) | the readings are curves in k, the artifact rank; a rise and a plateau need ranks past the suffix depth (page, the predictions as they separate) |
| cap | past the crossing: at least 4 × (2w² · b) / (bits per window), b = 3 bits per parameter (an import, marked), bits per window = whole · log₂(effective string count), in batches of 4 windows | page, 2: the match's description against what a window carries |
| C | at least 0.4 · cap · ā artifacts, ā the mean number of artifacts a window holds, so that cap steps of four windows draw each window at most ten times on average (ten, a few passes, is an import, marked) | the ball must be learned, not the corpus (decisions 21, 28); the corpus is finite by design and its size is the amount the world produced |
| η | 0.1 at the derived scale of initialisation | decisions 19; the scale, decisions 32 |

## The rows (applied in order; the first that holds fires)

| row | condition (read from the cell's JSON) | move | the count behind it |
| --- | --- | --- | --- |
| R1 instability | any held-out evaluation exceeds the one before it by more than a factor of 2 | η ← η / 2; re-run the vector | sweep 05a's arithmetic: two cells died of overflow, a third doubled (`docs/reviews/2026-10-08-sweep-05-round-1.md`); a step that doubles the loss is past the step size's range |
| R2 budget | `hit_cap` and the last evaluation improved on the one before it by more than the tolerance | cap ← 2 · cap; C adjusted by its floor; re-run | the plateau rule (decisions 20) only reads a plateau it reached; the crossing (page, 2) is a floor on the budget, not its value |
| R3 the k range | `whole` < 8 at any read radius | L ← ⌈1.5 L⌉; re-run | as the L floor above |
| R4 replication by size | the three sizes (w, ℓ), (⌈1.5w⌉₈, ℓ + 1), (2w, ℓ + 2) have run at this vector and a reading's sign differs among them: the slope in k of the form excess past the training radius at ranks 1 to 4, or the ordering of its level at the last rank against the inside reading, or the sign of the order control | add the next size (⌈1.5 · largest⌉₈, ℓ + 1 of the largest), drop the smallest; re-run | decisions 5: replication by size, not seed; a reading that moves with size was taken below the rank (page, 1) |
| R5 the floor | at the derived size, inside the ball, the form excess over `entropy_floor_form` at the last rank is above the sampling band of that rank (the identification excess of a window with `whole` artifacts, `schema.identification_excess`) and falls at the next size | treat as R4 (the size is binding) | the sampling floor: a window of k artifacts pins a share to √(kπ) levels (`docs/proofs/fixed-index-bound.md`, Lemma 1); excess above the band at the floor size is the model, not the count |
| stop | no row fires at three agreeing sizes | the readings stand for round one | — |
| host | the next vector's estimated time exceeds one hour per cell on this host (12 cores, 32 GB; the estimate from the last cell's seconds scaled by (w'/w)² · (ℓ'/ℓ)) | stop with the rows that still fire named in the log | the host is the only bound not in the counts |

## What a row may not do

No row moves the ensemble (tables, radii, rates): those are the antecedent of the predictions and are Izzy's. No row changes a reader. No row reports a number: the log entry says which comparison held, and the numbers wait for the rounds. No row lowers a size below the derived floor.

## The log entry

One line per iteration in `runs/loop-log.md`: the date and time, the vector, the cells run (names and seconds), the row that fired with the comparison it made (as signs and "held" / "did not hold"), and the next vector. The scratch logs hold the JSON lines; the summariser merges them for the round.

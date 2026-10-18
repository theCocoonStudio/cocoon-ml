# The training-drift pilot (sweep-08b and reference8), 2026-10-18: the design, the choices, what each outcome shows; the reading to follow

Claude, alone, 23:21 UTC (the owner, 22:5x: "any runs available while im still away?"). A pilot, not the r_train row of concept 03: the row is for the owner's strike and its null and statistic are not written; this pilot checks that the apparatus trains under drift wider than radius 1, measures the reading the fork of `docs/proofs/asymptote-goal.md` section 7 needs, the unresolved share S by rank at a distance (general-model.md 6.4), and places the row's knobs. Numbers stay in the logs under the rule; this file says what was run and what the instrument reads.

## The cells

- The world and the model of decisions 55 and `runs/launch-08.sh`: the Markov world, four states at resolution 8, Dirichlet 1 with end 1, two chains, corpus 42 000 artifacts, windows of 64 tokens; the array model at (96, 7), the derived scale, rate 0.05, mode cap to 2 488 steps with the three copies; seed 1.
- Two cells of 2026-10-11 reused through their saved copies: `sweep-08-s0-w96-lr0.05` (no steps: the training walk stationary at the origins, radius none) and `sweep-08-s1-r1-w96-lr0.05` (steps 1, training radius 1 grain).
- Two cells launched 2026-10-18 23:18:08 UTC (`~/.claude/scratch/runs/launch-08b.sh`, from the archive of 6416d32, the head of main): `sweep-08b-s1-r3-w96-lr0.05` and `sweep-08b-s1-r6-w96-lr0.05`, training radii 3 and 6 grains with steps 1, read radii 0, 1, 3, 6 and free. The first cells trained under drift wider than one grain.
- The instrument, `runs/reference8.py` (this branch): for a cell and a copy, the reading corpora regenerated from sweep8's seeds, the model's form loss by rank, the Bayes floor under the reading law (exact at radius 0, the particle filter with 200 particles elsewhere) and the marginal reader (the reading windows' counts) by rank on the first 64 windows of each reading, and S by rank; the record's cut at the rank every window reaches, and a deep curve past it with counts. Run on the best copy of every cell and the marginal copy of the radius-1 cell (`launch-ref8-a.sh`, then the new cells' copies).

## The choices (listed before the reading, the owner's protocol)

1. The world: the Markov world rather than the schema's, because its references exist (decisions 54, 55) and S needs both.
2. The training radii 3 and 6 grains beside the record's 0 and 1: the bound of the walk is five sixteenths per step at this world, so 6 grains is three quarters of the share scale, the wide end; 3 is between. Nothing derives these two numbers.
3. Steps 1 in training for both new cells, as the radius-1 cell: the radius is the one knob moved.
4. The reading radii 0, 1, 3, 6, free for the new cells (the record's cells read 0, 1, 2, 3, free): the training radii themselves plus the free walk, so each cell reads at its own radius and beyond.
5. 64 windows per reading for the references (about 3.5 s a window for the floor at 200 particles, measured 23:0x): the first 64 of each reading corpus, not a draw.
6. The marginal reader of S is the reading windows' own counts (4.3's R0 under the reading law); the training windows' counts are recorded beside it.
7. The deep curve's rule: no cut, the count of positions per rank recorded; the record's rule (ranks every window reaches, sweep 02 round one) is kept as the main curve. At length 64 in this world every window reaches two whole artifacts, so the main curve has two ranks; the k axis the fork needs is longer than this window gives (a design fact for the row: a longer window or a shorter artifact).
8. One seed. A pilot.

## What each outcome shows (written before the reading)

- If a new cell escapes or sits on the marginal to the cap: training under wide drift fails at this size or rate; the row needs the rate rule rerun at these radii before anything else.
- S at the free reading (the farthest), best copy, by training radius 0, 1, 3, 6: the fork's first look. Falling with the training radius is the direction of the owner's prediction (recovery past the training radius rises with r_train) and of Lemma 5's corollary (kappa_v falling with rho_t); flat within the spread is the degenerate case; rising is neither and a finding against both.
- S at the training radius itself across the four cells, against S at the free reading: the gap between inside and outside the ball per cell; a gap that shrinks with the training radius is the pull washing out.
- The marginal copy against the best copy at radius 1: whether S moved with training at all (the instrument's sanity).
- The ball's cardinality per cell (`ball_states`): the axis of Raventós et al. (literature entry 13); whether the four cells sit on one side or both of a threshold is read off these counts; at radius 1 the record holds 21 729 distinct states of 42 000 artifacts, so this world is at the high-diversity end in their terms and the radius, not the count, is the axis here.
- None of these is a result: one seed, one size, no null written; the row's form (decision 53) comes after the owner's strike.

## What the instrument read

(To follow, from the logs `~/.claude/scratch/runs/reference8-*.log` and the two cells' JSON lines.)

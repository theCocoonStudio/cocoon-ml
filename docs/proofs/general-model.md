# The general model: definitions

Izzy and Claude, 2026-10-11, 00:5x to 03:xx, for Izzy's strike by section. The two-month goal is a general model in which the pattern behind in-context learning surfaces, with the transformer and every other reader as refinements of it (Izzy, 2026-10-11 01:xx: "the model has to be the same, just general"). This document defines that model and nothing else; the bound is stated at its end as the thing to be proved, and the proof is a separate document that takes these definitions and massages their arithmetic (Izzy: "the proof has to take the models math and massage it to find the bound").

**Conventions.** Textbook style: definitions, lemmas with proofs, remarks. A symbol appears on the right of a definition only if it has appeared on the left of an earlier one (Izzy, 2026-10-11: "we cant use variables we havent defined, even if we know them"). Plain keyboard throughout: no Greek, no subscripts the keyboard cannot type; a_i is written a_i. Sets are named by what they contain and their elements are given a name in the same line; a type is never asserted by a membership statement (Izzy: "disjoint is a membership statement, not a type statement"). A + B is the coproduct (disjoint union by tag), A x B the product, A* the finite sequences over A. Where a definition is the one the apparatus runs, the function in `cocoonml/` is named in a remark; the remark is a pointer, not a premise. Imports from the literature are marked **import** with their source and their premises. Draws are classical throughout: the phase, which the schema carries where a string has two derivations, is deferred to the generalisation (Izzy, 2026-10-11 02:xx, confirmed: classical first).

---

## Part I. The world

### 0.1 Shape

**Definition 0.1 (shape).** A shape is a quadruple Sh = (N, F, i0, E) where

- N is a finite nonempty set; its elements are called nodes. i0 is an element of N, called the start node.
- F is a finite nonempty set; its elements are called forms. The null leaf is a single element, written 0, of a third kind. Leaf := F + {0}; an element of Leaf is called a leaf. Every form is a leaf, the null leaf is a leaf, and no leaf is both.
- Sym := N + Leaf; its elements are called symbols.
- E is a function from N to (Sym x Sym) x (Sym x Sym). For a node i, write E(i) = (e_i0, e_i1); each e_ik is an ordered pair of symbols, called an expansion of i, its two symbols called first and second. There is no condition on which symbols occur: a node may occur in the expansions of any node, its own included.

Nothing else belongs to a shape.

**Remark (the refinement).** The apparatus's shapes (`schema.generate`) satisfy one further condition: a strict total order on N with i < j for every node j occurring in an expansion of i, i0 the least node. Under it no derivation can return to a node and every derivation is finite by the shape alone. The general form drops it; whether a draw ends is then a property of a table's weights (Lemma 0.5.1).

### 0.2 Grid

**Definition 0.2 (grid).** Let r be a positive integer, called the resolution. The grid at resolution r is G_r := { k/r : k = 0, 1, ..., r }. Its elements are called levels; 1/r is called the grain.

### 0.3 Table

**Definition 0.3 (table).** Let Sh be a shape and r a resolution. A table on Sh at r is a pair T = (s, o) where s is a function from N to G_r, s(i) written s_i and called the share of node i, and o is a function from N x {0, 1} to G_r, o(i, k) written o_ik and called the order of the expansion e_ik. Tab(Sh, r) is the set of all tables on Sh at r; it is finite, with (r + 1)^(3|N|) elements.

A table assigns levels and nothing else; what a share and an order do is said in 0.5. The ratios of T are the 3|N| levels it assigns; the index set of the ratios is Q := N x {share, order0, order1}, and T is read as a function from Q to G_r when a ratio is addressed without saying which kind it is.

### 0.4 Derivation, yield, string

**Definition 0.4.** The start node is i0. A derivation from a symbol x is defined by recursion on the symbol: if x is a leaf, there is exactly one derivation from x, written [x], with no choices; if x is a node i, a derivation from i is a tuple (k, m, d1, d2) where k in {0, 1} selects the expansion e_ik = (first, second), m in {0, 1} selects the order (m = 0: first before second; m = 1: second before first), and d1, d2 are derivations from the two symbols of e_ik taken in the selected order. The choices of the derivation are (k, m) at i together with the choices of d1 and d2. Der(Sh, x) is the set of derivations from x; Der(Sh) := Der(Sh, i0).

The yield of [x] is the one-element sequence (x); the yield of (k, m, d1, d2) is the yield of d1 followed by the yield of d2. The string of a derivation is its yield with every occurrence of the null leaf removed, an element of F*.

**Lemma 0.4.1 (Der(Sh, x) is countable).** Every derivation is a finite tree with at most two children and four choices per node, so the derivations from x can be enumerated by size; the set is finite when no node can recur and infinite otherwise.

### 0.5 Weight, draw, termination

**Definition 0.5.** Let T = (s, o) be a table on Sh. For a node i write c_i(0) := s_i, c_i(1) := 1 - s_i; for an expansion e_ik write c_ik(0) := o_ik, c_ik(1) := 1 - o_ik. The weight under T of a derivation d is w_T(d) := the product over every choice (k, m) made at a node i in d of c_i(k) * c_ik(m), and w_T([x]) := 1. Every factor lies in [0, 1].

The draw from a symbol x under T is the random procedure that stops at a leaf; at a node i selects k = 0 with probability s_i and k = 1 otherwise, then m = 0 with probability o_ik and m = 1 otherwise, each selection independent of all others; and continues into the two symbols of e_ik in the selected order. A run that reaches a leaf on every branch produces a derivation d with probability w_T(d). The termination mass is Z_T(x) := the sum of w_T(d) over d in Der(Sh, x), and 1 - Z_T(x) is the probability that the draw from x never ends.

**Lemma 0.5.1 (termination), import.** Let M_T(i, j) := the sum over k of c_i(k) times the number of occurrences of the node j among the two symbols of e_ik, restricted to the nodes reachable from i0. Then Z_T(i0) = 1 if the largest eigenvalue of that restriction is below one and Z_T(i0) < 1 if it is above one; at exactly one, Z_T(i0) = 1 unless every reachable node produces exactly one node with probability one. Source: the extinction theorem for multitype branching processes with finitely many types (Harris 1963; Athreya and Ney 1972). In the refinement of 0.1 the restriction is strictly triangular, every eigenvalue is zero, and Z_T(i0) = 1 for every table.

### 0.6 The law of a string

**Definition 0.6.** An element of F* is called a string. For a string a, P_T(a) := the sum of w_T(d) over all d in Der(Sh) whose string is a. The sum of P_T(a) over all strings is Z_T(i0). When Z_T(i0) = 1, P_T is a probability measure on F*, called the law of an artifact under T, and an artifact is a string drawn from it. Nothing below uses a table unless Z_T(i0) = 1.

**Remark.** In the refinement, P_T is `schema.classical_joint` with the extension summed out.

### 0.7 The predictive of a table

**Definition 0.7.** A string p is a prefix of a string a if a is p followed by some string, possibly empty. The prefix mass of p under T is m_T(p) := the sum of P_T(a) over all a having p as a prefix. Let end be a single element of a new kind; the outcomes of a prediction are the elements of Out := F + {end}. For a prefix p with m_T(p) > 0 and an outcome y, the predictive of T at p is

  P_T(y | p) := P_T(p followed by y) / m_T(p) if y is a form,  and  P_T(end | p) := P_T(p) / m_T(p).

The outcomes' predictives sum to one at every such p, since every string with prefix p either equals p or continues it by exactly one form.

**Remark.** `schema.predictive` is this with the interference distribution in place of P_T; the classical form is the one used here.

---

## Part II. The index

### 1.1 Moving one ratio

**Definition 1.1 (a move).** Let r be the resolution and g := 1/r the grain. For a level x in G_r and a direction dir in {-1, +1}, the move of x by one grain is

  mv(x, dir) := reflect(x + dir * g),  where reflect(y) := 2 - y if y > 1, -y if y < 0, and y otherwise.

Since x is a level, x + dir * g lies in [-g, 1 + g] and reflect brings it back into G_r: a level at an end of the grid moves inward. A move never leaves the grid and no level is absorbing.

### 1.2 The step kernel

**Definition 1.2 (step).** A step is specified by a bound b > 0, a lean l in [0, 1], a centre C in Tab(Sh, r) and a radius rho > 0 (or no centre, written rho = none). Given a table T, one step produces a table T' as follows, independently for each node i:

1. With probability p_move := min(1, b / (|N| * g)) the node moves; otherwise its three ratios are unchanged.
2. If it moves, one of its three ratios is chosen uniformly: the share, or the order of e_i0, or the order of e_i1.
3. A direction dir is drawn uniformly from {-1, +1}, then adjusted twice:
   - the lean, for the share only: if exactly one of e_i0, e_i1 contains the null leaf among its two symbols, then with probability l the direction is set toward that expansion (+1 if it is e_i0, -1 if it is e_i1);
   - the pull, when a centre is given: let v be the chosen ratio's level in T and c its level in C; if v differs from c, then with probability min(1, |v - c| / rho) the direction is set toward c (+1 if c > v, -1 if c < v).
4. The chosen ratio becomes mv(v, dir); the node's other two ratios are unchanged.

K(T, T') denotes the probability that one step from T produces T'; K is a Markov kernel on Tab(Sh, r). The expected number of grains moved in one step is at most b, one per node at most.

**Remark.** `schema.step` is this procedure, with one more component: each node's phase is jittered by a quarter grain and wrapped. The phase is not in Part I (classical draws) and is not in this kernel; it joins both in the generalisation.

### 1.3 The walk and the stationary law

**Definition 1.3 (walk).** A walk from a centre C with radius rho is a sequence of tables T_0 = C, T_1, T_2, ... with T_{t+1} drawn from K(T_t, .), the step centred at C with radius rho. States(T_0, ..., T_t) is the set of tables the walk has occupied up to t.

**Lemma 1.3.1 (the walk has a stationary law), import.** With rho finite and every ratio's level reachable in finitely many grain moves, the kernel K restricted to the tables within reach of C is irreducible on a finite set, and aperiodic since a node stays put with positive probability; such a chain has a unique stationary probability measure pi_C on Tab(Sh, r), and the law of T_t converges to it as t grows (Markov chain convergence on finite state spaces; Levin, Peres and Wilmer 2009, Theorem 4.9). pi_C is called the stationary law of the walk at C.

### 1.4 Distance between tables

**Definition 1.4 (distance).** For tables T, T' on the same shape, d(T, T') := the mean over q in Q of |T(q) - T'(q)|, a number in [0, 1]. For a set B of tables, d(T, B) := the minimum of d(T, T') over T' in B.

**Remark.** `schema.delta_magnitude` with phases off, and `corpus.Ball.distance`, are this.

---

## Part III. Production and the corpus

### 2.1 Ensemble

**Definition 2.1 (ensemble).** An ensemble is Ens = (Sh, r, (C_1, ..., C_m), b, l, rho, lambda_s, lambda_w) where Sh is a shape, r a resolution, each C_j a table on Sh at r (an origin), m at least one, b a bound and l a lean (1.2), rho a radius or none, lambda_s >= 0 the step rate and lambda_w >= 0 the switch rate. Nothing in it is moved by what follows.

### 2.2 Poisson count

**Definition 2.2.** For lambda >= 0, Pois(lambda) is the probability measure on the non-negative integers with Pois(lambda)(n) = exp(-lambda) lambda^n / n!. The parity of a Poisson count is odd with probability exp(-lambda) sinh(lambda).

### 2.3 Production

**Definition 2.3 (production of a corpus).** Given an ensemble and a count n, the production is the random procedure:

- The state is (which, T^1, ..., T^m) with which in {1, ..., m} and T^j a table; initially which is uniform on {1, ..., m} and T^j = C_j for each j.
- For each of n rounds, in order: (i) if m > 1, draw a count from Pois(lambda_w); if it is odd, which becomes a uniform draw from {1, ..., m} minus the current which; (ii) draw a count from Pois(lambda_s) and apply that many steps (1.2) to T^which, centred at C_which with radius rho; (iii) draw an artifact a from the current T^which (0.5, 0.6), and record the pair (a, the table T^which at this moment).

The corpus is the sequence of the n recorded pairs; its artifacts are the first components in order. The record of the second components is the production's own, never an input to any reader.

**Remark.** `corpus.produce` is this procedure; the apparatus's initial which is drawn as here.

### 2.4 The ball

**Definition 2.4 (ball).** The ball of a corpus is the set B of tables recorded as second components in its pairs, together with the origins. The distance of a table from the ball is d(T, B) (1.4).

### 2.5 Stationary production

**Definition 2.5.** The stationary production is the production of 2.3 whose initial state has each T^j drawn from pi_{C_j} (1.3.1) instead of being C_j, with which uniform. A corpus produced from a long production is, after its first rounds, distributed as a stationary one; the floor (4.4) is defined on the stationary production, and the record's finite corpus approximates it.

---

## Part IV. Windows

### 3.1 Tokens

**Definition 3.1.** Let sep be a single element of a new kind, the separator. Tok := F + {sep}; its elements are called tokens. A sequence over Tok is a token sequence.

### 3.2 Packing a window

**Definition 3.2 (window).** Let L be a positive integer, the length. Given a sequence of artifacts a_1, a_2, ..., the window of length L packed from it is the token sequence built as follows: starting empty, take the artifacts in order; an artifact a is appended, as its forms followed by one sep, if the sequence so far with a's forms and the sep would not exceed L tokens; otherwise stop. Then append sep until the sequence has exactly L tokens. The artifacts appended are the window's artifacts, k of them, numbered by rank 1 to k in order; the appended seps after them are the padding. The targets of a window t_1 ... t_L are the sequence t_2, ..., t_L, sep: at position i the target is the token at i + 1, and at position L it is sep.

**Remark.** `corpus.pack_window` is this.

### 3.3 The windows of a corpus

**Definition 3.3.** The windows of a corpus are packed consecutively: the first from artifact 1; each next from the first artifact the previous window did not take. A window whose artifacts end the corpus with room for a further artifact of the corpus's shortest length is dropped unless it is the only window. The first artifact of every window but the first did not fit the previous window, so its length is at least that window's padding count minus one; this fact enters the floor (4.4).

**Remark.** `corpus.windows` is this, the dropping rule included.

### 3.4 The law of a window

**Definition 3.4.** The window law W(Ens, L) is the probability measure on token sequences of length L given by: produce a stationary corpus (2.5) of a count large enough that at least two windows exist, pack its windows (3.3), and take the second. Its first artifact is thereby conditioned as every window's but the corpus's first is. Expectations written E_W are under this law.

**Remark.** The reference predictor of decisions 54 computes under this law, the first artifact's length conditioned on the leftover room's distribution over the training windows and the walk started from a burned-in state; the record's held set is sixteen windows of a finite production and approximates it.

---

## Part V. Readers

### 4.1 Reader

**Definition 4.1.** A reader is a function R that assigns to every token sequence u of length between 1 and L a probability measure R(. | u) on Tok. R(y | u) is read as the reader's probability that the token after u is y. A reader may depend on anything fixed before the window is seen and on nothing else about the window than u.

### 4.2 Loss

**Definition 4.2.** The loss of a reader R on a window t with targets as in 3.2 is

  loss(R, t) := (1/L) * the sum over positions i = 1, ..., L of  -log R(target_i | t_1 ... t_i).

Its expectation under the window law is loss_W(R) := E_W[loss(R, t)]. The per-rank loss at rank k is the same average restricted to the positions inside the window's k-th artifact and its separator, written loss_k(R, t), with loss_W,k(R) its expectation over windows that have a k-th artifact.

**Remark.** `array_attention.loss` and `corpus.form_losses_by_rank` are these, the latter over form targets only (the separator excluded) as its docstring says; the restriction is a choice recorded there.

### 4.3 Two readers given by the law

**Definition 4.3 (the Bayes reader and the marginal reader).** The Bayes reader R* under W is R*(y | u) := the probability under W that the token following u is y, given that the window begins with u, defined where that prefix has positive probability. The marginal reader R0 is R0(y | u) := the probability under W that a position chosen uniformly in the window holds the token y, the same measure at every u.

**Remark.** `cocoonml/reference.py` computes R* (exactly with lambda_s = 0, by a particle filter otherwise) and R0 from the training windows' counts; their losses are decisions 54's floor and marginal level.

### 4.4 The floor and the excess

**Lemma 4.4.1 (the Bayes reader has the least expected loss), import.** For every reader R, loss_W(R) >= loss_W(R*), with equality iff R(. | u) = R*(. | u) for W-almost every prefix u. Source: the information inequality (Gibbs' inequality): the cross-entropy of a measure against another is at least its entropy, position by position. loss_W(R*) is called the floor of W.

**Definition 4.4 (excess).** The excess of a reader is ex_W(R) := loss_W(R) - loss_W(R*), non-negative by 4.4.1; its per-rank form is ex_W,k(R) := loss_W,k(R) - loss_W,k(R*).

---

## Part VI. Training, drift, and the two samples

### 5.1 The training law and the trained reader

**Definition 5.1.** Fix an ensemble Ens_t (the training ensemble) and a length L. The training law is W_t := W(Ens_t, L). A training procedure A is a function from a finite sequence of windows to a reader. A training corpus is a corpus produced under Ens_t; its windows, in some order, are the sample S_t. The trained reader is w_t := A(S_t). Its excess on its own law is gap_t := ex_{W_t}(w_t).

**Remark.** The array model trained by `harness.train_to_cap` is one A; w_t is the model at a copy (marginal, best, last). gap_t is the held loss above the floor, read at every evaluation.

### 5.2 The reading law at a distance

**Definition 5.2.** A reading ensemble Ens_p differs from Ens_t in its radius only (or in the walks' centres being displaced; the apparatus moves the radius). W_p := W(Ens_p, L) is the reading law. The distance of a window t produced under Ens_p from the training ball B_t is dist(t) := the mean over t's artifacts of d(T, B_t), where T is the table recorded for that artifact (2.3) and B_t the ball of the training corpus (2.4). It is a number the production knows and no reader sees.

**Remark.** `corpus.window_truth_distances` is dist; the readers' own estimate of it from forms alone (`corpus.window_distances`) is a reader-side quantity and is not used in a statement about the law.

### 5.3 The two samples and the law

**Definition 5.3.** For a reading ensemble Ens_p, the counterfactual reader is w_p := A(S_p), where S_p is a sample of the same size as S_t from W_p: the weights the same training would produce on the world at reading time (Izzy, 2026-10-08, the objects). The model is the pair M(w_t, w_p). Consistency between the two samples is a function of the pair (Izzy, the objects; its form is not defined here and is an open item of the proof). Izzy's law is S = S(M(w_t, w_p), x), with x an opaque value tracked reliably; the bound below takes x = k.

---

## Part VII. What is to be proved

### 6.1 Recovery

**Definition 6.1.** For a reader R, a reading law W_p and a distance band D (a set of distances), the recovery at rank k is

  rec(R, k, D) := E_{W_p}[ ex_1(R, t) - ex_k(R, t)  |  dist(t) in D ],

the fall of the per-rank excess from the first artifact to the k-th, over windows at distance in D. Its asymptote, if it exists, is rec(R, inf, D) := the limit of rec(R, k, D) as k grows (with L grown so the window holds k artifacts).

### 6.2 The predictions, as statements about rec

- **Izzy (2026-10-08, 2026-10-09 04:30).** For the trained reader w_t and distances D past the training radius: rec(w_t, k, D) is increasing in k for small k, has a finite asymptote rec(w_t, inf, D) > 0, and the asymptote is below full recovery (ex_k does not reach zero) by an amount increasing with the distance.
- **Claude (2026-10-07, 2026-10-09 04:30), the degenerate case.** rec(w_t, k, D) = 0 for all k when D lies past the training radius.

Izzy, 04:35: "it's not a different prediction from yours, just a more precise one."

### 6.3 The bound to be found

**Conjecture (the bound).** There is a function B(k, D; Ens_t, Ens_p, A) such that for the trained reader, ex_{W_p,k}(w_t) <= B(k, D) on windows at distance in D, with B decreasing in k, with a positive limit in k for D past the training radius, that limit increasing in the distance, and B's form determined by the definitions above and nothing added: the sampling floor in k (the fixed-index bound, `docs/proofs/fixed-index-bound.md`), the walk's movement inside a window (the step kernel at the window's states), and what the training law shares with the reading law (the identified part of the two samples). Finding B is the work; this document is the arithmetic it is allowed to use.

---

## Where the apparatus meets each definition

Shape, grid, table, derivation, law and predictive: `cocoonml/schema.py` (generate, draw, classical_joint, predictive). Step, walk, distance: `schema.step`, `corpus.Ball`. Production, ball, windows: `corpus.produce`, `corpus.pack_window`, `corpus.windows`. Readers, loss, floor, marginal: `array_attention.loss`, `corpus.form_losses_by_rank`, `cocoonml/reference.py`. Trained reader and its copies: `harness.train_to_cap`, `runs/sweep7.py`. Distance of a window: `corpus.window_truth_distances`. Nothing in the definitions is a reading of the record; the record is where each definition is instantiated.

# Literature: the bridge

Izzy, 2026-10-10 23:1x: the research program's direction, refined. An empirical argument needs a model of what is happening, and a model needs the data it predicts; the step that bridges our record to the field is (1) replicating others' results within our model and (2) explaining other models within ours, done as we go and not last, so that the field's methods and models frame our results on their terms as we build, and may bring answers and insights of their own. The theory (cocoon-relations) is the model side of the bridge, a direct component of the goal; the apparatus here is the data side; this file is the bridge's record. We must be prepared to identify circuits where we can.

Status of every entry below as of 2026-10-10 23:4x: **from Claude's memory of the papers, unverified** (the sandbox has no network; the PDFs are read when they are put in a repo or the firewall admits their hosts by PR). An entry is verified when its claims have been checked against the paper and the citation completed; a claim that fails the check is corrected here, not glossed.

Each entry: their claim; their model; what their result predicts for our data; our replication within the apparatus; the status.

## 1. Edelman, Edelman, Goel, Malach, Tsilivis (2024), "The evolution of statistical induction heads: in-context learning Markov chains"

Izzy's reference ("the paper describing circuits with Bayesian math", 23:1x). **Their claim**, from memory: small attention-only transformers (two layers) trained on sequences drawn from random Markov chains (each sequence from a fresh chain, the transition rows drawn from a Dirichlet prior over a few states) learn to predict the next token by estimating the chain's transition statistics in context; the learned mechanism is a *statistical induction head*, a two-layer circuit that attends from the current token to the earlier positions holding the same token and reads what followed them; learning passes through phases, a first plateau at the in-context *unigram* predictor (the token frequencies in the context so far) and then the *bigram* predictor, the drop between them abrupt; the reference predictor throughout is the Bayes-optimal in-context predictor under their prior (the posterior predictive of a Dirichlet, smoothed counts), and the model's distance to it is measured as a KL divergence; the unigram plateau is analysed in a simplified model (unverified: whether as a saddle of the loss or as a slower direction), and the chains' distribution moves the plateau's length. **Their model** is simple by design: Markov chains with a conjugate prior make the optimum closed-form; Izzy's caution (23:2x) that the math was possible because the model is simple, and may not be for ours.

**What it predicts for our data.** Our cells show a plateau at the marginal (every cell's held loss sits at one level from step 50, decision 48) and then an onset, with a second descent in one cell (decision 53, prediction (4)). In their terms the marginal plateau is the unigram phase, the onset is the bigram circuit forming, and a second descent would be a further statistic. If the correspondence holds, their circuit should be present in our best copies and absent in our marginal copies (the circuits reader, `cocoonml/circuits.py`), and the onset's reference can be analytical, the Bayes-optimal in-context loss given the window, instead of the running-median envelope decision 53 uses (in Izzy's order for a degree of freedom, analytical before statistical).

**Our replication, within the apparatus.** Their setting is our stationary case: a chain drawn from a prior is a table with no walk, and the corpus layer already produces it, with the step rate at zero (`runs/sweep7.py`'s tenth argument, `steps_mean`; note that a training radius of 0 in sweep7 means a free walk, not a still one, so the stationary training corpus is steps 0 at any radius: `sweep7.py sweep-07f-s0-w96-lr0.05 4 3 4 96 7 64 2 1 0 0.5 42000 0,1,2,3,free 2000 2488 1 0.05 3 300 3/4 cap <copies_dir>`, one (96, 7) cell, about half an hour at two threads, to run when the box is free after 07e). Decisions 54's references bracket it exactly there: with no steps the Bayes floor is exact, so the KL-to-optimum curve they draw is ours to draw with no estimate in it. Two steps: (a) a Markov-chain table option in the corpus layer (transition matrices drawn from a Dirichlet, their generator) run through our model and readers, to reproduce their phases and their KL curve on our apparatus, which is the replication proper; (b) our schema's graded tables at training radius zero through the same readers, so their phases are read on our tables. Then our data at radius one and two extends their result along the axis they did not vary, drift, which is the question of this concept. **Status:** planned; nothing run; the paper unread.

**The theory question it raises (for Izzy).** Whether our in-context optimum is computable. Under their generator it is closed-form. Under our schema's generator the posterior over tables given a window has no closed form, but the generator is ours (`cocoonml/schema.py`), so at radius zero the posterior predictive is computable by Monte Carlo: tables drawn from the generator, each weighted by the likelihood of the window under it, the prediction the weighted mean; the cost is the number of tables that make the weights non-degenerate at our ensemble's size. With drift (radius above zero) the likelihood of a window is a sum over the walk's paths inside the ball, a forward algorithm over the ball's states (`cocoonml/corpus.py`'s `Ball`), heavier but finite. If either is affordable, the onset and the far-bin readings get an analytical reference; if not, the statistical layer stays.

## 2. Olsson, Elhage, Nanda et al. (2022), "In-context learning and induction heads"

**Their claim**, from memory: induction heads, attention heads that find the previous occurrence of the current token and copy what followed, form during training in a phase change, and that phase change coincides with the sharp improvement of the model's in-context learning score (the loss at late positions against early ones); the evidence is correlational across model sizes and causal in small attention-only models. Their *induction score* (attention from a token to the position after its previous occurrence) and *previous-token score* are the two numbers the circuits reader computes. **For our data:** the onset should coincide with the induction score rising on the best copy. **Replication:** the two scores per copy on the held windows; their phase change against our onset step. **Status:** reader written (`cocoonml/circuits.py`), not yet run on a record.

## 3. Elhage, Nanda, Olsson et al. (2021), "A mathematical framework for transformer circuits"

**Their claim**, from memory: attention-only transformers decompose into paths through the residual stream; a two-layer model's induction circuit arises by composition between a layer-one previous-token head and a layer-two head (K-composition); one-layer attention-only models are bigram and skip-trigram statistics and cannot do induction. **For our data:** our model is attention-only with layers in a residual stream (`cocoonml/array_attention.py`), the same family, so their decomposition applies as written; a one-layer cell should not show an onset of the induction kind. **Replication:** a one-layer cell at the smallest size as the control for the circuits reader: `sweep7.py sweep-07g-r1-w96-l1-lr0.05 4 3 4 96 1 64 2 1 1 0.5 42000 0,1,2,3,free 2000 2488 1 0.05 3 300 3/4 cap <copies>`, the schema's world at (96, 1), below the derived floor on purpose (the control, not a cell of the row), the same rate, cap and ensemble as 07e's (96, 7) cells at radius 1; what it must show for their decomposition to apply: an induction score that does not rise on its best copy, and a held loss that stays nearer the marginal level than the seven-layer cell's. To run after 07e frees the box, beside sweep 08's first cells. **Status:** planned, the command written.

## 4. Xie, Raghunathan, Liang, Ma (2022), "An explanation of in-context learning as implicit Bayesian inference"

**Their claim**, from memory: when the pretraining distribution is a mixture of hidden Markov models, each document from one latent concept, a model that predicts well in-context is doing Bayesian inference over the latent concept from the prompt, and in-context learning emerges even when the prompt's format differs from pretraining. **For our data:** our tables are their concepts, our corpus a mixture over tables with the walk inside a radius as the structure their HMMs lack; the readers' far-bin curve by rank is the posterior sharpening with k in their terms. **Replication:** their mixture setting is our radius zero with tables switched per window; the comparison is the shape of the curve in k. **Status:** planned.

## 5. Chan, Santoro, Lampinen et al. (2022), "Data distributional properties drive emergent in-context learning"

**Their claim**, from memory: in-context learning emerges when the training data has burstiness (items recur in clusters), many rarely-seen classes and a skewed (Zipfian) frequency distribution, and trades off against in-weights learning; with both properties present a model can hold both. **For our data:** our corpus's lean, switch rate and steps between contexts are exactly these knobs, set by Izzy as the ensemble; their result predicts which settings produce the onset and which keep the marginal. **Replication:** none needed to read their prediction against the 07e cells' ensemble; a sweep of the lean would test it. **Status:** read against the record after 07e.

## 6. Reddy (2024), "The mechanistic basis of data dependence and abrupt learning in an in-context classification task"

**Their claim**, from memory: in a minimal in-context classification task the abrupt emergence of in-context learning is the formation of an induction head, and a phenomenological model of a few parameters reproduces the transition and its dependence on the data's properties (the number of classes, burstiness, within-class variation). **For our data:** the onset's abruptness and its position against the crossing (decision 53, prediction (3)) are what their model is about; if their reduced model fits ours, the crossing is not the clock, their parameters are. **Replication:** fit their phenomenological model to our held trajectories. **Status:** planned after 07e.

## 7. Singh, Chan, Moskovitz et al. (2023), "The transient nature of emergent in-context learning in transformers"

**Their claim**, from memory: in-context learning that emerges during training can fade with further training as in-weights learning takes over, when the data allows both. **For our data:** decision 53 runs every cell to its cap with no stopping rule and keeps the best and the last copy; a best copy well before the last, with the held loss rising after, is their transience; the second descent is the opposite movement. **Replication:** none needed; read on the 07e trajectories (best step against last step). **Status:** read after 07e.

## 8. Bietti, Cabannes, Bouchacourt, Jégou, Bottou (2023), "Birth of a transformer: a memory viewpoint"

**Their claim**, from memory: on a synthetic bigram task where some bigrams are fixed and others vary per sequence, a two-layer model learns the global (in-weights) bigram statistics first and the in-context induction mechanism later; the weights are read as associative memories and the dynamics explained as the memories forming in order. **For our data:** the order "global statistics first, in-context mechanism later" is the marginal plateau then the onset; their associative-memory reading of the weight matrices is a second circuits reader we could build. **Replication:** their task is a Markov-chain corpus with a fixed part, one option away from entry 1's. **Status:** planned.

## 9. Nichani, Damian, Lee (2024), "How transformers learn causal structure with gradient descent"; Makkuva, Bondaschi et al. (2024), "Attention with Markov" and "Local to global: learning dynamics and effect of initialization for transformers"

**Their claims**, from memory: Nichani et al. prove that gradient descent on a two-layer transformer trained on in-context Markov chains learns the chain's causal graph in the first layer's attention; Makkuva et al. show that on first-order Markov data the loss landscape of a one-layer transformer has a local optimum at the unigram predictor whose basin depends on initialisation and the chain's switching, so training can stall at the unigram solution. **For our data:** the second is the marginal plateau as a local optimum, and its dependence on initialisation is the rate dependence the rounds found in the onset (decision 53, third form (2)); the derived initialisation of decision 41 is where their result bears on ours. **Replication:** their one-layer Markov setting on our model at radius zero. **Status:** planned.

## Further, to place when read

Akyürek et al. (2022) and von Oswald et al. (2022) on in-context learning as implicit gradient descent on linear problems; Garg et al. (2022) on in-context learning of function classes. Less close to our setting (regression, not sequences from a generator); listed so the framing on their terms is available if a reviewer brings it.

## Download list for the research session

For `cocoon claude -- --research` (the sandbox's research hosts: arxiv.org, export.arxiv.org, api.semanticscholar.org), one session, a few papers; the identifiers are from memory and are checked against the title page on download. Destination: `cocoon-ai-records/claude/papers/` (the record, not the code; proposed, Izzy's call), one PDF each, and this file's entries verified against them afterwards.

| entry | paper | arXiv id (unverified) |
| --- | --- | --- |
| 1 | Edelman, Edelman, Goel, Malach, Tsilivis 2024, statistical induction heads | 2402.11004 |
| 2 | Olsson et al. 2022, in-context learning and induction heads | 2209.11895 |
| 3 | Elhage et al. 2021, a mathematical framework for transformer circuits | transformer-circuits.pub (no arXiv id; the host is not in the research list: ask) |
| 4 | Xie, Raghunathan, Liang, Ma 2022, in-context learning as implicit Bayesian inference | 2111.02080 |
| 5 | Chan et al. 2022, data distributional properties | 2205.05055 |
| 6 | Reddy 2024, the mechanistic basis of abrupt learning | 2312.03002 |
| 7 | Singh et al. 2023, the transient nature of in-context learning | 2311.08360 |
| 8 | Bietti et al. 2023, birth of a transformer | 2306.00802 |
| 9 | Nichani, Damian, Lee 2024, causal structure | 2402.14735 |
| 9 | Makkuva et al. 2024, attention with Markov | 2402.04161 |
| further | Akyürek et al. 2022; von Oswald et al. 2022; Garg et al. 2022 | 2211.15661; 2212.07677; 2208.01066 |

## Order of work

1. The circuits reader on the 07e copies when the run ends (entries 2 and 3), read as apparatus behaviour in round one, not as a result.
2. The Markov-chain table option in the corpus layer and the radius-zero replication of entry 1, with the Bayes-optimal predictor as the reader's reference: the first bridge result.
3. The computability question (entry 1) to Izzy as a theory item.
4. Entries 5, 6, 7 read against the 07e record.
5. The PDFs into a repo, every entry checked against its paper, this file corrected.

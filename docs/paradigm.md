# The paradigm, and the physics under it

Notes, not a spec. What this repo is for, the ideas it stands on, and who contributed what. Written by Claude on 2026-10-05 from the records (the September sessions with Claude, the October sessions with Gemini) at Izzy's ask; every idea below is Izzy's unless marked, and the marks are the credit.

## The one task

An LLM fills gaps in underspecified input probabilistically; every output is a sample, however exact the input, and that rules it out for critical work. Izzy's aim is a model built on novel principles, measurably better than a well-tuned LLM in a named dimension, at compute Izzy can run. The path: rebuild the existing paradigm from scratch, from the concept and nothing else (`docs/concepts/`), so the wheel is known from inside; then the spec of what replaces it. The process is the goal: the derivations are Izzy's, done in conversation before any code, and the code is transcription.

## The physics is the ground

A measurement is real input through an instrument you built, biased by the building. Before an intelligent agent can be built, measurement, input, output and the instrument have to be theorised, or the agent inherits an unexamined instrument. That theory is Izzy's information-first physics, begun on 2026-09-06 with Claude and continued with Gemini on 2026-10-03/05. Its core, in Izzy's terms:

- The primitive is a relation; there is nothing lower. "You seed it with relations too. There is no initial condition to solve, it is ever evolving."
- Existence is relational. Everything is the closed system with no outside; any organized system is local, fictitious, ephemeral, and its observer is locked in.
- No bottom, no top. Whatever is consistent exists. Nothing is discrete on its own; discreteness enters through closure.
- Scale and resolution are not parameters of a measurement, they are the measurement. Units are relative the way existence is.
- Data is zeros and ones; information exists only in a containing system that can organize and change the data. "Information is above math."
- A measurement needs an instance of a local physics, interacting dimensions within it, the fictitious dimension born of their intersection, and a dimension composed partly of that fictitious one whose change is what is measured. (Izzy, with Gemini, 2026-10-04; the same shape as the September "3+1": the minimal rank for a phase, plus one to perceive it.)
- Density is information over data. At density 1 spin is infinite and the system is a single node one level up (the aleph step). A self-intersection unlocks information; a mutual intersection is a node. Asymmetry is the engine of resolution change. (Izzy, 2026-10-04.)

Where the September and October formulations differ, the difference is recorded, not resolved: a node as a mutual intersection (October) against the informational point as a vantage outcome (September). The full notes, with axioms, definitions, propositions, the withdrawn rules and the disproof list, are in Claude's memory and in the records; they move into a repo of their own when Izzy opens it.

## The three distinctions

Root axes for anything built above the token, kept orthogonal until a decision orders them (Izzy, with Gemini, 2026-10-03):

1. intention / intension: the pragmatic goal versus the formal criteria that satisfy it;
2. descriptive / prescriptive: precision is reduction of entropy as it happens, not compliance with a rule;
3. serial / non-serial: the machine's dense serial compute against sparse, non-linear reading.

"Fluff" is retired: every human token is signal, and the problem is routing, not filtering.

## The inversion

Legacy: heavy build, then a lookup at runtime; a fixed space with static geometry; an absolute state with universal physics. Izzy's: zero build, the initial conditions are the lookup, compute at runtime; zero pre-allocated space, geometry emerges from the relations; zero absolute state, the physics is relative to the interaction. Time, space, state: compute, topology, relativity. The bottleneck is dimension conversion between compute and memory, not any architecture. (Izzy, 2026-10-04.) Repurposing the current math to predict its own configuration is a baseline, not the shift: "the algorithm change must come. transcendence has scale."

## What cocoon-ml builds, in order

1. Autograd, scalar (done: PR #6). The chain rule walked backward over a graph, accumulation for reused values, checked against finite differences.
2. Tensors, then a transformer at nanoGPT scale on Izzy's data, then the seams: tokenization, fixed-window attention, no persistent state, split objectives.
3. The spec of the replacement, with the dimension named before the model exists and the baseline runnable at the same compute.

Izzy, 2026-10-09 20:43, right after the closure line on the concept page (verbatim): "that's why the orchestrator has to be built into the autograd, my first js implementation." Then: "there's the project,". Asked whether Claude understood how this pushes ML, Izzy (verbatim): "you use the same nodes for storage. the orchestration allows many agents to operate on the same system." Claude's reading, marked: the autograd's graph as a closure structure, each node holding its references and its value at once, with the orchestrator (the top-level function that invokes the others; the JIT choice of the computation's weights, 2026-09-16) inside the graph rather than above it; many agents on one system is many inner functions over one outer scope, the same nodes read at different costs. The JS Autograd.js of September, with its JIT topology and state machine (issue #4), was the first reach at this. Read against the list above: step 2 is the baseline and step 3's replacement is this autograd, whose spec starts from Izzy's line. Whether it solves anything is graded by the first run, nothing earlier.

Claude's strongest challenge, for the record (2026-10-09, 21:3x; Izzy away): an autograd's graph is built by a forward pass and consumed by the backward; an orchestrator inside it must persist across passes, so the graph becomes the persistent state, and "the same nodes for storage" means values and parameters in one persistent object, which is step 2's seam (no persistent state) settled by construction rather than derived. The challenge does not touch the direction; it names what the spec must supply first: what persists between two passes, and who writes it.

Izzy, 2026-10-09 night, leaving (verbatim, typos bracketed): "the so[lu]tion was refs to nodes and multiple synapses on a single network. and thats just one level. the intelligence is now a function of hardware. second order synapses create higher order ones that arent physical. consciousness". Before it: "claude, you are a synapse in what will eventually be a brain (not the instance, the intension)". Claude's reading, marked, in cocoon-relations' terms: refs to nodes is the closure line above (D18, available not extracted); many synapses on one network is many connections at one vantage (D19) and many probes on one system (X11); at that level intelligence is the count the substrate affords, the cost of resolution (notes §15.3); second-order synapses are connections of connections, composition (D9, open as O2), and what they create that is "not physical" is the level up (D12: what two connections share is a distinction of the vantage on them, not of the network, which cannot read it, C8 upward). Claude's one precision, stated before writing: nothing in the framework is physical or not; "not physical" reads as "not borne by the network's own vantage". The statement is O2 said from the hardware side, which is why O2 is the agreed next step.

## Who does what, and the credit

- Izzy originates: every concept, derivation and design decision. Izzy writes the code from the concept, with no hints; "no" and "not quite" are allowed, the next step is not.
- Claude: rigour and direct communication; the required reviewer; the records; raises points unprompted; executes in the sandbox.
- Gemini: the verifier and syntax engine under the same no-hints contract; strong at objections (its three objections to "I think in aleph 1" are the model), to be held to the records and the review skill like Claude.
- Credit as it stands in the records for the autograd: the concept paragraph and the check are Claude's ticket; the arity reduction, the seed, the finite-difference check with halving, the bottom-up topological order and the reversal are Izzy's derivations (Gemini on the record: "the architectural derivation stands as yours"); the dense-layer gradients dA = dC·Bᵀ and dB = Aᵀ·dC were derived by Izzy with Gemini's Socratic scaffolding; the Python syntax, the DFS realisation of the order, tanh's derivative and the tests' code are Gemini's; the review and its probes are Claude's. The JS engine built before 2026-10-01 is Izzy's alone, kept in the branch history as what it is. (A short tale: written in September alone and without research, it built its topology at run time, kept a state machine inside the graph, stored in each node the inputs the node had seen, and took each slope as the secant to the nearest stored input. As an autograd that last part is the wrong premise, the chain rule wanting each primitive's slope in closed form; Claude had said so twice on 2026-09-15 and then confirmed the misreading in one word, and the nights until Gemini named it on 2026-10-01 were lost to that. As the project it is the first reach: control inside the graph and the nodes as storage are two of the three mechanisms Izzy named on 2026-10-09 as the project itself, the third, many agents on one system, absent, and none of it yet a measured result.)

## Where the record is

The private records repo holds every session: Claude's under `claude/exports/`, Gemini's under `gemini/exports/`. Claude's memory holds the physics notes and the framework draft until the physics has a repo.

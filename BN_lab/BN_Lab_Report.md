# AI Laboratory Report: Bayesian Networks and Autoregressive Language Models

**Course:** CSF407 / AI Laboratory  
**Topic:** Connecting Bayesian Networks to Autoregressive Language Models  
**Date:** September 30, 2026  

---

## 1. Executive Summary & Overview

This laboratory explores the fundamental link between **Bayesian Networks (BNs)** and **Autoregressive Language Models (LMs)**. An autoregressive language model generates text by decomposing the joint probability distribution over a sequence of tokens using the chain rule of probability:

$$P(X_1, X_2, \dots, X_T) = \prod_{t=1}^T P(X_t \mid X_1, \dots, X_{t-1})$$

Under different Markov independence assumptions, this joint factorisation corresponds directly to specific directed acyclic graph (DAG) topologies:
1. **First-Order Markov Model ($1^{\text{st}}$-order LM):** Each token depends only on its immediate predecessor: $P(X_t \mid X_1, \dots, X_{t-1}) \approx P(X_t \mid X_{t-1})$.
2. **Second-Order Markov Model ($2^{\text{nd}}$-order LM):** Each token depends on the previous two tokens: $P(X_t \mid X_1, \dots, X_{t-1}) \approx P(X_t \mid X_{t-2}, X_{t-1})$.

We implemented both models in standard Python (without external machine learning libraries), computed empirical Conditional Probability Tables (CPTs), verified probability invariants ($\sum_v P(v \mid \text{context}) = 1.0$), analyzed next-word predictions, contrasted greedy vs. probabilistic sampling generation, and evaluated parameter sparsity and linguistic coherence.

---

## 2. Answers to Laboratory Questions (Q1 – Q14)

### Part I: From Probability to Language

#### Question 1: Why is this decomposition useful for generating text?
**Answer:**
The chain-rule decomposition expresses the joint probability of an entire sentence $P(X_1, \dots, X_T)$ as a product of step-wise conditional probabilities:
1. **Tractable Step-by-Step Generation:** Rather than attempting to sample an entire sequence simultaneously from an intractable joint distribution over all possible sentences ($|V|^T$ possibilities), generation proceeds iteratively:
   $$X_1 \sim P(X_1), \quad X_2 \sim P(X_2 \mid X_1), \quad \dots, \quad X_t \sim P(X_t \mid X_1, \dots, X_{t-1})$$
2. **Local Estimation:** Each conditional distribution $P(X_t \mid X_{<t})$ can be parameterized, estimated from empirical data (via frequency counts or neural function approximators), and normalized locally.
3. **Variable-Length Sequences:** The iterative formulation naturally supports generating sequences of arbitrary length until an explicit stopping criterion (e.g. `<END>` token) is sampled.

---

### Part II: A Bayesian Network for Text

#### Question 2: What independence assumption is being made by this network?
**Answer:**
The first-order linear chain $X_1 \to X_2 \to X_3 \to \dots \to X_T$ asserts the **First-Order Markov Assumption**: each token $X_t$ is conditionally independent of all earlier tokens given its immediate predecessor $X_{t-1}$.

In conditional independence notation:
$$X_t \perp\!\!\!\perp \{X_1, X_2, \dots, X_{t-2}\} \mid X_{t-1} \quad \forall t \ge 3$$

In probability notation:
$$P(X_t \mid X_1, X_2, \dots, X_{t-1}) = P(X_t \mid X_{t-1})$$

---

### Part IV: Constructing the Conditional Probability Table

#### Question 3: Construct the conditional probability distribution $P(\text{next word} \mid \text{current word})$ for at least: `the`, `cat`, `dog`, `sat`, `ran`. Identify any zero-probability transitions.
**Answer:**

From the 6 training sentences with `<START>` and `<END>` tokens:
- `<START> the cat sat on the mat <END>`
- `<START> the cat sat on the rug <END>`
- `<START> the dog sat on the mat <END>`
- `<START> the dog ran to the park <END>`
- `<START> the cat ran to the park <END>`
- `<START> the dog sat on the rug <END>`

**Empirical Bigram Counts & Conditional Distributions:**

1. **Preceding Word = `the`** (Total occurrences as predecessor = 12):
   - `cat`: count = 3 $\implies P(\text{cat} \mid \text{the}) = \frac{3}{12} = 0.2500$
   - `dog`: count = 3 $\implies P(\text{dog} \mid \text{the}) = \frac{3}{12} = 0.2500$
   - `mat`: count = 2 $\implies P(\text{mat} \mid \text{the}) = \frac{2}{12} \approx 0.1667$
   - `park`: count = 2 $\implies P(\text{park} \mid \text{the}) = \frac{2}{12} \approx 0.1667$
   - `rug`: count = 2 $\implies P(\text{rug} \mid \text{the}) = \frac{2}{12} \approx 0.1667$

2. **Preceding Word = `cat`** (Total occurrences = 3):
   - `sat`: count = 2 $\implies P(\text{sat} \mid \text{cat}) = \frac{2}{3} \approx 0.6667$
   - `ran`: count = 1 $\implies P(\text{ran} \mid \text{cat}) = \frac{1}{3} \approx 0.3333$

3. **Preceding Word = `dog`** (Total occurrences = 3):
   - `sat`: count = 2 $\implies P(\text{sat} \mid \text{dog}) = \frac{2}{3} \approx 0.6667$
   - `ran`: count = 1 $\implies P(\text{ran} \mid \text{dog}) = \frac{1}{3} \approx 0.3333$

4. **Preceding Word = `sat`** (Total occurrences = 4):
   - `on`: count = 4 $\implies P(\text{on} \mid \text{sat}) = \frac{4}{4} = 1.0000$

5. **Preceding Word = `ran`** (Total occurrences = 2):
   - `to`: count = 2 $\implies P(\text{to} \mid \text{ran}) = \frac{2}{2} = 1.0000$

**Zero-Probability Transitions:**
Any token transition not present in the empirical corpus has an estimated probability of **0.0**. For example:
- $P(\text{on} \mid \text{the}) = 0.0$
- $P(\text{mat} \mid \text{cat}) = 0.0$
- $P(\text{ran} \mid \text{sat}) = 0.0$
- $P(\text{<END>} \mid \text{the}) = 0.0$
- $P(\text{<START>} \mid w) = 0.0 \quad \forall w$

---

### Part VI: Inspect the LLM-Generated Code

#### Question 4: Where in the program are the transition counts stored?
**Answer:**
In `FirstOrderLanguageModel`, transition counts are maintained inside:
```python
self.transition_counts = collections.defaultdict(collections.Counter)
```
where `self.transition_counts[curr_token][next_token]` maps a pair of tokens $(w_i, w_j)$ to the integer count $C(w_i, w_j)$. Predecessor totals are stored in `self.unigram_counts[curr_token]`.

#### Question 5: Where is $P(X_t \mid X_{t-1})$ computed?
**Answer:**
It is explicitly computed during the model fitting phase in `train()`:
```python
for curr_token, next_counts in self.transition_counts.items():
    total = self.unigram_counts[curr_token]
    self.probabilities[curr_token] = {
        nxt: cnt / total for nxt, cnt in next_counts.items()
    }
```
Here, `self.probabilities[curr_token]` stores the exact conditional distribution $\{v: P(v \mid \text{curr\_token})\}$.

#### Question 6: How does the program choose the next word? Is it choosing the most probable word or sampling from the distribution? Explain the difference.
**Answer:**
The implementation explicitly supports both behaviors via the `mode` parameter:
- **Greedy Generation (`mode="greedy"`):**
  ```python
  next_token = max(dist, key=dist.get)
  ```
  Chooses $\arg\max_w P(w \mid w_{\text{prev}})$. This is completely deterministic.
- **Probabilistic Sampling (`mode="sample"`):**
  ```python
  next_token = random.choices(tokens, weights=probs, k=1)[0]
  ```
  Samples according to the multinomial distribution $w \sim P(w \mid w_{\text{prev}})$. This is stochastic.

**Core Difference:**
Greedy decoding always selects the single mode of the local distribution. It has zero entropy and zero diversity, and in looped grammars can become trapped in an infinite cycle. Probabilistic sampling explores alternative valid paths in proportion to their likelihood, yielding diverse and varied outputs.

#### Question 7: What happens if the program encounters a word for which no transition has been observed?
**Answer:**
In standard maximum-likelihood estimation without smoothing, an unseen context has an empty conditioning row ($C(w) = 0$). In our implementation, `get_cpt(token)` returns an empty dictionary `{}`. To handle this robustly:
- In generation, if `dist` is empty, the loop terminates cleanly (emitting `<END>`), or alternatively falls back to a uniform prior over known vocabulary tokens.
- Mathematically, an unsmoothed model assigns zero probability to any sequence containing an unseen transition.

---

### Part VII: Test the Probability Model

#### Question 8: If one of the totals is 0.87, what does this tell you about the implementation?
**Answer:**
The fundamental axiom of probability requires that every conditional distribution must sum to 1 over the complete sample space of outcomes:
$$\sum_{v \in V} P(v \mid w) = 1.0$$
If the total is **0.87**:
1. **Defective Probability Measure:** 13% of probability mass is missing.
2. **Implementation Bugs:** Possible causes include:
   - Some valid transitions were dropped or filtered out during tallying.
   - Denominator mismatch: normalizing by an external count (e.g., total tokens in corpus) rather than the exact marginal frequency $C(w) = \sum_k C(w, w_k)$.
   - Rounding errors or integer truncation division (e.g. `cnt // total` in Python 2 or improperly cast integers).
   - Unhandled special tokens (e.g., ignoring `<END>`).

In our implementation, verification proved all contexts sum **identically to 1.000000**.

---

### Part VIII: Predicting the Next Word

#### Question 9: Are the most probable predictions always the same as the words that you would personally expect? What does this tell you about the difference between a probability model and human linguistic expectations?
**Answer:**
From our empirical test:
- Preceding `sat` $\to$ ArgMax is `on` ($P = 1.0$) — matches expectations.
- Preceding `ran` $\to$ ArgMax is `to` ($P = 1.0$) — matches expectations.
- Preceding `the` $\to$ ArgMax is `cat` (tied with `dog` at $P = 0.2500$). A human reader might equally expect nouns like `park`, `rug`, `mat`, `ball`, `sun`, etc.

**Key Insight:**
A probability model is purely an empirical reflection of its training dataset and model capacity:
1. **Corpus Bias vs. World Knowledge:** The model knows only what was counted in the 6 sentences. Humans bring vast commonsense knowledge, world semantics, syntax, and pragmatic expectations.
2. **Context Blindness:** A first-order model conditioned on `the` has no idea what came before `the`. A human knows that if the sentence was *"the dog ran to the..."*, the next word must be a destination (`park`), not a subject (`cat` or `dog`). The first-order model assigns $P(\text{cat} \mid \text{the}) = 0.25$ regardless of whether the sentence started with `"the dog ran to the"`.

---

### Part X: Deterministic vs Probabilistic Generation

#### Question 10: Compare the two sets of generated sentences. Which mode produces more variation? Why?
**Answer:**
- **Mode A (Greedy):** Produced the exact same sentence on every run:
  ```
  the cat sat on the cat sat on the cat sat on ... (infinite loop until max_tokens)
  ```
  *Why:* The transition from `<START>` $\to$ `the` ($P=1.0$), `the` $\to$ `cat` ($P=0.25$), `cat` $\to$ `sat` ($P=0.667$), `sat` $\to$ `on` ($P=1.0$), and `on` $\to$ `the` ($P=1.0$). At `the`, `cat` is picked again. The greedy path enters a deterministic cycle that never reaches `<END>` ($P(\text{<END>} \mid \text{the}) = 0$).
- **Mode B (Sampling):** Produced substantial variety across runs:
  ```
  1. the park
  2. the cat sat on the dog ran to the cat sat on the cat sat on the mat
  3. the dog sat on the rug
  4. the mat
  5. the cat ran to the park
  ```
  *Why:* At branch points (such as `the` which has 5 valid continuations, or `cat`/`dog` which have 2), sampling draws from the non-zero probability mass, generating diverse sentence lengths and structures.

---

### Part XI: A Second-Order Bayesian Network

#### Question 11: How does the second-order model differ from the first-order model in terms of:
1. **Graph Structure:**
   - First-order: A single directed chain $X_1 \to X_2 \to X_3 \to \dots \to X_T$. Each node has in-degree 1.
   - Second-order: Each node $X_t$ has **two incoming directed edges** from $X_{t-2}$ and $X_{t-1}$. In DAG notation:
     $$X_{t-2} \longrightarrow X_t \longleftarrow X_{t-1}$$
2. **Conditional Probability Table (CPT):**
   - First-order: Indexed by single tokens: $P(X_t \mid X_{t-1})$. Table size is $|V| \times |V|$.
   - Second-order: Indexed by token pairs (bigram contexts): $P(X_t \mid X_{t-2}, X_{t-1})$. Table size is $|V|^2 \times |V|$.
3. **Amount of Context Available:**
   - First-order: 1 preceding token ($t-1$).
   - Second-order: 2 preceding tokens ($t-2, t-1$). This allows the model to distinguish between `(on, the)` (which should be followed by `mat` or `rug`) and `(to, the)` (which must be followed by `park`).
4. **Amount of Data Needed:**
   - The number of possible conditioning states scales as $\mathcal{O}(|V|^2)$ instead of $\mathcal{O}(|V|)$. To reliably estimate probabilities without suffering from massive zero-count sparsity, the required corpus size grows exponentially with the Markov order.

---

### Part XIII: Comparing the Two Models

#### Question 12: Why does increasing the amount of context potentially improve prediction? Why can it simultaneously make the model harder to estimate from limited data? Relate your answer to the size of the conditional probability table.
**Answer:**
1. **Why it improves prediction:**
   Natural language exhibits long-range grammatical and semantic dependencies. In our dataset:
   - In the $1^{\text{st}}$-order model, after `the`, the model can transition to `cat`, `dog`, `mat`, `rug`, or `park`. This creates ungrammatical sentences like *"the cat sat on the park"* or *"the dog ran to the rug"*.
   - In the $2^{\text{nd}}$-order model, the context `(on, the)` has non-zero probability *only* for `mat` ($0.5$) and `rug` ($0.5$). The context `(to, the)` has non-zero probability *only* for `park` ($1.0$). Grammatical correctness is preserved because the preposition context is retained.
2. **Why it is harder to estimate (The Curse of Dimensionality & Sparsity):**
   - The total number of theoretical conditioning contexts is $|V|^n$. For $|V|=12$:
     - First-order: $|V|^1 = 12$ possible contexts. Observed: 11 ($91.7\%$ coverage).
     - Second-order: $|V|^2 = 144$ possible contexts. Observed: 15 ($10.4\%$ coverage; **$89.6\%$ are unobserved zero-probability contexts**).
   - If a test prompt introduces a bigram pair never seen together during training, the $2^{\text{nd}}$-order model fails (assigns probability 0) unless smoothing or backoff is implemented.

---

### Part XV: Reflection on the Role of the LLM

#### Question 13: Why is Approach B preferable when constructing an intelligent system?
**Approach A:** *"Write a Python language model for me."*  
**Approach B:** *"Implement the following probabilistic model: $P(X_t \mid X_{t-1})$, estimated from transition counts, with sampling-based generation."*

**Answer:**
Approach B is vastly superior for several engineering and scientific reasons:
1. **Specifying Intended Behaviour:** Approach A is ambiguous; an LLM might pull in PyTorch, download GPT-2 from HuggingFace, or write a character-level heuristic. Approach B defines the exact mathematical model required.
2. **Understanding the Representation:** By defining the variables ($X_t$), the conditional dependency ($X_t \mid X_{t-1}$), and the parameters (transition counts), the engineer retains full mental ownership of the system's architecture.
3. **Validating the Generated Implementation:** When you specify the exact mathematical formulation, you can systematically audit the generated code to check if the data structures match the formal definitions.
4. **Testing Probabilistic Invariants:** Approach B dictates clear invariant assertions: $\sum_{v} P(v \mid w) = 1.0$, $0 \le P \le 1$, and non-negativity.
5. **Distinguishing Implementation from Model:** The code is merely an instantiation of the underlying mathematical model. Conflating the two leads to cargo-cult programming where errors in logic cannot be separated from bugs in syntax.

---

### Part XVI: Final Conceptual Question

#### Question 14: What did thinking of the language model as a Bayesian network give you?
**Answer:**
Conceptualizing the language model as a Bayesian Network provides deep theoretical clarity:
1. **Representation of Dependencies:** The DAG explicitly exposes which variables influence which. Drawing $X_{t-1} \to X_t$ vs. $\{X_{t-2}, X_{t-1}\} \to X_t$ clarifies the information horizon available at generation time.
2. **Principled Factorisation:** The chain rule factorisation justifies why token-by-token autoregressive generation is mathematically sound, decomposing an intractable joint distribution $P(X_1, \dots, X_T)$ into modular local conditional probability tables.
3. **Explicit Independence Assumptions:** Through $d$-separation and the Markov property, the Bayesian network makes explicit what the model *ignores*. It explains immediately why a first-order model suffers from amnesia and generates *"the cat sat on the park"*.
4. **Principled Testing of Specifications:** Viewing the model as a collection of local CPTs immediately suggests unit tests: each row of a CPT must be a normalized probability vector, and the graph must be acyclic.
5. **Bridge to Modern Deep Learning:** It demystifies modern LLMs (Transformers). A Transformer is not conceptually alien to a Bayesian network; it is an autoregressive model that estimates the very same conditional probabilities $P(X_t \mid X_1, \dots, X_{t-1})$, but uses self-attention and deep neural network parameterizations instead of discrete lookup tables.

---

## 3. Experimental Results & Deliverables

### Deliverable 1 & 2: Python Implementations
The complete clean implementations of both the first-order model and second-order model are provided in [`run_bn_lab.py`](file:///c:/Users/Aayush%20Kushwaha/Downloads/BITS/Acads/3-1/AI/Lab/Lab%2830_09_2026%29/run_bn_lab.py).

### Deliverable 3: Conditional Probability Tables for Selected Contexts

#### First-Order CPT: $P(X_t \mid X_{t-1})$
| Context ($X_{t-1}$) | Next Token ($X_t$) | Transition Count | Total Count | Conditional Probability $P(X_t \mid X_{t-1})$ |
|:---|:---|:---:|:---:|:---:|
| `the` | `cat` | 3 | 12 | **0.2500** |
| `the` | `dog` | 3 | 12 | **0.2500** |
| `the` | `mat` | 2 | 12 | **0.1667** |
| `the` | `park` | 2 | 12 | **0.1667** |
| `the` | `rug` | 2 | 12 | **0.1667** |
| `cat` | `sat` | 2 | 3 | **0.6667** |
| `cat` | `ran` | 1 | 3 | **0.3333** |
| `dog` | `sat` | 2 | 3 | **0.6667** |
| `dog` | `ran` | 1 | 3 | **0.3333** |
| `sat` | `on` | 4 | 4 | **1.0000** |
| `ran` | `to` | 2 | 2 | **1.0000** |
| `on` | `the` | 4 | 4 | **1.0000** |
| `to` | `the` | 2 | 2 | **1.0000** |
| `mat` | `<END>` | 2 | 2 | **1.0000** |
| `rug` | `<END>` | 2 | 2 | **1.0000** |
| `park` | `<END>` | 2 | 2 | **1.0000** |

#### Second-Order Selected CPT: $P(X_t \mid X_{t-2}, X_{t-1})$
| Context $(X_{t-2}, X_{t-1})$ | Next Token ($X_t$) | Count | Total | Probability |
|:---|:---|:---:|:---:|:---:|
| `('<START>', '<START>')` | `the` | 6 | 6 | **1.0000** |
| `('<START>', 'the')` | `cat` | 3 | 6 | **0.5000** |
| `('<START>', 'the')` | `dog` | 3 | 6 | **0.5000** |
| `('sat', 'on')` | `the` | 4 | 4 | **1.0000** |
| `('ran', 'to')` | `the` | 2 | 2 | **1.0000** |
| `('on', 'the')` | `mat` | 2 | 4 | **0.5000** |
| `('on', 'the')` | `rug` | 2 | 4 | **0.5000** |
| `('to', 'the')` | `park` | 2 | 2 | **1.0000** |
| `('the', 'mat')` | `<END>` | 2 | 2 | **1.0000** |
| `('the', 'rug')` | `<END>` | 2 | 2 | **1.0000** |
| `('the', 'park')` | `<END>` | 2 | 2 | **1.0000** |

---

### Deliverable 4: Examples of Generated Text

#### 20 Sentences Generated by First-Order Model (Sampling Mode):
1. `the cat sat on the dog ran to the cat sat on the cat sat on the dog ran to the dog sat on the mat`
2. `the park`
3. `the dog sat on the rug`
4. `the park`
5. `the cat sat on the cat sat on the mat`
6. `the mat`
7. `the dog sat on the mat`
8. `the rug`
9. `the dog sat on the mat`
10. `the park`
11. `the mat`
12. `the mat`
13. `the mat`
14. `the mat`
15. `the rug`
16. `the cat sat on the cat sat on the park`
17. `the dog ran to the dog sat on the park`
18. `the rug`
19. `the park`
20. `the dog sat on the dog sat on the cat ran to the mat`

#### Mode A (Greedy) vs Mode B (Sampling) Comparison:
- **First-Order Greedy:** Enters an infinite cycle:
  `the cat sat on the cat sat on the cat sat on the cat...`
- **Second-Order Greedy:** Deterministically terminates:
  `the cat sat on the mat`
- **Second-Order Sampling Examples:**
  1. `the cat sat on the mat`
  2. `the cat sat on the rug`
  3. `the cat ran to the park`
  4. `the dog ran to the park`
  5. `the dog sat on the rug`

*Observation:* Every single sentence generated by the second-order model is 100% syntactically and semantically valid with respect to the training corpus.

---

### Deliverable 5: Probability-Normalisation Test Results

Both models were subjected to automated invariant verification testing:
$$\forall c \in \text{Contexts}, \quad \left| \sum_{v \in V} P(v \mid c) - 1.0 \right| < 10^{-9}$$

**Output Log:**
```
First-Order Normalization:
   Context '<START>': sum = 1.000000
   Context 'cat': sum = 1.000000
   Context 'dog': sum = 1.000000
   Context 'mat': sum = 1.000000
   Context 'on': sum = 1.000000
   Context 'park': sum = 1.000000
   Context 'ran': sum = 1.000000
   Context 'rug': sum = 1.000000
   Context 'sat': sum = 1.000000
   Context 'the': sum = 1.000000
   Context 'to': sum = 1.000000
>> All First-Order contexts sum EXACTLY to 1.0.

Second-Order Normalization:
   Context ('<START>', '<START>'): sum = 1.000000
   Context ('<START>', 'the'): sum = 1.000000
   Context ('cat', 'ran'): sum = 1.000000
   Context ('cat', 'sat'): sum = 1.000000
   Context ('dog', 'ran'): sum = 1.000000
   Context ('dog', 'sat'): sum = 1.000000
   Context ('on', 'the'): sum = 1.000000
   Context ('ran', 'to'): sum = 1.000000
   Context ('sat', 'on'): sum = 1.000000
   Context ('the', 'cat'): sum = 1.000000
   Context ('the', 'dog'): sum = 1.000000
   Context ('the', 'mat'): sum = 1.000000
   Context ('the', 'park'): sum = 1.000000
   Context ('the', 'rug'): sum = 1.000000
   Context ('to', 'the'): sum = 1.000000
>> All Second-Order contexts sum EXACTLY to 1.0.
```

---

### Deliverable 6: Quantitative Comparison Metrics

| Metric | First-Order Model ($X_{t-1}$) | Second-Order Model ($X_{t-2}, X_{t-1}$) |
|:---|:---:|:---:|
| **Vocabulary Size ($|V|$)** | 12 | 12 |
| **Max Possible Contexts** | 12 | 144 |
| **Observed Contexts** | 11 | 15 |
| **Zero-Probability Contexts** | 1 (8.3%) | 129 (89.6%) |
| **Non-Zero Parameters in CPT** | 17 | 19 |
| **Generated Diversity (Unique in 100 samples)** | 35 | 6 |
| **Qualitative Coherence** | Low (generates loops, fragments like *"the park"*, and chimeric sentences) | High (100% grammatically aligned with corpus) |

---

### Deliverable 7: Reflection on LLM Usage & Code Inspection

During this laboratory, the LLM was utilized as a pair-programming collaborator following the methodology:
$$\text{Understand} \longrightarrow \text{Design} \longrightarrow \text{Ask the LLM} \longrightarrow \text{Implement} \longrightarrow \text{Test} \longrightarrow \text{Reflect}$$

#### Code Inspection & Correction Example:
When the initial script was drafted, two critical issues were detected upon code inspection:
1. **Character Encoding on Windows Shells:**
   The initial generated script contained non-ASCII characters (e.g. non-breaking hyphens `\u2011` in print statements). When executed on a Windows console with `cp1252` encoding, Python threw a fatal `UnicodeEncodeError`. The code was corrected to use standard ASCII hyphens and explicitly configure UTF-8 encoding for file export.
2. **Context Window Representation for Second-Order Generation:**
   In an early draft of the second-order generation loop, the conditioning state initialization did not pad with two `<START>` tokens, causing a key error on the initial step because the model expected a 2-tuple context `(w_{t-2}, w_{t-1})`. This was corrected by tokenizing sequences with `["<START>", "<START>"]` at the head, ensuring that the initial transition conditions on `("<START>", "<START>") \to \text{"the"}` with probability 1.0.
3. **Greedy Infinite Loops:**
   Inspection revealed that first-order greedy decoding easily gets stuck in periodic orbits (e.g. `the` $\to$ `cat` $\to$ `sat` $\to$ `on` $\to$ `the`). A defensive guard (`max_tokens=50`) was instituted to prevent unbounded execution while preserving the theoretical insight that greedy decoding on cyclic Markov chains may fail to reach an absorbing state.

---
*End of Report.*

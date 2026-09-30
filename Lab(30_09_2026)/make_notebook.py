import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("""# AI Laboratory: Bayesian Networks and Autoregressive Language Models
**Course:** CSF407 / AI Laboratory  
**Lab Assignment:** BN_lab.pdf  
**Date:** September 30, 2026  

---
## 1. Overview & Mathematical Formulation
An autoregressive language model models the joint probability of a sequence of tokens using the chain rule of probability:
$$P(X_1, X_2, \dots, X_T) = \prod_{t=1}^T P(X_t \mid X_1, \dots, X_{t-1})$$

In this notebook:
1. **First-Order Markov Model:** $P(X_t \mid X_{<t}) \approx P(X_t \mid X_{t-1})$ (Graph: $X_{t-1} \to X_t$)
2. **Second-Order Markov Model:** $P(X_t \mid X_{<t}) \approx P(X_t \mid X_{t-2}, X_{t-1})$ (Graph: $X_{t-2} \to X_t \leftarrow X_{t-1}$)
"""))

# Dataset
cells.append(nbf.v4.new_markdown_cell("""## 2. Dataset Preparation
We tokenize the 6 sentences and attach special boundary tokens `⟨START⟩` and `⟨END⟩`."""))

dataset_code = """import random
import collections
from typing import List, Tuple, Dict, Set

# Training corpus
RAW_SENTENCES = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]

def tokenize_first_order(sentence: str) -> List[str]:
    return ["<START>"] + sentence.lower().split() + ["<END>"]

def tokenize_second_order(sentence: str) -> List[str]:
    return ["<START>", "<START>"] + sentence.lower().split() + ["<END>"]

print("Example tokenized (1st order):", tokenize_first_order(RAW_SENTENCES[0]))
print("Example tokenized (2nd order):", tokenize_second_order(RAW_SENTENCES[0]))
"""
cells.append(nbf.v4.new_code_cell(dataset_code))

# First-Order Model
cells.append(nbf.v4.new_markdown_cell("""## 3. First-Order Autoregressive Language Model
Estimates transition probabilities:
$$P(w_j \mid w_i) = \\frac{C(w_i, w_j)}{\\sum_k C(w_i, w_k)}$$"""))

fo_code = """class FirstOrderLanguageModel:
    def __init__(self, raw_sentences: List[str]):
        self.sentences = [tokenize_first_order(s) for s in raw_sentences]
        self.vocab: Set[str] = set()
        self.unigram_counts = collections.Counter()
        self.transition_counts = collections.defaultdict(collections.Counter)
        self.probabilities: Dict[str, Dict[str, float]] = {}
        self.train()

    def train(self):
        for sentence in self.sentences:
            for token in sentence:
                self.vocab.add(token)
            for i in range(len(sentence) - 1):
                curr_token = sentence[i]
                next_token = sentence[i + 1]
                self.unigram_counts[curr_token] += 1
                self.transition_counts[curr_token][next_token] += 1

        for curr_token, next_counts in self.transition_counts.items():
            total = self.unigram_counts[curr_token]
            self.probabilities[curr_token] = {
                nxt: cnt / total for nxt, cnt in next_counts.items()
            }

    def get_cpt(self, token: str) -> Dict[str, float]:
        return self.probabilities.get(token, {})

    def predict_next(self, token: str) -> Tuple[str, float]:
        dist = self.get_cpt(token)
        if not dist:
            return ("<END>", 0.0)
        best_token = max(dist, key=dist.get)
        return best_token, dist[best_token]

    def test_normalization(self) -> Dict[str, float]:
        return {w: sum(dist.values()) for w, dist in self.probabilities.items()}

    def generate(self, mode: str = "sample", max_tokens: int = 50) -> str:
        current_token = "<START>"
        generated = []
        steps = 0
        while current_token != "<END>" and steps < max_tokens:
            dist = self.get_cpt(current_token)
            if not dist:
                break
            if mode == "greedy":
                next_token = max(dist, key=dist.get)
            elif mode == "sample":
                tokens = list(dist.keys())
                probs = list(dist.values())
                next_token = random.choices(tokens, weights=probs, k=1)[0]
            else:
                raise ValueError("Mode must be 'greedy' or 'sample'")

            if next_token != "<END>":
                generated.append(next_token)
            current_token = next_token
            steps += 1
        return " ".join(generated)

fo_lm = FirstOrderLanguageModel(RAW_SENTENCES)
print("Vocabulary:", sorted(fo_lm.vocab))
"""
cells.append(nbf.v4.new_code_cell(fo_code))

# CPT verification
cells.append(nbf.v4.new_markdown_cell("""## 4. Conditional Probability Tables & Normalisation Check"""))
cpt_code = """selected_words = ["the", "cat", "dog", "sat", "ran"]
print("=== First-Order Conditional Probability Distributions ===")
for w in selected_words:
    cpt = fo_lm.get_cpt(w)
    print(f"P(X_t | X_{{t-1}} = '{w}'):")
    for nxt, p in sorted(cpt.items()):
        cnt = fo_lm.transition_counts[w][nxt]
        denom = fo_lm.unigram_counts[w]
        print(f"   -> '{nxt}': {p:.4f} (count: {cnt}/{denom})")
    print()

print("=== Invariant Normalisation Test ===")
fo_sums = fo_lm.test_normalization()
for w, s in sorted(fo_sums.items()):
    print(f"Sum P(v | '{w}') = {s:.6f}")
assert all(abs(s - 1.0) < 1e-9 for s in fo_sums.values()), "Normalisation failed!"
print(">> All first-order probability distributions sum to exactly 1.0!")
"""
cells.append(nbf.v4.new_code_cell(cpt_code))

# Next word prediction
cells.append(nbf.v4.new_markdown_cell("""## 5. Next-Word Prediction (Argmax vs Human Expectation)"""))
pred_code = """print("=== Argmax Next-Word Prediction ===")
for w in selected_words:
    best_word, prob = fo_lm.predict_next(w)
    print(f"Preceding: '{w:<4}' -> Predicted next: '{best_word}' (P = {prob:.4f})")
"""
cells.append(nbf.v4.new_code_cell(pred_code))

# Text generation 20 sentences
cells.append(nbf.v4.new_markdown_cell("""## 6. Text Generation (20 Sentences & Greedy vs Sampling)"""))
gen_code = """random.seed(42)
print("=== 20 Sentences Generated by First-Order Model (Sampling) ===")
for i in range(1, 21):
    print(f"{i:2d}. {fo_lm.generate(mode='sample')}")

print("\\n=== Greedy (Mode A) vs Sampling (Mode B) ===")
print("Greedy:")
for i in range(3):
    print(f"  {i+1}: {fo_lm.generate(mode='greedy')}")
print("Sampling:")
for i in range(3):
    print(f"  {i+1}: {fo_lm.generate(mode='sample')}")
"""
cells.append(nbf.v4.new_code_cell(gen_code))

# Second-Order Model
cells.append(nbf.v4.new_markdown_cell("""## 7. Second-Order Autoregressive Language Model
Estimates:
$$P(X_t \mid X_{t-2}, X_{t-1}) = \\frac{C(X_{t-2}, X_{t-1}, X_t)}{C(X_{t-2}, X_{t-1})}$$"""))

so_code = """class SecondOrderLanguageModel:
    def __init__(self, raw_sentences: List[str]):
        self.sentences = [tokenize_second_order(s) for s in raw_sentences]
        self.vocab: Set[str] = set()
        self.context_counts = collections.Counter()
        self.transition_counts = collections.defaultdict(collections.Counter)
        self.probabilities: Dict[Tuple[str, str], Dict[str, float]] = {}
        self.train()

    def train(self):
        for sentence in self.sentences:
            for token in sentence:
                self.vocab.add(token)
            for i in range(len(sentence) - 2):
                ctx = (sentence[i], sentence[i + 1])
                nxt = sentence[i + 2]
                self.context_counts[ctx] += 1
                self.transition_counts[ctx][nxt] += 1

        for ctx, next_counts in self.transition_counts.items():
            total = self.context_counts[ctx]
            self.probabilities[ctx] = {
                nxt: cnt / total for nxt, cnt in next_counts.items()
            }

    def get_cpt(self, ctx: Tuple[str, str]) -> Dict[str, float]:
        return self.probabilities.get(ctx, {})

    def predict_next(self, ctx: Tuple[str, str]) -> Tuple[str, float]:
        dist = self.get_cpt(ctx)
        if not dist:
            return ("<END>", 0.0)
        best_token = max(dist, key=dist.get)
        return best_token, dist[best_token]

    def test_normalization(self) -> Dict[Tuple[str, str], float]:
        return {ctx: sum(dist.values()) for ctx, dist in self.probabilities.items()}

    def generate(self, mode: str = "sample", max_tokens: int = 50) -> str:
        ctx = ("<START>", "<START>")
        generated = []
        steps = 0
        while steps < max_tokens:
            dist = self.get_cpt(ctx)
            if not dist:
                break
            if mode == "greedy":
                next_token = max(dist, key=dist.get)
            elif mode == "sample":
                tokens = list(dist.keys())
                probs = list(dist.values())
                next_token = random.choices(tokens, weights=probs, k=1)[0]
            else:
                raise ValueError("Mode must be 'greedy' or 'sample'")

            if next_token == "<END>":
                break
            generated.append(next_token)
            ctx = (ctx[1], next_token)
            steps += 1
        return " ".join(generated)

so_lm = SecondOrderLanguageModel(RAW_SENTENCES)
print("Second-Order Normalisation check:")
so_sums = so_lm.test_normalization()
assert all(abs(s - 1.0) < 1e-9 for s in so_sums.values()), "Second order normalisation failed!"
print(f"Verified {len(so_sums)} contexts sum exactly to 1.0!")
"""
cells.append(nbf.v4.new_code_cell(so_code))

# Second order generation and comparison
cells.append(nbf.v4.new_markdown_cell("""## 8. Model Comparison: 1st-Order vs 2nd-Order"""))
comp_code = """print("=== Second-Order Generation Examples ===")
print("Greedy:")
for i in range(3):
    print(f"  {i+1}: {so_lm.generate(mode='greedy')}")
print("Sampling:")
for i in range(5):
    print(f"  {i+1}: {so_lm.generate(mode='sample')}")

vocab_size = len(fo_lm.vocab)
fo_nonzeros = sum(len(d) for d in fo_lm.probabilities.values())
so_nonzeros = sum(len(d) for d in so_lm.probabilities.values())

print("\\n=== Quantitative Model Comparison ===")
print(f"Vocabulary Size: {vocab_size}")
print(f"First-Order  - Parameters: {fo_nonzeros}, Zero-Contexts: {vocab_size - len(fo_lm.probabilities)}/{vocab_size}")
print(f"Second-Order - Parameters: {so_nonzeros}, Zero-Contexts: {vocab_size**2 - len(so_lm.probabilities)}/{vocab_size**2}")
"""
cells.append(nbf.v4.new_code_cell(comp_code))

# Summary
cells.append(nbf.v4.new_markdown_cell("""## 9. Summary & Conclusions
- **Chain Rule:** Foundation of modern autoregressive sequence generation.
- **Markov Assumption:** Higher-order context preserves grammatical coherence (e.g. `(on, the)` $\\to$ `mat/rug` vs `(to, the)` $\\to$ `park`), but suffers from quadratic parameter growth and extreme sparsity.
- **Deep Learning Connection:** Modern Transformers model the same conditional distribution $P(X_t \\mid X_{<t})$, but replace discrete sparse CPTs with continuous dense neural representations.
"""))

nb.cells = cells

target_nb = r"c:\Users\Aayush Kushwaha\Downloads\BITS\Acads\3-1\AI\Lab\Lab(30_09_2026)\BN_lab_notebook.ipynb"
with open(target_nb, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook successfully written to {target_nb}")

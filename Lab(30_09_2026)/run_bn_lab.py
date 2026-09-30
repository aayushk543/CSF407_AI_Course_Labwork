import os
import random
import collections
from typing import List, Tuple, Dict, Set

# Set random seed for reproducibility
random.seed(42)

# Training dataset as given in Part III
RAW_SENTENCES = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]

def tokenize_first_order(sentence: str) -> List[str]:
    """Tokenize sentence and add <START> and <END>."""
    return ["<START>"] + sentence.lower().split() + ["<END>"]

def tokenize_second_order(sentence: str) -> List[str]:
    """Tokenize sentence and add two <START> tokens and one <END>."""
    return ["<START>", "<START>"] + sentence.lower().split() + ["<END>"]

# -----------------------------------------------------------------------------
# Part IV & V: First-Order Autoregressive Language Model
# -----------------------------------------------------------------------------
class FirstOrderLanguageModel:
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

        # Compute conditional probabilities P(next | curr)
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
        """Verify sum_v P(v | w) == 1.0 for each context w."""
        sums = {}
        for w, dist in self.probabilities.items():
            sums[w] = sum(dist.values())
        return sums

    def generate(self, mode: str = "sample", max_tokens: int = 50) -> str:
        """
        Generate sentence:
        Mode A ('greedy'): argmax_w P(w | prev)
        Mode B ('sample'): w ~ P(w | prev)
        """
        current_token = "<START>"
        generated = []
        steps = 0
        while current_token != "<END>" and steps < max_tokens:
            dist = self.get_cpt(current_token)
            if not dist:
                break
            if mode == "greedy":
                # Deterministic argmax
                next_token = max(dist, key=dist.get)
            elif mode == "sample":
                # Probabilistic sampling
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


# -----------------------------------------------------------------------------
# Part XI & XII: Second-Order Autoregressive Language Model
# -----------------------------------------------------------------------------
class SecondOrderLanguageModel:
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

        # Compute conditional probabilities P(X_t | X_{t-2}, X_{t-1})
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
        """Verify sum_v P(v | u, w) == 1.0 for each context (u, w)."""
        sums = {}
        for ctx, dist in self.probabilities.items():
            sums[ctx] = sum(dist.values())
        return sums

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


# -----------------------------------------------------------------------------
# Execution & Deliverables Verification
# -----------------------------------------------------------------------------
def run_lab():
    print("=================================================================")
    print("           AI LABORATORY: BAYESIAN NETWORKS & AUTOREGRESSIVE LM")
    print("=================================================================\n")

    fo_lm = FirstOrderLanguageModel(RAW_SENTENCES)
    so_lm = SecondOrderLanguageModel(RAW_SENTENCES)

    # 1. First-Order CPT for selected words
    print("--- PART IV: First-Order Conditional Probability Distributions ---")
    selected_words = ["the", "cat", "dog", "sat", "ran"]
    for w in selected_words:
        cpt = fo_lm.get_cpt(w)
        print(f"P(X_t | X_{{t-1}} = '{w}'):")
        for nxt, p in sorted(cpt.items()):
            count = fo_lm.transition_counts[w][nxt]
            denom = fo_lm.unigram_counts[w]
            print(f"   -> '{nxt}': {p:.4f} (count: {count}/{denom})")
        print()

    # 2. Probability Normalization Tests
    print("--- PART VII: Probability Normalization Verification ---")
    fo_sums = fo_lm.test_normalization()
    print("First-Order Normalization (Sum of P(v | w)):")
    for w, s in sorted(fo_sums.items()):
        print(f"   Context '{w}': sum = {s:.6f}")
    assert all(abs(s - 1.0) < 1e-9 for s in fo_sums.values()), "First order normalisation failed!"
    print(">> All First-Order contexts sum EXACTLY to 1.0.\n")

    so_sums = so_lm.test_normalization()
    print("Second-Order Normalization (Sum of P(v | u, w)):")
    for ctx, s in sorted(so_sums.items()):
        print(f"   Context {ctx}: sum = {s:.6f}")
    assert all(abs(s - 1.0) < 1e-9 for s in so_sums.values()), "Second order normalisation failed!"
    print(">> All Second-Order contexts sum EXACTLY to 1.0.\n")

    # 3. Next-Word Prediction (Argmax) for 5 contexts
    print("--- PART VIII: Next-Word Prediction (ArgMax) ---")
    test_contexts = ["the", "cat", "dog", "sat", "ran"]
    for ctx in test_contexts:
        best_word, prob = fo_lm.predict_next(ctx)
        cpt = fo_lm.get_cpt(ctx)
        print(f"Preceding word: '{ctx}'")
        print(f"   Full distribution: {dict(sorted(cpt.items()))}")
        print(f"   ArgMax next word: '{best_word}' (Probability: {prob:.4f})")
    print()

    # 4. Text Generation: 20 sentences using sampling
    print("--- PART IX: Generating 20 Sentences (First-Order Sampling) ---")
    fo_sampled_20 = [fo_lm.generate(mode="sample") for _ in range(20)]
    for i, sent in enumerate(fo_sampled_20, 1):
        print(f"{i:2d}. {sent}")
    print()

    # 5. Greedy vs Sampling Comparison (5 sentences each)
    print("--- PART X: Mode A (Greedy) vs Mode B (Sampling) Comparison ---")
    print("First-Order Model:")
    print("  Greedy (Mode A):")
    for i in range(5):
        print(f"    {i+1}: {fo_lm.generate(mode='greedy')}")
    print("  Sampling (Mode B):")
    for i in range(5):
        print(f"    {i+1}: {fo_lm.generate(mode='sample')}")
    print()

    print("Second-Order Model:")
    print("  Greedy (Mode A):")
    for i in range(5):
        print(f"    {i+1}: {so_lm.generate(mode='greedy')}")
    print("  Sampling (Mode B):")
    for i in range(5):
        print(f"    {i+1}: {so_lm.generate(mode='sample')}")
    print()

    # 6. Part XIII: Model Comparison Metrics
    print("--- PART XIII: Comparing First-Order and Second-Order Models ---")
    # Number of parameters (number of non-zero conditional probabilities)
    fo_nonzeros = sum(len(d) for d in fo_lm.probabilities.values())
    so_nonzeros = sum(len(d) for d in so_lm.probabilities.values())

    # Total possible contexts
    vocab_size = len(fo_lm.vocab)
    fo_possible_contexts = vocab_size
    so_possible_contexts = vocab_size * vocab_size

    fo_observed_contexts = len(fo_lm.probabilities)
    so_observed_contexts = len(so_lm.probabilities)

    fo_zero_contexts = fo_possible_contexts - fo_observed_contexts
    so_zero_contexts = so_possible_contexts - so_observed_contexts

    # Generation diversity (distinct sentences out of 100 samples)
    fo_samples_100 = [fo_lm.generate(mode="sample") for _ in range(100)]
    so_samples_100 = [so_lm.generate(mode="sample") for _ in range(100)]
    fo_unique = len(set(fo_samples_100))
    so_unique = len(set(so_samples_100))

    print(f"Vocabulary Size: {vocab_size} tokens (including <START>, <END>)")
    print(f"First-Order Model:")
    print(f"   - Observed contexts: {fo_observed_contexts}")
    print(f"   - Distinct non-zero parameters: {fo_nonzeros}")
    print(f"   - Zero-probability contexts: {fo_zero_contexts} / {fo_possible_contexts} ({fo_zero_contexts/fo_possible_contexts*100:.1f}%)")
    print(f"   - Unique sentences generated in 100 samples: {fo_unique}")
    print(f"Second-Order Model:")
    print(f"   - Observed contexts: {so_observed_contexts}")
    print(f"   - Distinct non-zero parameters: {so_nonzeros}")
    print(f"   - Zero-probability contexts: {so_zero_contexts} / {so_possible_contexts} ({so_zero_contexts/so_possible_contexts*100:.1f}%)")
    print(f"   - Unique sentences generated in 100 samples: {so_unique}")
    print()

    # Save all generated 20 sentences and results to a text file for submission
    output_path = r"c:\Users\Aayush Kushwaha\Downloads\BITS\Acads\3-1\AI\Lab\Lab(30_09_2026)\lab_execution_output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("AI LABORATORY: BAYESIAN NETWORKS & AUTOREGRESSIVE LM OUTPUT\n\n")
        f.write("20 GENERATED SENTENCES (First-Order Sampling):\n")
        for i, s in enumerate(fo_sampled_20, 1):
            f.write(f"{i}. {s}\n")
        f.write("\nFIRST-ORDER GREEDY:\n")
        for i in range(5):
            f.write(f"{i+1}. {fo_lm.generate(mode='greedy')}\n")
        f.write("\nSECOND-ORDER GREEDY:\n")
        for i in range(5):
            f.write(f"{i+1}. {so_lm.generate(mode='greedy')}\n")
        f.write("\nSECOND-ORDER SAMPLING (10 examples):\n")
        for i in range(10):
            f.write(f"{i+1}. {so_lm.generate(mode='sample')}\n")

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    run_lab()

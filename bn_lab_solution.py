import random
import collections
from typing import List, Tuple, Dict

# ---------------------------
# Data preparation
# ---------------------------
raw_sentences = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]

def tokenize(sentence: str) -> List[str]:
    """Convert a raw sentence to a list of tokens with START/END markers."""
    tokens = sentence.lower().split()
    return ["<START>"] + tokens + ["<END>"]

# Tokenised dataset
sentences = [tokenize(s) for s in raw_sentences]

# ---------------------------
# First‑order (bigram) model
# ---------------------------
class FirstOrderModel:
    def __init__(self, sentences: List[List[str]]):
        self.unigram_counts: collections.Counter = collections.Counter()
        self.bigram_counts: Dict[Tuple[str, str], int] = collections.Counter()
        self.probabilities: Dict[str, Dict[str, float]] = {}
        self._train(sentences)

    def _train(self, sentences: List[List[str]]):
        for sent in sentences:
            for i in range(len(sent) - 1):
                prev, nxt = sent[i], sent[i + 1]
                self.unigram_counts[prev] += 1
                self.bigram_counts[(prev, nxt)] += 1
        # Compute conditional probabilities P(next|prev)
        for (prev, nxt), cnt in self.bigram_counts.items():
            self.probabilities.setdefault(prev, {})[nxt] = cnt / self.unigram_counts[prev]

    def display_cpt(self, prev_word: str):
        """Print the conditional distribution P(next|prev_word)."""
        if prev_word not in self.probabilities:
            print(f"No outgoing transitions from '{prev_word}'.")
            return
        print(f"P(next | {prev_word})")
        for nxt, prob in sorted(self.probabilities[prev_word].items()):
            print(f"  {nxt:>6}: {prob:.3f}")

    def next_word_distribution(self, prev_word: str) -> Dict[str, float]:
        return self.probabilities.get(prev_word, {})

    def sample_next(self, prev_word: str) -> str:
        dist = self.next_word_distribution(prev_word)
        if not dist:
            # unseen context – fall back to uniform over vocabulary
            vocab = list(self.unigram_counts.keys())
            return random.choice(vocab)
        words, probs = zip(*dist.items())
        return random.choices(words, weights=probs, k=1)[0]

    def generate_sentence(self, mode: str = "sample") -> str:
        """Generate a sentence.
        mode = "sample"  → sample from the distribution
        mode = "greedy" → choose the most probable next token.
        """
        token = "<START>"
        generated = []
        while token != "<END>":
            if mode == "greedy":
                # choose argmax
                dist = self.next_word_distribution(token)
                if not dist:
                    token = "<END>"
                    break
                token = max(dist, key=dist.get)
            else:
                token = self.sample_next(token)
            if token != "<END>":
                generated.append(token)
        return " ".join(generated)

    def test_normalisation(self, tolerance: float = 1e-6) -> List[Tuple[str, float]]:
        """Return a list of (prev_word, total_probability) for each conditioning context.
        Useful for checking that Σ_w P(w|prev) ≈ 1.
        """
        results = []
        for prev, nxt_dict in self.probabilities.items():
            total = sum(nxt_dict.values())
            results.append((prev, total))
        return results

# ---------------------------
# Second‑order (trigram) model
# ---------------------------
class SecondOrderModel:
    def __init__(self, sentences: List[List[str]]):
        self.bigram_counts: collections.Counter = collections.Counter()
        self.trigram_counts: Dict[Tuple[str, str, str], int] = collections.Counter()
        self.probabilities: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._train(sentences)

    def _train(self, sentences: List[List[str]]):
        for sent in sentences:
            # count bigrams for denominator
            for i in range(len(sent) - 1):
                pair = (sent[i], sent[i + 1])
                self.bigram_counts[pair] += 1
            # count trigrams for numerator
            for i in range(len(sent) - 2):
                tri = (sent[i], sent[i + 1], sent[i + 2])
                self.trigram_counts[tri] += 1
        # Build conditional distribution P(z | x, y)
        for (x, y, z), cnt in self.trigram_counts.items():
            denom = self.bigram_counts[(x, y)]
            self.probabilities.setdefault((x, y), {})[z] = cnt / denom

    def display_cpt(self, prev_pair: Tuple[str, str]):
        """Print P(next | prev1, prev2) for a given context."""
        if prev_pair not in self.probabilities:
            print(f"No transitions observed for context {prev_pair}.")
            return
        print(f"P(next | {prev_pair[0]}, {prev_pair[1]})")
        for nxt, prob in sorted(self.probabilities[prev_pair].items()):
            print(f"  {nxt:>6}: {prob:.3f}")

    def next_word_distribution(self, prev1: str, prev2: str) -> Dict[str, float]:
        return self.probabilities.get((prev1, prev2), {})

    def sample_next(self, prev1: str, prev2: str) -> str:
        dist = self.next_word_distribution(prev1, prev2)
        if not dist:
            # fallback to uniform vocab
            vocab = set(w for pair in self.probabilities for w in self.probabilities[pair])
            return random.choice(list(vocab))
        words, probs = zip(*dist.items())
        return random.choices(words, weights=probs, k=1)[0]

    def generate_sentence(self, mode: str = "sample") -> str:
        token1, token2 = "<START>", "<START>"  # we treat start token twice for simplicity
        generated = []
        while True:
            if mode == "greedy":
                dist = self.next_word_distribution(token1, token2)
                if not dist:
                    break
                token = max(dist, key=dist.get)
            else:
                token = self.sample_next(token1, token2)
            if token == "<END>" or token is None:
                break
            generated.append(token)
            token1, token2 = token2, token
        return " ".join(generated)

    def test_normalisation(self) -> List[Tuple[Tuple[str, str], float]]:
        results = []
        for context, nxt_dict in self.probabilities.items():
            total = sum(nxt_dict.values())
            results.append((context, total))
        return results

# ---------------------------
# Utility functions for the lab
# ---------------------------
def print_cpt_first_order(model: FirstOrderModel, words: List[str]):
    for w in words:
        model.display_cpt(w)
        print()

def check_normalisation_first_order(model: FirstOrderModel):
    for prev, total in model.test_normalisation():
        print(f"{prev:>10}: sum = {total:.3f}")

def check_normalisation_second_order(model: SecondOrderModel):
    for ctx, total in model.test_normalisation():
        print(f"{ctx}: sum = {total:.3f}")

if __name__ == "__main__":
    # Build models
    fo = FirstOrderModel(sentences)
    so = SecondOrderModel(sentences)

    # ---- Question 3: CPT for selected words ----
    print("--- First‑order CPTs for selected words ---")
    print_cpt_first_order(fo, ["the", "cat", "dog", "sat", "ran"])

    # ---- Question 8: normalisation test ----
    print("--- Normalisation check (first‑order) ---")
    check_normalisation_first_order(fo)

    # ---- Generate some sentences ----
    print("--- Sampled sentences (first‑order) ---")
    for _ in range(5):
        print(fo.generate_sentence(mode="sample"))
    print("--- Greedy (arg‑max) sentences (first‑order) ---")
    for _ in range(5):
        print(fo.generate_sentence(mode="greedy"))

    # ---- Second‑order examples ----
    print("--- Sampled sentences (second‑order) ---")
    for _ in range(5):
        print(so.generate_sentence(mode="sample"))
    print("--- Greedy sentences (second‑order) ---")
    for _ in range(5):
        print(so.generate_sentence(mode="greedy"))

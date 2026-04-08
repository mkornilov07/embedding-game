import os
import random
from itertools import permutations

from gensim.models import KeyedVectors

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

AVAILABLE_MODELS = {
    "word2vec-google-news-300": "Word2Vec Google News (300d)",
    "glove-wiki-gigaword-50": "GloVe Wikipedia (50d)",
    "glove-wiki-gigaword-100": "GloVe Wikipedia (100d)",
    "glove-wiki-gigaword-200": "GloVe Wikipedia (200d)",
    "glove-wiki-gigaword-300": "GloVe Wikipedia (300d)",
    "glove-twitter-25": "GloVe Twitter (25d)",
    "glove-twitter-50": "GloVe Twitter (50d)",
    "glove-twitter-100": "GloVe Twitter (100d)",
    "glove-twitter-200": "GloVe Twitter (200d)",
    "fasttext-wiki-news-subwords-300": "FastText Wiki News (300d)",
}

DEFAULT_MODEL = "word2vec-google-news-300"
MIN_SIMILARITY = 0.50
MIN_GAP_RATIO = 1.05
MAX_SYNONYM_SIMILARITY = 0.50
MAX_ATTEMPTS_PER_RESULT = 5000
REFINE_ITERATIONS = 10

_cache = {}


def _load_model(model_name):
    if model_name in _cache:
        return _cache[model_name]

    path = os.path.join(DATA_DIR, f"{model_name}.kv")
    model = KeyedVectors.load(path)
    vocab = list(model.index_to_key)
    _cache[model_name] = (model, vocab)
    return model, vocab


def _any_pair_too_similar(model, words):
    """Return True if any two words in the triple are near-synonyms."""
    w1, w2, w3 = words
    return (model.similarity(w1, w2) > MAX_SYNONYM_SIMILARITY
            or model.similarity(w1, w3) > MAX_SYNONYM_SIMILARITY
            or model.similarity(w2, w3) > MAX_SYNONYM_SIMILARITY)


def _refine_triple(model, a, b, c):
    """Iteratively tighten a triple by substituting each word via vector arithmetic.

    Each iteration:
      - Replace A with best match for C - B
      - Replace B with best match for C - A
      - Replace C with best match for A + B
    Stops early if any pair becomes too similar.
    """
    for _ in range(REFINE_ITERATIONS):
        # C - B -> new A
        new_a_candidates = model.most_similar(positive=[c], negative=[b], topn=3)
        new_a = next((w for w, _ in new_a_candidates if w not in (b, c)), None)
        if new_a:
            a = new_a

        # C - A -> new B
        new_b_candidates = model.most_similar(positive=[c], negative=[a], topn=3)
        new_b = next((w for w, _ in new_b_candidates if w not in (a, c)), None)
        if new_b:
            b = new_b

        # A + B -> new C
        new_c_candidates = model.most_similar(positive=[a, b], topn=3)
        new_c = next((w for w, _ in new_c_candidates if w not in (a, b)), None)
        if new_c:
            c = new_c

        if _any_pair_too_similar(model, (a, b, c)):
            return None

    return a, b, c


def _best_arrangement(model, words):
    """Try all 3 arrangements of a triple as A+B=C, return the one with the highest similarity."""
    best_addends, best_sum, best_sim = None, None, -1
    for a, b, c in permutations(words, 3):
        if a > b:
            continue
        top_word, sim = model.most_similar(positive=[a, b], topn=1)[0]
        if top_word == c and sim > best_sim:
            best_addends, best_sum, best_sim = (a, b), c, sim
    return best_addends, best_sum, best_sim


def generate_word_sums(count=5, model_name=None, related_pairs=False):
    """Generate word sums using 3CosMul vector arithmetic.

    If related_pairs is True, the second word is sampled from neighbors of the
    first word (around the 10th most similar) instead of being fully random.
    """
    model, vocab = _load_model(model_name or DEFAULT_MODEL)

    results = []
    seen = set()

    for _ in range(count * MAX_ATTEMPTS_PER_RESULT):
        if len(results) >= count:
            break

        w1 = random.choice(vocab)
        if related_pairs:
            neighbors = model.most_similar(w1, topn=50)
            w2 = random.choice(neighbors[9:])[0]
        else:
            w2 = random.choice(vocab)
            if w2 == w1:
                continue
        top = model.most_similar(positive=[w1, w2], topn=2)

        if any(w in (w1, w2) for w, _ in top):
            continue

        w3, sim1 = top[0]
        sim2 = top[1][1]

        if sim1 < MIN_SIMILARITY or sim1 < sim2 * MIN_GAP_RATIO:
            continue

        if _any_pair_too_similar(model, (w1, w2, w3)):
            continue

        refined = _refine_triple(model, w1, w2, w3)
        if refined is None:
            continue
        w1, w2, w3 = refined

        triple = frozenset({w1, w2, w3})
        if triple in seen:
            continue
        seen.add(triple)

        addends, sum_word, _ = _best_arrangement(model, (w1, w2, w3))
        if addends is None:
            continue

        results.append({
            "type": "word_sum",
            "addend1": addends[0],
            "addend2": addends[1],
            "sum_word": sum_word,
        })

    return results


def display_combinations(combinations):
    for combo in combinations:
        if combo["type"] == "word_sum":
            print(f"  {combo['addend1']} + {combo['addend2']} = {combo['sum_word']}")


if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    print(f"Using: {AVAILABLE_MODELS.get(model_name, model_name)}")
    display_combinations(generate_word_sums(model_name=model_name))

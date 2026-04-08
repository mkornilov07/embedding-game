import os
import random
from itertools import permutations

from gensim.models import KeyedVectors

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "word2vec_filtered.kv")
MIN_SIMILARITY = 0.42
MIN_GAP_RATIO = 1.05
MAX_ATTEMPTS_PER_RESULT = 5000

_model = None
_vocab = None


def _load_model():
    global _model, _vocab
    if _model is not None:
        return
    _model = KeyedVectors.load(MODEL_PATH)
    _vocab = list(_model.index_to_key)


def _best_arrangement(words):
    """Try all 3 arrangements of a triple as A+B=C, return the one with the highest cosmul similarity."""
    best_addends, best_sum, best_sim = None, None, -1
    for a, b, c in permutations(words, 3):
        if a > b:
            continue  # skip duplicate orderings (a+b == b+a)
        top_word, sim = _model.most_similar_cosmul(positive=[a, b], topn=1)[0]
        if top_word == c and sim > best_sim:
            best_addends, best_sum, best_sim = (a, b), c, sim
    return best_addends, best_sum, best_sim


def generate_word_sums(count=5):
    """Generate word sums using 3CosMul vector arithmetic."""
    _load_model()

    results = []
    seen = set()

    for _ in range(count * MAX_ATTEMPTS_PER_RESULT):
        if len(results) >= count:
            break

        w1, w2 = random.sample(_vocab, 2)
        top2 = _model.most_similar_cosmul(positive=[w1, w2], topn=2)

        # Skip if either input word appears in top 2 (weak analogy)
        if any(w in (w1, w2) for w, _ in top2):
            continue

        w3, sim1 = top2[0]
        sim2 = top2[1][1]

        if sim1 < MIN_SIMILARITY or sim1 < sim2 * MIN_GAP_RATIO:
            continue

        triple = frozenset({w1, w2, w3})
        if triple in seen:
            continue
        seen.add(triple)

        addends, sum_word, _ = _best_arrangement((w1, w2, w3))
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
    display_combinations(generate_word_sums())

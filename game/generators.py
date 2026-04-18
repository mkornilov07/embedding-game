import json
import os
import random
from itertools import permutations

import numpy as np
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
    "conceptnet-numberbatch-en": "ConceptNet Numberbatch (300d)",
}

DEFAULT_MODEL = "glove-wiki-gigaword-300"
MIN_SIMILARITY_COSADD = 0.65
MIN_REFINED_SIMILARITY_COSADD = 0.65
MIN_SIMILARITY_COSMUL = 0.25
MIN_REFINED_SIMILARITY_COSMUL = 0.5
MIN_GAP_RATIO = 1.0
MAX_SYNONYM_SIMILARITY = 0.60
MAX_ATTEMPTS_PER_RESULT = 1000
REFINE_ITERATIONS = 3
DEFAULT_TOP_N_VOCAB = 3000

_cache = {}


def _apply_abtt(model):
    """All-but-the-top postprocessing: subtract mean, remove top D/100 PCs.

    Mu & Viswanath (2018), "All-but-the-Top", ICLR.
    """
    vectors = model.vectors.astype(np.float32)
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    n_components = max(1, vectors.shape[1] // 100)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    top = vt[:n_components]
    projection = (centered @ top.T) @ top
    model.vectors = (centered - projection).astype(vectors.dtype)
    if hasattr(model, "norms"):
        model.norms = None


def _load_model(model_name, abtt=False):
    key = (model_name, abtt)
    if key in _cache:
        return _cache[key]

    path = os.path.join(DATA_DIR, f"{model_name}.kv")
    model = KeyedVectors.load(path)
    if abtt:
        _apply_abtt(model)
    vocab = list(model.index_to_key)
    _cache[key] = (model, vocab)
    return model, vocab


def _sum_vector_similarity(model, addend_words, target_word):
    """Cosine similarity of target_word's vector to the sum of addend vectors."""
    sum_vec = np.sum([model[w] for w in addend_words], axis=0)
    target_vec = model[target_word]
    norm = np.linalg.norm(sum_vec) * np.linalg.norm(target_vec)
    if norm == 0:
        return 0.0
    return float(np.dot(sum_vec, target_vec) / norm)


def _any_pair_too_similar(model, words, threshold):
    """Return True if any two words in the triple are near-synonyms."""
    w1, w2, w3 = words
    return (model.similarity(w1, w2) > threshold
            or model.similarity(w1, w3) > threshold
            or model.similarity(w2, w3) > threshold)


def _refine_triple(model, a, b, c, similar, iterations, max_synonym_similarity):
    """Iteratively tighten a triple by substituting each word via vector arithmetic.

    Each iteration:
      - Replace A with best match for C - B
      - Replace B with best match for C - A
      - Replace C with best match for A + B
    Stops early if any pair becomes too similar.
    """
    for _ in range(iterations):
        # C - B -> new A
        new_a_candidates = similar(positive=[c], negative=[b], topn=3)
        new_a = next((w for w, _ in new_a_candidates if w not in (b, c)), None)
        if new_a:
            a = new_a

        # C - A -> new B
        new_b_candidates = similar(positive=[c], negative=[a], topn=3)
        new_b = next((w for w, _ in new_b_candidates if w not in (a, c)), None)
        if new_b:
            b = new_b

        # A + B -> new C
        new_c_candidates = similar(positive=[a, b], topn=3)
        new_c = next((w for w, _ in new_c_candidates if w not in (a, b)), None)
        if new_c:
            c = new_c

        if _any_pair_too_similar(model, (a, b, c), max_synonym_similarity):
            return None

    return a, b, c


def _best_arrangement(words, similar):
    """Try all 3 arrangements of a triple as A+B=C, return the one with the highest similarity."""
    best_addends, best_sum, best_sim = None, None, -1
    for a, b, c in permutations(words, 3):
        if a > b:
            continue
        top_word, sim = similar(positive=[a, b], topn=1)[0]
        if top_word == c and sim > best_sim:
            best_addends, best_sum, best_sim = (a, b), c, sim
    return best_addends, best_sum, best_sim


def generate_word_sums(count=5, model_name=None, related_pairs=False,
                        cosmul=True, abtt=False, top_n_vocab=DEFAULT_TOP_N_VOCAB,
                        refine_iterations=REFINE_ITERATIONS,
                        min_similarity=None,
                        min_refined_similarity=None,
                        min_gap_ratio=MIN_GAP_RATIO,
                        max_synonym_similarity=MAX_SYNONYM_SIMILARITY):
    """Generate word sums using vector arithmetic.

    Args:
        cosmul: if True, use 3CosMul (Levy & Goldberg 2014) instead of 3CosAdd.
        abtt: if True, postprocess embeddings via all-but-the-top
              (Mu & Viswanath 2018) — subtract mean, remove top D/100 PCs.
        top_n_vocab: restrict seed vocabulary to the top-N most frequent tokens.
                     KeyedVectors are ordered by frequency; this avoids
                     long-tail noise. Pass 0 to disable.
        related_pairs: if True, sample w2 from w1's neighborhood (~10th most
                       similar) rather than uniformly at random.
        refine_iterations: number of substitution passes in _refine_triple.
        min_similarity: lower bound on raw top similarity during sampling.
                        Defaults to the cosmul/cosadd constant based on mode.
        min_refined_similarity: lower bound on _best_arrangement score.
                                Defaults to the cosmul/cosadd constant.
        min_gap_ratio: top similarity must exceed runner-up by this factor.
        max_synonym_similarity: reject triples where any pair exceeds this.
    """
    model, vocab = _load_model(model_name or DEFAULT_MODEL, abtt=abtt)
    vocab_pool = vocab[:top_n_vocab] if top_n_vocab and top_n_vocab > 0 else vocab

    similar = model.most_similar_cosmul if cosmul else model.most_similar
    if min_similarity is None:
        min_similarity = MIN_SIMILARITY_COSMUL if cosmul else MIN_SIMILARITY_COSADD
    if min_refined_similarity is None:
        min_refined_similarity = MIN_REFINED_SIMILARITY_COSMUL if cosmul else MIN_REFINED_SIMILARITY_COSADD

    results = []
    seen = set()

    for _ in range(count * MAX_ATTEMPTS_PER_RESULT):
        if len(results) >= count:
            break

        w1 = random.choice(vocab_pool)
        if related_pairs:
            neighbors = similar(positive=[w1], topn=50)
            w2 = random.choice(neighbors[9:])[0]
        else:
            w2 = random.choice(vocab_pool)
            if w2 == w1:
                continue
        top = similar(positive=[w1, w2], topn=2)

        if any(w in (w1, w2) for w, _ in top):
            continue

        w3, sim1 = top[0]
        sim2 = top[1][1]

        if sim1 < min_similarity or sim1 < sim2 * min_gap_ratio:
            continue

        if _any_pair_too_similar(model, (w1, w2, w3), max_synonym_similarity):
            continue

        refined = _refine_triple(model, w1, w2, w3, similar,
                                  refine_iterations, max_synonym_similarity)
        if refined is None:
            continue
        w1, w2, w3 = refined

        triple = frozenset({w1, w2, w3})
        if triple in seen:
            continue
        seen.add(triple)

        addends, sum_word, best_sim = _best_arrangement((w1, w2, w3), similar)
        if addends is None or best_sim < min_refined_similarity:
            continue

        addend_sim = float(model.similarity(addends[0], addends[1]))
        sum_sim = _sum_vector_similarity(model, addends, sum_word)

        results.append({
            "type": "word_sum",
            "addend1": addends[0],
            "addend2": addends[1],
            "sum_word": sum_word,
            "addend_similarity": addend_sim,
            "sum_similarity": sum_sim,
        })

    return results


CURATED_PATH = os.path.join(os.path.dirname(__file__), "word_sums.json")
_curated = None


def _load_curated():
    global _curated
    if _curated is None:
        with open(CURATED_PATH, "r", encoding="utf-8") as f:
            _curated = json.load(f)
    return _curated


def generate_curated_word_sums(count=5):
    """Pick random word sums from the curated JSON list."""
    pool = _load_curated()
    selected = random.sample(pool, min(count, len(pool)))
    return [{"type": "word_sum", **s} for s in selected]


def display_combinations(combinations):
    for combo in combinations:
        if combo["type"] == "word_sum":
            print(f"  {combo['addend1']} + {combo['addend2']} = {combo['sum_word']}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "curated":
        print("=== Curated ===")
        display_combinations(generate_curated_word_sums(5))
    else:
        model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
        print(f"Using: {AVAILABLE_MODELS.get(model_name, model_name)}")
        display_combinations(generate_word_sums(model_name=model_name))

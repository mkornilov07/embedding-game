import json
import math
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import permutations

import numpy as np
from gensim.models import KeyedVectors

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

AVAILABLE_MODELS = {
    "conceptnet-numberbatch-en-3k": "ConceptNet Numberbatch (3,000 words)",
    "conceptnet-numberbatch-en-3k-abtt": "ConceptNet Numberbatch (3,000 words, ABTT)",
    "conceptnet-numberbatch-en-10k": "ConceptNet Numberbatch (10,000 words)",
    "conceptnet-numberbatch-en-10k-abtt": "ConceptNet Numberbatch (10,000 words, ABTT)",
    "openai-text-embedding-3-large-subtlex-us-5000": (
        "OpenAI text-embedding-3-large (5,000 words)"
    ),
    "openai-text-embedding-3-large-subtlex-us-5000-abtt": (
        "OpenAI text-embedding-3-large (5,000 words, ABTT)"
    ),
    "openai-text-embedding-3-large-300d-subtlex-us-5000": (
        "OpenAI text-embedding-3-large (300 dimensions, 5,000 words)"
    ),
    "openai-text-embedding-3-large-300d-subtlex-us-5000-abtt": (
        "OpenAI text-embedding-3-large (300 dimensions, 5,000 words, ABTT)"
    ),
}

DEFAULT_MODEL = "conceptnet-numberbatch-en-3k"


MIN_SIMILARITY_COSADD = 0.60
MIN_REFINED_SIMILARITY_COSADD = 0.60
MIN_SIMILARITY_COSMUL = 0.25
MIN_REFINED_SIMILARITY_COSMUL = 0.5
MIN_SIMILARITY_RAW_SUM = 0.60
MIN_REFINED_SIMILARITY_RAW_SUM = 0.60
DEFAULT_SUM_SCORING_METHOD = "cosmul"
SUM_SCORING_METHODS = {
    "cosadd": (MIN_SIMILARITY_COSADD, MIN_REFINED_SIMILARITY_COSADD),
    "cosmul": (MIN_SIMILARITY_COSMUL, MIN_REFINED_SIMILARITY_COSMUL),
    "raw_sum": (MIN_SIMILARITY_RAW_SUM, MIN_REFINED_SIMILARITY_RAW_SUM),
}
ORTHOGONALITY_TOLERANCE = 0.1
MIN_GAP_RATIO = 1.0
MAX_SYNONYM_SIMILARITY = 0.45
MAX_ATTEMPTS_PER_RESULT = 300
MAX_PARALLEL_SEARCHES = 4
REFINE_ITERATIONS = 1
WORD_CHAIN_3K_MODEL = "conceptnet-numberbatch-en-3k-abtt"
WORD_CHAIN_10K_MODEL = "conceptnet-numberbatch-en-10k-abtt"
WORD_CHAIN_MODEL = WORD_CHAIN_10K_MODEL
WORD_CHAIN_MIN_SIMILARITY = 0.47
WORD_CHAIN_MAX_EARLIER_SIMILARITY = 0.31
WORD_CHAIN_START_PERCENT = 20
WORD_CHAIN_MAX_ATTEMPTS = 20

_cache = {}
_cache_lock = threading.Lock()
_search_cache_lock = threading.Lock()


def _load_model(model_name):
    if model_name in _cache:
        return _cache[model_name]

    with _cache_lock:
        if model_name in _cache:
            return _cache[model_name]
        filename = f"{model_name}.kv"
        path = os.path.join(DATA_DIR, filename)
        model = KeyedVectors.load(path)
        model.fill_norms()
        model._normalized_vectors = np.divide(
            model.vectors,
            model.norms[:, None],
            out=np.zeros_like(model.vectors, dtype=np.float32),
            where=model.norms[:, None] != 0,
        )
        model._sum_searchers = {}
        vocab = list(model.index_to_key)
        _cache[model_name] = (model, vocab)
        return model, vocab


def validate_word_chain(words, model_name=WORD_CHAIN_MODEL,
                        min_previous_similarity=WORD_CHAIN_MIN_SIMILARITY):
    """Check vocabulary and adjacent minimum similarity."""
    model, _ = _load_model(model_name)
    words = list(words)
    unknown = {
        index for index, word in enumerate(words)
        if not isinstance(word, str) or word not in model.key_to_index
    }
    invalid = set(unknown)
    invalid_links = []
    similarities = []

    for index, (first, second) in enumerate(zip(words, words[1:])):
        if index in unknown or index + 1 in unknown:
            similarities.append(None)
            continue
        similarity = float(model.similarity(first, second))
        similarities.append(similarity)
        if similarity < min_previous_similarity:
            invalid_links.append(index)

    if len(words) < 2:
        invalid.update(range(len(words)))

    return {
        "valid": len(words) >= 2 and not invalid and not invalid_links,
        "invalid_slots": sorted(invalid),
        "invalid_links": invalid_links,
        "similarities": similarities,
    }


def generate_word_chain(
    length=5,
    min_previous_similarity=WORD_CHAIN_MIN_SIMILARITY,
    max_earlier_similarity=WORD_CHAIN_MAX_EARLIER_SIMILARITY,
    model_name=WORD_CHAIN_MODEL,
    start_percent=WORD_CHAIN_START_PERCENT,
):
    """Sample a chain with adjacent and non-adjacent cosine constraints."""
    if length < 2:
        raise ValueError("A word chain must contain at least two words.")
    min_previous_similarity = float(min_previous_similarity)
    max_earlier_similarity = float(max_earlier_similarity)
    start_percent = float(start_percent)
    if not np.isfinite(min_previous_similarity) or not -1 <= min_previous_similarity <= 1:
        raise ValueError("Minimum previous-word similarity must be between -1 and 1.")
    if not np.isfinite(max_earlier_similarity) or not -1 <= max_earlier_similarity <= 1:
        raise ValueError("Maximum earlier-word similarity must be between -1 and 1.")
    if not np.isfinite(start_percent) or not 0 < start_percent <= 100:
        raise ValueError("Starting vocabulary percentage must be greater than 0 and at most 100.")

    model, vocab = _load_model(model_name)
    start_limit = math.ceil(len(vocab) * start_percent / 100)
    use_on_demand_similarities = len(vocab) > 3000 or model_name == WORD_CHAIN_10K_MODEL
    if use_on_demand_similarities:
        @lru_cache(maxsize=32)
        def similarity_row(index):
            return np.clip(model.similar_by_key(vocab[index], topn=None), -1.0, 1.0)

        seeds = range(start_limit)
    else:
        if not hasattr(model, "_chain_similarities"):
            vectors = model._normalized_vectors
            similarities = np.clip(vectors @ vectors.T, -1.0, 1.0)
            np.fill_diagonal(similarities, 1.0)
            model._chain_similarities = similarities
            model._chain_seeds = {}

        similarities = model._chain_similarities
        seed_key = (length, min_previous_similarity)
        if seed_key not in model._chain_seeds:
            eligible = (similarities >= min_previous_similarity) & (similarities < 1.0)
            unseen = np.ones(len(vocab), dtype=bool)
            seeds = []
            for start in range(len(vocab)):
                if not unseen[start]:
                    continue
                component = []
                stack = [start]
                unseen[start] = False
                while stack:
                    current = stack.pop()
                    component.append(current)
                    neighbors = np.flatnonzero(eligible[current] & unseen)
                    unseen[neighbors] = False
                    stack.extend(neighbors.tolist())
                if len(component) >= length:
                    seeds.extend(component)
            model._chain_seeds[seed_key] = seeds

        seeds = [index for index in model._chain_seeds[seed_key] if index < start_limit]

        def similarity_row(index):
            return similarities[index]

    if not seeds:
        return None

    for _ in range(WORD_CHAIN_MAX_ATTEMPTS):
        chain_indexes = [random.choice(seeds)]
        while len(chain_indexes) < length:
            current = chain_indexes[-1]
            row = similarity_row(current)
            allowed = (row >= min_previous_similarity) & (row < 1.0)
            allowed[chain_indexes] = False
            for earlier in chain_indexes[:-1]:
                allowed &= similarity_row(earlier) <= max_earlier_similarity
            options = np.flatnonzero(allowed)
            if not len(options):
                break
            chain_indexes.append(random.choices(
                options.tolist(), weights=(1.0 - row[options]).tolist(), k=1,
            )[0])
        if len(chain_indexes) == length:
            return [vocab[index] for index in chain_indexes]
    return None


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


def _similarity_search(model, method):
    if method not in SUM_SCORING_METHODS:
        raise ValueError(f"Unknown sum scoring method: {method}")

    with _search_cache_lock:
        searchers = getattr(model, "_sum_searchers", None)
        if searchers is None:
            searchers = model._sum_searchers = {}
        if method in searchers:
            return searchers[method]

        normalized = getattr(model, "_normalized_vectors", None)
        if normalized is None:
            model.fill_norms()
            normalized = model._normalized_vectors = np.divide(
                model.vectors,
                model.norms[:, None],
                out=np.zeros_like(model.vectors, dtype=np.float32),
                where=model.norms[:, None] != 0,
            )

    def score_query(positive, negative):
        excluded = {model.key_to_index[word] for word in positive + negative}
        if method == "cosmul":
            scores = np.ones(len(model), dtype=np.float32)
            for word in positive:
                scores *= (1 + normalized @ normalized[model.key_to_index[word]]) / 2
            negative_scores = np.ones(len(model), dtype=np.float32)
            for word in negative:
                negative_scores *= (1 + normalized @ normalized[model.key_to_index[word]]) / 2
            return scores / (negative_scores + 1e-6), excluded

        if method == "raw_sum":
            query = np.zeros(model.vector_size, dtype=np.float32)
            for word in positive:
                query += model[word]
            for word in negative:
                query -= model[word]
        else:
            query = np.zeros(model.vector_size, dtype=np.float32)
            for word in positive:
                query += normalized[model.key_to_index[word]]
            for word in negative:
                query -= normalized[model.key_to_index[word]]

        query_norm = np.linalg.norm(query)
        if query_norm:
            query /= query_norm
        return normalized @ query, excluded

    @lru_cache(maxsize=4096)
    def cached_top_words(positive, negative, limit):
        scores, excluded = score_query(positive, negative)
        candidate_count = min(len(scores), limit + len(excluded))
        if candidate_count == len(scores):
            candidates = np.arange(len(scores))
        else:
            candidates = np.argpartition(scores, -candidate_count)[-candidate_count:]
        candidates = candidates[np.argsort(scores[candidates])[::-1]]
        return tuple(
            (model.index_to_key[index], float(scores[index]))
            for index in candidates
            if index not in excluded
        )[:limit]

    def similar(positive=(), negative=(), topn=10):
        if topn < 1:
            return []
        positive = tuple(sorted(positive))
        negative = tuple(sorted(negative))
        limit = max(50, topn)
        return list(cached_top_words(positive, negative, limit)[:topn])

    similar.cache_info = cached_top_words.cache_info
    with _search_cache_lock:
        return model._sum_searchers.setdefault(method, similar)


def _find_orthogonal_partner(model, vocab_pool, word, prefer_related, rng=random):
    candidate_indexes = rng.sample(range(len(vocab_pool)), min(256, len(vocab_pool)))
    candidates = [vocab_pool[index] for index in candidate_indexes if vocab_pool[index] != word]
    if not candidates:
        return None

    normalized = model._normalized_vectors
    base = normalized[model.key_to_index[word]]
    candidate_ids = [model.key_to_index[candidate] for candidate in candidates]
    similarities = normalized[candidate_ids] @ base
    eligible = [
        (candidate, float(similarity))
        for candidate, similarity in zip(candidates, similarities)
        if abs(similarity) <= ORTHOGONALITY_TOLERANCE
    ]
    if not eligible:
        return None
    if prefer_related:
        return max(eligible, key=lambda item: item[1])[0]
    return rng.choice(eligible)[0]


def _best_arrangement(words, similar, model):
    """Return the highest-scoring A+B=C arrangement with near-orthogonal addends."""
    best_addends, best_sum, best_sim = None, None, -1
    for a, b, c in permutations(words, 3):
        if a > b:
            continue
        if abs(model.similarity(a, b)) > ORTHOGONALITY_TOLERANCE:
            continue
        top_word, sim = similar(positive=[a, b], topn=1)[0]
        if top_word == c and sim > best_sim:
            best_addends, best_sum, best_sim = (a, b), c, sim
    return best_addends, best_sum, best_sim


def _try_word_sum(model, vocab_pool, similar, parameters, rng):
    w1 = rng.choice(vocab_pool)
    w2 = _find_orthogonal_partner(
        model, vocab_pool, w1, parameters["related_pairs"], rng,
    )
    if w2 is None:
        return None

    top = similar(positive=[w1, w2], topn=2)
    if len(top) < 2 or any(word in (w1, w2) for word, _ in top):
        return None

    w3, best_score = top[0]
    runner_up_score = top[1][1]
    if (best_score < parameters["min_similarity"]
            or best_score < runner_up_score * parameters["min_gap_ratio"]):
        return None
    if _any_pair_too_similar(
        model, (w1, w2, w3), parameters["max_synonym_similarity"],
    ):
        return None

    refined = _refine_triple(
        model, w1, w2, w3, similar,
        parameters["refine_iterations"], parameters["max_synonym_similarity"],
    )
    if refined is None:
        return None

    addends, sum_word, score = _best_arrangement(refined, similar, model)
    if addends is None or score < parameters["min_refined_similarity"]:
        return None

    return {
        "type": "word_sum",
        "addend1": addends[0],
        "addend2": addends[1],
        "sum_word": sum_word,
        "addend_similarity": float(model.similarity(*addends)),
        "sum_similarity": _sum_vector_similarity(model, addends, sum_word),
    }


def generate_word_sums(
    count=5,
    model_name=None,
    related_pairs=None,
    sum_scoring_method=None,
    cosmul=None,
    refine_iterations=None,
    min_similarity=None,
    min_refined_similarity=None,
    min_gap_ratio=None,
    max_synonym_similarity=None,
    max_attempts_per_result=MAX_ATTEMPTS_PER_RESULT,
):
    """Generate word sums using vector arithmetic.

    Args:
        sum_scoring_method: one of ``cosadd``, ``cosmul``, or ``raw_sum``.
        cosmul: deprecated compatibility option; selects ``cosmul`` or ``cosadd``.
        related_pairs: if True, prefer the most similar sampled addend that
                       still satisfies the near-orthogonality constraint.
        refine_iterations: number of substitution passes in _refine_triple.
        min_similarity: lower bound on raw top similarity during sampling.
                        Defaults to the cosmul/cosadd constant based on mode.
        min_refined_similarity: lower bound on _best_arrangement score.
                                Defaults to the cosmul/cosadd constant.
        min_gap_ratio: top similarity must exceed runner-up by this factor.
        max_synonym_similarity: reject triples where any pair exceeds this.
    """
    if sum_scoring_method is None and cosmul is not None:
        sum_scoring_method = "cosmul" if cosmul else "cosadd"

    if any(value is None for value in (
        model_name, related_pairs, sum_scoring_method,
        refine_iterations, min_similarity, min_refined_similarity,
        min_gap_ratio, max_synonym_similarity,
    )):
        try:
            from .models import GenerationDefaults
            defaults = GenerationDefaults.objects.get(key=model_name or DEFAULT_MODEL)
        except Exception:
            defaults = None
        if defaults:
            model_name = defaults.model_name if model_name is None else model_name
            related_pairs = defaults.related_pairs if related_pairs is None else related_pairs
            sum_scoring_method = defaults.sum_scoring_method if sum_scoring_method is None else sum_scoring_method
            refine_iterations = defaults.refine_iterations if refine_iterations is None else refine_iterations
            min_similarity = defaults.min_similarity if min_similarity is None else min_similarity
            min_refined_similarity = defaults.min_refined_similarity if min_refined_similarity is None else min_refined_similarity
            min_gap_ratio = defaults.min_gap_ratio if min_gap_ratio is None else min_gap_ratio
            max_synonym_similarity = defaults.max_synonym_similarity if max_synonym_similarity is None else max_synonym_similarity

    model_name = model_name or DEFAULT_MODEL
    related_pairs = True if related_pairs is None else related_pairs
    if sum_scoring_method is None:
        sum_scoring_method = "cosmul" if cosmul is None or cosmul else "cosadd"
    if sum_scoring_method not in SUM_SCORING_METHODS:
        raise ValueError(f"Unknown sum scoring method: {sum_scoring_method}")
    refine_iterations = REFINE_ITERATIONS if refine_iterations is None else refine_iterations
    min_gap_ratio = MIN_GAP_RATIO if min_gap_ratio is None else min_gap_ratio
    max_synonym_similarity = MAX_SYNONYM_SIMILARITY if max_synonym_similarity is None else max_synonym_similarity

    model, vocab = _load_model(model_name)
    vocab_pool = vocab

    similar = _similarity_search(model, sum_scoring_method)
    default_min_similarity, default_min_refined_similarity = SUM_SCORING_METHODS[sum_scoring_method]
    if min_similarity is None:
        min_similarity = default_min_similarity
    if min_refined_similarity is None:
        min_refined_similarity = default_min_refined_similarity

    parameters = {
        "related_pairs": related_pairs,
        "refine_iterations": refine_iterations,
        "min_similarity": min_similarity,
        "min_refined_similarity": min_refined_similarity,
        "min_gap_ratio": min_gap_ratio,
        "max_synonym_similarity": max_synonym_similarity,
    }

    def try_seed(seed):
        return _try_word_sum(
            model, vocab_pool, similar, parameters, random.Random(seed),
        )

    results = []
    seen = set()
    attempts_remaining = max(0, count * max_attempts_per_result)
    workers = min(MAX_PARALLEL_SEARCHES, max(1, count))

    def accept_results(candidates):
        for candidate in candidates:
            if candidate is None:
                continue
            key = frozenset((candidate["addend1"], candidate["addend2"], candidate["sum_word"]))
            if key not in seen:
                seen.add(key)
                results.append(candidate)
                if len(results) == count:
                    break

    if workers == 1:
        while attempts_remaining and len(results) < count:
            candidate = try_seed(random.getrandbits(64))
            attempts_remaining -= 1
            accept_results([candidate])
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while attempts_remaining and len(results) < count:
                batch_size = min(workers, attempts_remaining)
                seeds = [random.getrandbits(64) for _ in range(batch_size)]
                accept_results(executor.map(try_seed, seeds))
                attempts_remaining -= batch_size

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

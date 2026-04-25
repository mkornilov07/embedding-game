"""Comprehensive sweep: verify the recommended regime and test alternatives."""
import os
import sys
import django

sys.path.insert(0, ".")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mysite.settings")
django.setup()

from game.generators import generate_word_sums

MODEL = "conceptnet-numberbatch-en"

REC = dict(cosmul=True, abtt=True, related_pairs=True,
           top_n_vocab=2000, refine_iterations=1, max_synonym_similarity=0.45)


def run(label, n_batches=3, count=5, **overrides):
    cfg = {**REC, **overrides}
    print(f"\n=== {label} | {cfg} ===")
    all_results = []
    for _ in range(n_batches):
        try:
            results = generate_word_sums(count=count, model_name=MODEL, **cfg)
            for r in results:
                print(f"  {r['addend1']} + {r['addend2']} = {r['sum_word']}")
                all_results.append(r)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
    print(f"  TOTAL: {len(all_results)} results across {n_batches} batches")


if __name__ == "__main__":
    print("\n########## PHASE 1: Robustness — recommended config × 10 batches ##########")
    run("recommended × 10", n_batches=10)

    print("\n########## PHASE 2: One-axis variations ##########")
    for top in [1500, 2500, 3000]:
        run(f"top_n_vocab={top}", n_batches=3, top_n_vocab=top)
    for mss in [0.42, 0.43, 0.44, 0.46, 0.48, 0.50]:
        run(f"mss={mss}", n_batches=3, max_synonym_similarity=mss)
    for refine in [0, 2, 3]:
        run(f"refine={refine}", n_batches=3, refine_iterations=refine)
    for cosmul in [False]:
        run(f"cosmul={cosmul}", n_batches=3, cosmul=cosmul)

    print("\n########## PHASE 3: Alternative regimes ##########")
    run("LOOSE: top=3000, mss=0.50", n_batches=3,
        top_n_vocab=3000, max_synonym_similarity=0.50)
    run("STRICT: top=1500, mss=0.45, gap=1.1", n_batches=3,
        top_n_vocab=1500, max_synonym_similarity=0.45, min_gap_ratio=1.1)
    run("COSADD: cosmul=False, mss=0.50", n_batches=3,
        cosmul=False, max_synonym_similarity=0.50)
    run("DEEP: refine=3, mss=0.45", n_batches=3,
        refine_iterations=3, max_synonym_similarity=0.45)
    run("FAST: refine=0, mss=0.45", n_batches=3,
        refine_iterations=0, max_synonym_similarity=0.45)

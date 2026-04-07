import json
import random

from django.shortcuts import get_object_or_404, render

from .models import Puzzle, WordSum


def puzzle_list(request):
    puzzles = Puzzle.objects.all()
    return render(request, "game/puzzle_list.html", {"puzzles": puzzles})


def puzzle(request, pk):
    p = get_object_or_404(Puzzle, pk=pk)
    combinations = list(p.combinations.all())
    random.shuffle(combinations)

    words = []
    sequences = []

    for i, combo in enumerate(combinations):
        # Try to resolve to a specific subclass
        try:
            ws = combo.wordsum
        except WordSum.DoesNotExist:
            continue

        combo_words = [
            {"text": ws.addend1, "seq": i, "group": 0},
            {"text": ws.addend2, "seq": i, "group": 0},
            {"text": ws.sum_word, "seq": i, "group": 1},
        ]
        words.extend(combo_words)
        sequences.append({
            "type": "word_sum",
            "num_slots": 3,
            "operators": ["+", "="],
            "slot_groups": [[0], [0], [1]],
        })

    random.shuffle(words)

    return render(request, "game/puzzle.html", {
        "puzzle": p,
        "words_json": json.dumps(words),
        "sequences_json": json.dumps(sequences),
    })

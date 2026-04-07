import json
import random

from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_POST

from .models import Puzzle, WordSum


def puzzle_list(request):
    puzzles = Puzzle.objects.all()
    return render(request, "game/puzzle_list.html", {"puzzles": puzzles})


def puzzle(request, pk):
    p = get_object_or_404(Puzzle, pk=pk)
    db_combinations = list(p.combinations.all())
    random.shuffle(db_combinations)

    words = []
    combinations = []

    for i, combo in enumerate(db_combinations):
        try:
            ws = combo.wordsum
        except WordSum.DoesNotExist:
            continue

        words.extend([
            {"text": ws.addend1, "combo": i},
            {"text": ws.addend2, "combo": i},
            {"text": ws.sum_word, "combo": i},
        ])
        combinations.append({"id": combo.pk, "type": "word_sum"})

    random.shuffle(words)

    return render(request, "game/puzzle.html", {
        "puzzle": p,
        "words_json": json.dumps(words),
        "combinations_json": json.dumps(combinations),
    })


@require_POST
def check_puzzle(request, pk):
    p = get_object_or_404(Puzzle, pk=pk)
    data = json.loads(request.body)

    # Build set of valid word sums to match against
    unmatched = set()
    for combo in p.combinations.all():
        try:
            ws = combo.wordsum
            # Frozenset for addends since order doesn't matter
            unmatched.add((frozenset({ws.addend1, ws.addend2}), ws.sum_word))
        except WordSum.DoesNotExist:
            continue

    for entry in data:
        slot_words = entry["words"]
        key = (frozenset(slot_words[:2]), slot_words[2])
        if key not in unmatched:
            return JsonResponse({"correct": False})
        unmatched.remove(key)

    return JsonResponse({"correct": len(unmatched) == 0})

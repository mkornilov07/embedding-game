import json
import random

from django.shortcuts import render

from .models import WordSum


def puzzle(request):
    word_sums = list(WordSum.objects.all())
    random.shuffle(word_sums)

    # Collect all words and shuffle them for the word bank
    words = []
    for ws in word_sums:
        words.extend([ws.addend1, ws.addend2, ws.sum_word])
    random.shuffle(words)

    # Build equation data for the template
    equations = [{"id": ws.id} for ws in word_sums]

    return render(request, "game/puzzle.html", {
        "words_json": json.dumps(words),
        "equations_json": json.dumps(equations),
    })

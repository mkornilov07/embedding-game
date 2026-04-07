import json
import random

from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .models import Puzzle, PuzzleCompletion, WordSum


def puzzle_list(request):
    puzzles = Puzzle.objects.all()
    if request.user.is_authenticated:
        completed_ids = set(
            PuzzleCompletion.objects.filter(user=request.user).values_list("puzzle_id", flat=True)
        )
    else:
        completed_ids = set()
    return render(request, "game/puzzle_list.html", {
        "puzzles": puzzles,
        "completed_ids": completed_ids,
    })


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

    correct = len(unmatched) == 0
    if correct and request.user.is_authenticated:
        PuzzleCompletion.objects.get_or_create(user=request.user, puzzle=p)
    return JsonResponse({"correct": correct})


def register(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("puzzle_list")
    else:
        form = UserCreationForm()
    return render(request, "game/register.html", {"form": form})

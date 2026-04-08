import json
import random

from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from django.contrib.auth.decorators import login_required

from .generators import AVAILABLE_MODELS, generate_word_sums
from .models import Puzzle, PuzzleCompletion, WordSum


def puzzle_list(request):
    puzzles = Puzzle.objects.select_related("created_by").all()
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


@login_required
def create_puzzle(request):
    return render(request, "game/create_puzzle.html", {
        "available_models": AVAILABLE_MODELS,
    })


@login_required
@require_POST
def save_puzzle(request):
    data = json.loads(request.body)
    name = data.get("name", "").strip()
    word_sums = data.get("word_sums", [])

    if not name or not word_sums:
        return JsonResponse({"error": "Name and at least one word sum required."}, status=400)

    puzzle = Puzzle.objects.create(name=name, created_by=request.user)
    for ws in word_sums:
        WordSum.objects.create(
            puzzle=puzzle,
            addend1=ws["addend1"].strip(),
            addend2=ws["addend2"].strip(),
            sum_word=ws["sum_word"].strip(),
        )

    return JsonResponse({"id": puzzle.pk})


@login_required
@require_POST
def delete_puzzle(request, pk):
    puzzle = get_object_or_404(Puzzle, pk=pk, created_by=request.user)
    puzzle.delete()
    return JsonResponse({"ok": True})


@login_required
@require_POST
def generate_combinations(request):
    data = json.loads(request.body)
    gen_type = data.get("type", "word_sum")
    count = min(data.get("count", 5), 5)

    model_name = data.get("model")

    related_pairs = data.get("related_pairs", False)

    if gen_type == "word_sum":
        combos = generate_word_sums(count, model_name=model_name, related_pairs=related_pairs)
    else:
        return JsonResponse({"error": f"Unknown type: {gen_type}"}, status=400)

    return JsonResponse({"combinations": combos})

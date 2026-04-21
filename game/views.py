import json
import random

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.contrib.auth import get_user_model, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.db.models import Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_POST

from .generators import AVAILABLE_MODELS, generate_curated_word_sums, generate_word_sums
from .models import Duel, DuelProgress, Puzzle, PuzzleCompletion, WordSum


def puzzle_list(request):
    puzzles = Puzzle.objects.select_related("created_by").order_by("for_duel", "-pk")
    has_duel_puzzles = any(p.for_duel for p in puzzles)
    completed_ids = set()
    incoming_invites = []
    outgoing_invite = None
    active_duel = None
    if request.user.is_authenticated:
        completed_ids = set(
            PuzzleCompletion.objects.filter(user=request.user).values_list("puzzle_id", flat=True)
        )
        incoming_invites = list(
            Duel.objects.filter(opponent=request.user, status=Duel.STATUS_PENDING)
            .select_related("inviter", "puzzle")
        )
        outgoing_invite = (
            Duel.objects.filter(inviter=request.user, status=Duel.STATUS_PENDING)
            .select_related("opponent", "puzzle")
            .first()
        )
        active_duel = (
            Duel.objects.filter(status=Duel.STATUS_ACTIVE)
            .filter(Q(inviter=request.user) | Q(opponent=request.user))
            .first()
        )
    return render(request, "game/puzzle_list.html", {
        "puzzles": puzzles,
        "completed_ids": completed_ids,
        "available_models": AVAILABLE_MODELS,
        "incoming_invites": incoming_invites,
        "outgoing_invite": outgoing_invite,
        "active_duel": active_duel,
        "has_duel_puzzles": has_duel_puzzles,
    })


def puzzle(request, pk):
    p = get_object_or_404(Puzzle, pk=pk)
    db_combinations = list(p.combinations.all())
    random.shuffle(db_combinations)

    words = []
    combinations = []

    for combo in db_combinations:
        try:
            ws = combo.wordsum
        except WordSum.DoesNotExist:
            continue

        idx = len(combinations)
        words.extend([
            {"text": ws.addend1, "combo": idx},
            {"text": ws.addend2, "combo": idx},
            {"text": ws.sum_word, "combo": idx},
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


@require_POST
def check_row(request, pk):
    """Check a single filled row against the puzzle's word sums.

    Returns which slots are wrong:
    - If 2 of the 3 words match a word sum (in valid positions), only the odd one out is marked wrong.
    - Otherwise all 3 are marked wrong.
    """
    p = get_object_or_404(Puzzle, pk=pk)
    data = json.loads(request.body)
    slot_words = data["words"]  # [addend_slot1, addend_slot2, sum_slot]

    word_sums = []
    for combo in p.combinations.all():
        try:
            ws = combo.wordsum
            word_sums.append((frozenset({ws.addend1, ws.addend2}), ws.sum_word))
        except WordSum.DoesNotExist:
            continue

    addends = frozenset(slot_words[:2])
    sum_word = slot_words[2]

    # Exact match
    if (addends, sum_word) in word_sums:
        return JsonResponse({"wrong_slots": []})

    # Check if 2 of 3 match — addends correct, sum wrong
    for ws_addends, ws_sum in word_sums:
        if addends == ws_addends:
            return JsonResponse({"wrong_slots": [2]})

    # Check if one addend + sum match — the other addend is wrong
    for ws_addends, ws_sum in word_sums:
        if sum_word == ws_sum:
            for i in range(2):
                if slot_words[i] in ws_addends:
                    other = 1 - i
                    return JsonResponse({"wrong_slots": [other]})

    # Check if sum is correct for some word sum, and one addend is in that word sum
    # (already covered above)

    return JsonResponse({"wrong_slots": [0, 1, 2]})


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
def create_puzzle(request, pk=None):
    existing = None
    word_sums = []
    if pk is not None:
        existing = get_object_or_404(Puzzle, pk=pk, created_by=request.user)
        for combo in existing.combinations.all():
            try:
                ws = combo.wordsum
            except WordSum.DoesNotExist:
                continue
            word_sums.append({
                "addend1": ws.addend1, "addend2": ws.addend2, "sum_word": ws.sum_word,
            })
    return render(request, "game/create_puzzle.html", {
        "available_models": AVAILABLE_MODELS,
        "puzzle": existing,
        "word_sums_json": json.dumps(word_sums),
    })


@login_required
@require_POST
def save_puzzle(request):
    data = json.loads(request.body)
    name = data.get("name", "").strip()
    word_sums = data.get("word_sums", [])
    puzzle_id = data.get("id")

    if not name or not word_sums:
        return JsonResponse({"error": "Name and at least one word sum required."}, status=400)

    if puzzle_id is not None:
        puzzle = get_object_or_404(Puzzle, pk=puzzle_id, created_by=request.user)
        puzzle.name = name
        puzzle.save()
        puzzle.combinations.all().delete()
    else:
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

    related_pairs = data.get("related_pairs", True)
    cosmul = data.get("cosmul", False)
    abtt = data.get("abtt", True)
    top_n_vocab = int(data.get("top_n_vocab") or 0)
    refine_iterations = int(data.get("refine_iterations") or 1)
    min_similarity = data.get("min_similarity")
    min_refined_similarity = data.get("min_refined_similarity")
    min_gap_ratio = data.get("min_gap_ratio")
    max_synonym_similarity = data.get("max_synonym_similarity")

    if gen_type == "word_sum":
        if model_name == "curated":
            combos = generate_curated_word_sums(count)
        else:
            kwargs = dict(
                model_name=model_name, related_pairs=related_pairs,
                cosmul=cosmul, abtt=abtt, top_n_vocab=top_n_vocab,
                refine_iterations=refine_iterations,
                min_similarity=min_similarity,
                min_refined_similarity=min_refined_similarity,
            )
            if min_gap_ratio is not None:
                kwargs["min_gap_ratio"] = float(min_gap_ratio)
            if max_synonym_similarity is not None:
                kwargs["max_synonym_similarity"] = float(max_synonym_similarity)
            try:
                combos = generate_word_sums(count, **kwargs)
            except FileNotFoundError:
                return JsonResponse(
                    {"error": f"Model file for {model_name!r} not found. Run build_models.py."},
                    status=400,
                )
    else:
        return JsonResponse({"error": f"Unknown type: {gen_type}"}, status=400)

    return JsonResponse({"combinations": combos})


# ---------------------------- Duels ----------------------------

def _row_key(addend1, addend2, sum_word):
    """Canonical, JSON-serializable key for a solved word-sum row."""
    return [sorted([addend1, addend2]), sum_word]


def _duel_rowkeys(puzzle):
    keys = []
    for combo in puzzle.combinations.all():
        try:
            ws = combo.wordsum
        except WordSum.DoesNotExist:
            continue
        keys.append(_row_key(ws.addend1, ws.addend2, ws.sum_word))
    return keys


def _user_has_open_duel(user):
    return Duel.objects.filter(status__in=[Duel.STATUS_PENDING, Duel.STATUS_ACTIVE]).filter(
        Q(inviter=user) | Q(opponent=user)
    ).exists()


def _lobby_send(user_id, payload):
    layer = get_channel_layer()
    async_to_sync(layer.group_send)(
        f"lobby_{user_id}", {"type": "lobby_event", "data": payload}
    )


def _duel_send(duel_id, payload):
    layer = get_channel_layer()
    async_to_sync(layer.group_send)(
        f"duel_{duel_id}", {"type": "duel_event", "data": payload}
    )


def _duel_summary(duel):
    return {
        "id": duel.pk,
        "inviter": duel.inviter.username,
        "opponent": duel.opponent.username,
        "puzzle_name": duel.puzzle.name,
        "url": reverse("duel_detail", args=[duel.pk]),
    }


@login_required
@require_POST
def create_duel(request):
    if _user_has_open_duel(request.user):
        return JsonResponse(
            {"error": "You already have a pending or active duel."}, status=400
        )

    data = json.loads(request.body)
    opponent_username = (data.get("opponent") or "").strip()
    puzzle_id = data.get("puzzle_id")
    generate = bool(data.get("generate"))
    model_name = data.get("model")

    if not opponent_username:
        return JsonResponse({"error": "Opponent username required."}, status=400)
    if opponent_username == request.user.username:
        return JsonResponse({"error": "You can't duel yourself."}, status=400)

    User = get_user_model()
    try:
        opponent = User.objects.get(username=opponent_username)
    except User.DoesNotExist:
        return JsonResponse({"error": f"No user named {opponent_username!r}."}, status=404)

    if _user_has_open_duel(opponent):
        return JsonResponse(
            {"error": f"{opponent_username} is already in a duel."}, status=400
        )

    if generate:
        if not model_name:
            return JsonResponse({"error": "Pick a model to generate with."}, status=400)
        try:
            count = int(data.get("count", 5))
        except (TypeError, ValueError):
            count = 5
        count = max(1, min(count, 10))
        try:
            if model_name == "curated":
                combos = generate_curated_word_sums(count)
            else:
                combos = generate_word_sums(count, model_name=model_name)
        except FileNotFoundError:
            return JsonResponse(
                {"error": f"Model file for {model_name!r} not found."}, status=400
            )
        if not combos:
            return JsonResponse({"error": "Generator returned nothing; try again."}, status=500)
        puzzle = Puzzle.objects.create(
            name=f"Duel: {request.user.username} vs {opponent.username}",
            created_by=request.user,
            for_duel=True,
        )
        for c in combos:
            WordSum.objects.create(
                puzzle=puzzle,
                addend1=c["addend1"].strip(),
                addend2=c["addend2"].strip(),
                sum_word=c["sum_word"].strip(),
            )
    else:
        if not puzzle_id:
            return JsonResponse({"error": "Pick a puzzle or generate one."}, status=400)
        puzzle = get_object_or_404(Puzzle, pk=puzzle_id)

    duel = Duel.objects.create(
        inviter=request.user, opponent=opponent, puzzle=puzzle,
        status=Duel.STATUS_PENDING,
    )

    _lobby_send(opponent.id, {
        "type": "invite_received",
        "duel": _duel_summary(duel),
    })
    return JsonResponse({"duel_id": duel.pk})


@login_required
@require_POST
def cancel_duel(request, pk):
    duel = get_object_or_404(Duel, pk=pk, inviter=request.user, status=Duel.STATUS_PENDING)
    duel.status = Duel.STATUS_CANCELLED
    duel.save(update_fields=["status"])
    _lobby_send(duel.opponent_id, {"type": "invite_cancelled", "duel_id": duel.pk})
    return JsonResponse({"ok": True})


@login_required
@require_POST
def decline_duel(request, pk):
    duel = get_object_or_404(Duel, pk=pk, opponent=request.user, status=Duel.STATUS_PENDING)
    duel.status = Duel.STATUS_DECLINED
    duel.save(update_fields=["status"])
    _lobby_send(duel.inviter_id, {"type": "invite_declined", "duel_id": duel.pk})
    return JsonResponse({"ok": True})


@login_required
@require_POST
def accept_duel(request, pk):
    duel = get_object_or_404(Duel, pk=pk, opponent=request.user, status=Duel.STATUS_PENDING)
    duel.status = Duel.STATUS_ACTIVE
    duel.started_at = timezone.now()
    duel.save(update_fields=["status", "started_at"])
    DuelProgress.objects.get_or_create(duel=duel, user=duel.inviter)
    DuelProgress.objects.get_or_create(duel=duel, user=duel.opponent)

    url = reverse("duel_detail", args=[duel.pk])
    _lobby_send(duel.inviter_id, {
        "type": "invite_accepted", "duel_id": duel.pk, "url": url,
    })
    return JsonResponse({"duel_id": duel.pk, "url": url})


@login_required
def duel_detail(request, pk):
    duel = get_object_or_404(Duel, pk=pk)
    if request.user.id not in (duel.inviter_id, duel.opponent_id):
        return redirect("puzzle_list")
    if duel.status != Duel.STATUS_ACTIVE and duel.status != Duel.STATUS_COMPLETED:
        return redirect("puzzle_list")

    puzzle = duel.puzzle
    db_combinations = list(puzzle.combinations.all())
    random.shuffle(db_combinations)

    words = []
    combinations = []
    for combo in db_combinations:
        try:
            ws = combo.wordsum
        except WordSum.DoesNotExist:
            continue
        idx = len(combinations)
        words.extend([
            {"text": ws.addend1, "combo": idx},
            {"text": ws.addend2, "combo": idx},
            {"text": ws.sum_word, "combo": idx},
        ])
        combinations.append({"id": combo.pk, "type": "word_sum"})
    random.shuffle(words)

    total = len(combinations)
    opponent = duel.other_player(request.user)
    my_progress = DuelProgress.objects.filter(duel=duel, user=request.user).first()
    opp_progress = DuelProgress.objects.filter(duel=duel, user=opponent).first()

    return render(request, "game/duel.html", {
        "duel": duel,
        "opponent": opponent,
        "puzzle": puzzle,
        "words_json": json.dumps(words),
        "combinations_json": json.dumps(combinations),
        "total": total,
        "my_count": my_progress.count if my_progress else 0,
        "opp_count": opp_progress.count if opp_progress else 0,
        "winner_username": duel.winner.username if duel.winner else "",
    })


@login_required
@require_POST
def duel_row_solved(request, pk):
    duel = get_object_or_404(Duel, pk=pk, status=Duel.STATUS_ACTIVE)
    if request.user.id not in (duel.inviter_id, duel.opponent_id):
        return JsonResponse({"error": "Not a participant."}, status=403)

    data = json.loads(request.body)
    slot_words = data.get("words") or []
    if len(slot_words) != 3:
        return JsonResponse({"error": "Invalid row."}, status=400)

    valid_keys = _duel_rowkeys(duel.puzzle)
    addends_set = frozenset(slot_words[:2])
    sum_word = slot_words[2]
    matched = None
    for k in valid_keys:
        if frozenset(k[0]) == addends_set and k[1] == sum_word:
            matched = k
            break
    if not matched:
        return JsonResponse({"ok": False, "reason": "incorrect"})

    progress, _ = DuelProgress.objects.get_or_create(duel=duel, user=request.user)
    # Dedupe on canonical key
    existing = {(tuple(row[0]), row[1]) for row in progress.solved_rows}
    key_tuple = (tuple(matched[0]), matched[1])
    if key_tuple not in existing:
        progress.solved_rows.append(matched)
        progress.save(update_fields=["solved_rows"])

    total = len(valid_keys)
    my_count = len(progress.solved_rows)

    finished = my_count >= total
    if finished:
        duel.status = Duel.STATUS_COMPLETED
        duel.winner = request.user
        duel.completed_at = timezone.now()
        duel.save(update_fields=["status", "winner", "completed_at"])
        PuzzleCompletion.objects.get_or_create(user=request.user, puzzle=duel.puzzle)

    _duel_send(duel.pk, {
        "type": "progress",
        "user_id": request.user.id,
        "username": request.user.username,
        "count": my_count,
        "total": total,
    })
    if finished:
        _duel_send(duel.pk, {
            "type": "duel_ended",
            "winner_id": request.user.id,
            "winner_username": request.user.username,
        })

    return JsonResponse({"ok": True, "count": my_count, "total": total, "finished": finished})

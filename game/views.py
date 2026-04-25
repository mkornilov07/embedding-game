import json
import random
import traceback

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

from .generators import available_local_models, generate_curated_word_sums, generate_word_sums
from .models import Duel, DuelProgress, Puzzle, PuzzleCompletion, WordSum


def puzzle_list(request):
    puzzles = list(Puzzle.objects.select_related("created_by").order_by("-pinned", "-pk"))
    has_duel_puzzles = any(p.for_duel for p in puzzles)
    # Dropdown in the duel form keeps duel-generated puzzles at the bottom.
    # Python's sort is stable, so within each group the -pk chronological order is preserved.
    dropdown_puzzles = sorted(puzzles, key=lambda p: p.for_duel)
    completed_ids = set()
    incoming_invites = []
    outgoing_invite = None
    active_duel = None
    recent_opponents = []
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
        recent_opponents = _recent_opponents(request.user)
    return render(request, "game/puzzle_list.html", {
        "puzzles": puzzles,
        "dropdown_puzzles": dropdown_puzzles,
        "completed_ids": completed_ids,
        "available_models": available_local_models(),
        "incoming_invites": incoming_invites,
        "outgoing_invite": outgoing_invite,
        "active_duel": active_duel,
        "has_duel_puzzles": has_duel_puzzles,
        "recent_opponents": recent_opponents,
    })


def _recent_opponents(user, limit=5):
    """Distinct opponents from this user's most recent completed duels, newest first."""
    duels = (
        Duel.objects.filter(status=Duel.STATUS_COMPLETED)
        .filter(Q(inviter=user) | Q(opponent=user))
        .select_related("inviter", "opponent")
        .order_by("-completed_at")[: limit * 3]
    )
    seen = set()
    names = []
    for d in duels:
        other = d.opponent if d.inviter_id == user.id else d.inviter
        if other.id in seen:
            continue
        seen.add(other.id)
        names.append(other.username)
        if len(names) >= limit:
            break
    return names


def _build_board_context(puzzle):
    """Shuffle a puzzle's combinations into the shape expected by board.js.

    Returns (words_json, combinations_json, total_combinations). `total` is the
    count of word-sum rows, useful for duel progress bars.
    """
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
    return json.dumps(words), json.dumps(combinations), len(combinations)


def puzzle(request, pk):
    p = get_object_or_404(Puzzle, pk=pk)
    words_json, combinations_json, _ = _build_board_context(p)
    return render(request, "game/puzzle.html", {
        "puzzle": p,
        "words_json": words_json,
        "combinations_json": combinations_json,
    })


@require_POST
def check_puzzle(request, pk):
    p = get_object_or_404(Puzzle, pk=pk)
    data = json.loads(request.body)

    unmatched = set(_puzzle_word_sums(p))
    for entry in data:
        slot_words = entry["words"]
        key = (frozenset(slot_words[:2]), slot_words[2])
        if key not in unmatched:
            return JsonResponse({"correct": False})
        unmatched.remove(key)

    correct = not unmatched
    if correct and request.user.is_authenticated:
        PuzzleCompletion.objects.get_or_create(user=request.user, puzzle=p)
    return JsonResponse({"correct": correct})


def _puzzle_word_sums(puzzle):
    """All (frozenset{addend1, addend2}, sum_word) tuples for a puzzle."""
    result = []
    for combo in puzzle.combinations.all():
        try:
            ws = combo.wordsum
        except WordSum.DoesNotExist:
            continue
        result.append((frozenset({ws.addend1, ws.addend2}), ws.sum_word))
    return result


def _wrong_slots(word_sums, slot_words):
    """Which slot indices of a 3-word row are wrong against a list of word sums.

    Slots are [addend1, addend2, sum]. If two of three match some sum, only the
    odd one out is returned; otherwise all three are wrong.
    """
    addends = frozenset(slot_words[:2])
    sum_word = slot_words[2]
    if (addends, sum_word) in word_sums:
        return []
    for ws_addends, ws_sum in word_sums:
        if addends == ws_addends:
            return [2]
    for ws_addends, ws_sum in word_sums:
        if sum_word == ws_sum:
            for i in range(2):
                if slot_words[i] in ws_addends:
                    return [1 - i]
    return [0, 1, 2]


@require_POST
def check_row(request, pk):
    p = get_object_or_404(Puzzle, pk=pk)
    data = json.loads(request.body)
    return JsonResponse({"wrong_slots": _wrong_slots(_puzzle_word_sums(p), data["words"])})


def register(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("puzzle_list")
    else:
        form = UserCreationForm()
    return render(request, "game/auth.html", {"form": form, "mode": "register"})


def _editable_puzzle_or_404(request, pk):
    """Owner can edit their puzzle; staff can edit anyone's."""
    if request.user.is_staff:
        return get_object_or_404(Puzzle, pk=pk)
    return get_object_or_404(Puzzle, pk=pk, created_by=request.user)


@login_required
def create_puzzle(request, pk=None):
    existing = None
    word_sums = []
    if pk is not None:
        existing = _editable_puzzle_or_404(request, pk)
        for combo in existing.combinations.all():
            try:
                ws = combo.wordsum
            except WordSum.DoesNotExist:
                continue
            word_sums.append({
                "addend1": ws.addend1, "addend2": ws.addend2, "sum_word": ws.sum_word,
            })
    return render(request, "game/create_puzzle.html", {
        "available_models": available_local_models(),
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
        puzzle = _editable_puzzle_or_404(request, puzzle_id)
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
    puzzle = _editable_puzzle_or_404(request, pk)
    puzzle.delete()
    return JsonResponse({"ok": True})


@login_required
@require_POST
def pin_puzzle(request, pk):
    if not request.user.is_staff:
        return JsonResponse({"error": "Admins only."}, status=403)
    puzzle = get_object_or_404(Puzzle, pk=pk)
    puzzle.pinned = not puzzle.pinned
    puzzle.save(update_fields=["pinned"])
    return JsonResponse({"pinned": puzzle.pinned})


@login_required
@require_POST
def generate_combinations(request):
    data = json.loads(request.body)
    gen_type = data.get("type", "word_sum")
    count = min(data.get("count", 5), 5)

    model_name = data.get("model")

    related_pairs = data.get("related_pairs", True)
    cosmul = data.get("cosmul", True)
    abtt = data.get("abtt", True)
    top_n_vocab = int(data.get("top_n_vocab") or 3000)
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
            except Exception as e:
                return JsonResponse(
                    {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()},
                    status=500,
                )
    else:
        return JsonResponse({"error": f"Unknown type: {gen_type}"}, status=400)

    return JsonResponse({"combinations": combos})


# ---------------------------- Duels ----------------------------

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
    words_json, combinations_json, total = _build_board_context(puzzle)
    opponent = duel.other_player(request.user)
    my_progress = DuelProgress.objects.filter(duel=duel, user=request.user).first()
    opp_progress = DuelProgress.objects.filter(duel=duel, user=opponent).first()

    return render(request, "game/duel.html", {
        "duel": duel,
        "opponent": opponent,
        "puzzle": puzzle,
        "words_json": words_json,
        "combinations_json": combinations_json,
        "total": total,
        "my_count": my_progress.count if my_progress else 0,
        "opp_count": opp_progress.count if opp_progress else 0,
        "winner_username": duel.winner.username if duel.winner else "",
    })


@login_required
@require_POST
def duel_surrender(request, pk):
    duel = get_object_or_404(Duel, pk=pk, status=Duel.STATUS_ACTIVE)
    if request.user.id not in (duel.inviter_id, duel.opponent_id):
        return JsonResponse({"error": "Not a participant."}, status=403)
    winner = duel.other_player(request.user)
    duel.status = Duel.STATUS_COMPLETED
    duel.winner = winner
    duel.completed_at = timezone.now()
    duel.save(update_fields=["status", "winner", "completed_at"])
    _duel_send(duel.pk, {
        "type": "duel_ended",
        "winner_id": winner.id,
        "winner_username": winner.username,
        "surrender": True,
        "loser_username": request.user.username,
    })
    return JsonResponse({"ok": True})


@login_required
@require_POST
def duel_row_solved(request, pk):
    """Same contract as check_row (returns wrong_slots) plus duel bookkeeping.

    If the row is fully correct, records it against the user's DuelProgress,
    broadcasts a progress event, and (if this was the final row) a duel_ended
    event. The response always includes wrong_slots so the duel UI can render
    the same slot-level feedback the stand-alone puzzle page gets.
    """
    duel = get_object_or_404(Duel, pk=pk, status=Duel.STATUS_ACTIVE)
    if request.user.id not in (duel.inviter_id, duel.opponent_id):
        return JsonResponse({"error": "Not a participant."}, status=403)

    data = json.loads(request.body)
    slot_words = data.get("words") or []
    if len(slot_words) != 3:
        return JsonResponse({"error": "Invalid row."}, status=400)

    word_sums = _puzzle_word_sums(duel.puzzle)
    wrong = _wrong_slots(word_sums, slot_words)
    total = len(word_sums)

    if wrong:
        progress = DuelProgress.objects.filter(duel=duel, user=request.user).first()
        return JsonResponse({
            "wrong_slots": wrong,
            "count": progress.count if progress else 0,
            "total": total,
            "finished": False,
        })

    # Correct row: dedupe into the user's solved set.
    matched = [sorted(slot_words[:2]), slot_words[2]]
    progress, _ = DuelProgress.objects.get_or_create(duel=duel, user=request.user)
    existing = {(tuple(row[0]), row[1]) for row in progress.solved_rows}
    if (tuple(matched[0]), matched[1]) not in existing:
        progress.solved_rows.append(matched)
        progress.save(update_fields=["solved_rows"])
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

    return JsonResponse({
        "wrong_slots": [],
        "count": my_count,
        "total": total,
        "finished": finished,
    })

import json
import math
import random
import traceback

from django.contrib.auth import get_user_model, login
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.db import transaction
from django.db.models import Count, Q
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_POST

from .generators import (
    AVAILABLE_MODELS,
    WORD_CHAIN_3K_MODEL,
    WORD_CHAIN_10K_MODEL,
    WORD_CHAIN_MAX_ATTEMPTS,
    generate_curated_word_sums, generate_word_chain,
    generate_word_sums, validate_word_chain,
)
from .models import (
    Duel, DuelProgress, PreferenceLabelExample, Puzzle, PuzzleCompletion,
    WordChain, WordChainSlot, WordSum,
)
from . import services
from . import calibration


def puzzle_list(request):
    puzzles = list(
        Puzzle.objects.select_related("created_by")
        .annotate(solver_count=Count("completions__user", distinct=True))
        .order_by("-pinned", "-pk")
    )
    has_duel_puzzles = any(p.for_duel for p in puzzles)
    has_daily_puzzles = any(p.is_daily for p in puzzles)
    has_user_puzzles = any(
        p.created_by_id and not p.is_daily and not p.for_duel for p in puzzles
    )
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
            PuzzleCompletion.objects.filter(
                user=request.user,
                first_solved_at__isnull=False,
            ).values_list("puzzle_id", flat=True)
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
        "available_models": AVAILABLE_MODELS,
        "incoming_invites": incoming_invites,
        "outgoing_invite": outgoing_invite,
        "active_duel": active_duel,
        "has_duel_puzzles": has_duel_puzzles,
        "has_daily_puzzles": has_daily_puzzles,
        "has_user_puzzles": has_user_puzzles,
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


def puzzle(request, pk):
    p = get_object_or_404(Puzzle, pk=pk)
    services.record_puzzle_open(request.user, p)
    words_json, combinations_json, _ = services.board_context(p)
    return render(request, "game/puzzle.html", {
        "puzzle": p,
        "words_json": words_json,
        "combinations_json": combinations_json,
    })


@require_POST
def check_puzzle(request, pk):
    p = get_object_or_404(Puzzle, pk=pk)
    data = json.loads(request.body)
    entries = {entry.get("id"): entry.get("words", []) for entry in data}
    combinations = services.puzzle_combinations(p)
    correct = len(entries) == len(data) == len(combinations)
    unmatched_sums = []
    for combination in combinations:
        try:
            word_sum = combination.wordsum
        except WordSum.DoesNotExist:
            if combination.pk not in entries or not services.combination_is_solved(
                combination, entries[combination.pk],
            ):
                correct = False
        else:
            unmatched_sums.append((
                frozenset((word_sum.addend1, word_sum.addend2)), word_sum.sum_word,
            ))
    for combination in combinations:
        try:
            combination.wordsum
        except WordSum.DoesNotExist:
            continue
        slot_words = entries.get(combination.pk, [])
        key = (frozenset(slot_words[:2]), slot_words[2]) if len(slot_words) == 3 else None
        if key not in unmatched_sums:
            correct = False
        else:
            unmatched_sums.remove(key)
    correct = correct and not unmatched_sums
    if correct:
        services.mark_puzzle_completed(request.user, p)
    return JsonResponse({"correct": correct})


@require_POST
def check_row(request, pk):
    p = get_object_or_404(Puzzle, pk=pk)
    data = json.loads(request.body)
    services.record_hint_used(request.user, p)
    slot_words = data.get("words", [])
    if not isinstance(slot_words, list):
        return JsonResponse({"error": "Invalid row."}, status=400)
    combination_id = data.get("combination_id")
    if combination_id is None:
        green, yellow = services.check_word_sum_row(services.puzzle_word_sums(p), slot_words)
        invalid_links = []
    else:
        combination = get_object_or_404(p.combinations, pk=combination_id)
        try:
            chain = combination.wordchain
        except WordChain.DoesNotExist:
            green, yellow = services.check_word_sum_row(services.puzzle_word_sums(p), slot_words)
            invalid_links = []
        else:
            green, yellow, invalid_links = services.check_word_chain(chain, slot_words)
    return JsonResponse({
        "green_slots": green,
        "yellow_slots": yellow,
        "invalid_links": invalid_links,
    })


def puzzle_leaderboard(request, pk):
    puzzle = get_object_or_404(Puzzle, pk=pk)
    completions = (
        PuzzleCompletion.objects.filter(
            puzzle=puzzle,
            first_opened_at__isnull=False,
            first_solved_at__isnull=False,
        )
        .select_related("user")
        .order_by("first_solved_at")
    )
    leaderboard = [
        {
            "username": completion.user.username,
            "duration": completion.first_solved_at - completion.first_opened_at,
            "used_hints": completion.used_hints,
        }
        for completion in completions
    ]
    leaderboard.sort(key=lambda entry: entry["duration"])
    return render(request, "game/leaderboard.html", {
        "puzzle": puzzle,
        "leaderboard": leaderboard,
    })


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
    defaults = calibration.get_defaults()
    chain_defaults = calibration.get_chain_defaults()
    models = AVAILABLE_MODELS
    defaults_by_model = {
        model_name: calibration.as_dict(calibration.get_defaults(model_name))
        for model_name in models
    }
    existing = None
    initial_combinations = []
    if pk is not None:
        existing = _editable_puzzle_or_404(request, pk)
        for combo in services.puzzle_combinations(existing):
            try:
                ws = combo.wordsum
            except WordSum.DoesNotExist:
                chain = services.word_chain_for(combo)
                if chain is not None:
                    initial_combinations.append({
                        "type": "word_chain",
                        "words": [
                            {"word": slot.word, "kind": slot.kind}
                            for slot in chain.slots.all()
                        ],
                    })
            else:
                initial_combinations.append({
                    "type": "word_sum",
                    "addend1": ws.addend1,
                    "addend2": ws.addend2,
                    "sum_word": ws.sum_word,
                })
    return render(request, "game/create_puzzle.html", {
        "available_models": models,
        "generation_defaults": calibration.as_dict(defaults),
        "generation_defaults_by_model": defaults_by_model,
        "puzzle": existing,
        "combinations_json": json.dumps(initial_combinations),
        "chain_min_similarity": chain_defaults.min_previous_similarity,
    })


@login_required
@require_POST
@transaction.atomic
def save_puzzle(request):
    data = json.loads(request.body)
    name = data.get("name", "").strip()
    combinations = data.get("combinations", [])
    puzzle_id = data.get("id")

    if not name or not combinations:
        return JsonResponse({"error": "Name and at least one word combination required."}, status=400)

    normalized = []
    chain_defaults = calibration.get_chain_defaults()
    try:
        for index, combination in enumerate(combinations, start=1):
            if combination.get("type") == "word_sum":
                words = [
                    combination["addend1"].strip(),
                    combination["addend2"].strip(),
                    combination["sum_word"].strip(),
                ]
                if not all(words):
                    raise ValueError(f"Word sum {index}: fill in all three words.")
                normalized.append(("word_sum", words))
            elif combination.get("type") == "word_chain":
                slots = combination.get("words", [])
                words = [slot.get("word", "").strip() for slot in slots]
                if len(words) < 2 or not all(words):
                    raise ValueError(f"Word chain {index}: enter at least two words.")
                if not validate_word_chain(
                    words, model_name=chain_defaults.model_name,
                    min_previous_similarity=chain_defaults.min_previous_similarity,
                )["valid"]:
                    raise ValueError(
                        f"Word chain {index}: every word must be in the ConceptNet dictionary, "
                        f"and adjacent words need cosine similarity of at least "
                        f"{chain_defaults.min_previous_similarity}."
                    )
                kinds = [slot.get("kind") for slot in slots]
                allowed_kinds = {choice for choice, _ in WordChainSlot.KIND_CHOICES}
                if any(kind not in allowed_kinds for kind in kinds):
                    raise ValueError(f"Word chain {index}: choose an interaction for every word.")
                normalized.append(("word_chain", list(zip(words, kinds))))
            else:
                raise ValueError(f"Combination {index}: unknown type.")
    except (KeyError, AttributeError, TypeError, ValueError) as error:
        return JsonResponse({"error": str(error)}, status=400)
    except FileNotFoundError:
        return JsonResponse({
            "error": "Missing data/conceptnet-numberbatch-en-10k-abtt.kv."
        }, status=400)

    if puzzle_id is not None:
        puzzle = _editable_puzzle_or_404(request, puzzle_id)
        puzzle.name = name
        puzzle.save()
        puzzle.combinations.all().delete()
    else:
        puzzle = Puzzle.objects.create(name=name, created_by=request.user)
    for combination_type, words in normalized:
        if combination_type == "word_sum":
            WordSum.objects.create(
                puzzle=puzzle,
                addend1=words[0],
                addend2=words[1],
                sum_word=words[2],
            )
            continue
        chain = WordChain.objects.create(puzzle=puzzle)
        WordChainSlot.objects.bulk_create([
            WordChainSlot(chain=chain, position=position, word=word, kind=kind)
            for position, (word, kind) in enumerate(words)
        ])

    return JsonResponse({"id": puzzle.pk})


@login_required
@require_POST
def validate_word_chain_view(request):
    try:
        words = json.loads(request.body).get("words", [])
        defaults = calibration.get_chain_defaults()
        result = validate_word_chain(
            words, model_name=defaults.model_name,
            min_previous_similarity=defaults.min_previous_similarity,
        )
        if not request.user.is_staff:
            result.pop("similarities")
        return JsonResponse(result)
    except FileNotFoundError:
        return JsonResponse({
            "error": "Missing data/conceptnet-numberbatch-en-10k-abtt.kv."
        }, status=400)


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


@staff_member_required
def debug_tools(request):
    return render(request, "game/debug_tools.html", {
        "word_chain_count": PreferenceLabelExample.objects.filter(
            example_type=PreferenceLabelExample.WORD_CHAIN,
        ).count(),
    })


@staff_member_required
def calibration_view(request):
    model_name = request.GET.get("model")
    chain_model = request.GET.get("chain_model")
    if model_name not in AVAILABLE_MODELS:
        model_name = None
    if chain_model not in AVAILABLE_MODELS:
        chain_model = None
    defaults = calibration.get_defaults(model_name)
    chain_defaults = calibration.get_chain_defaults(chain_model)
    models = AVAILABLE_MODELS
    return render(request, "game/calibration.html", {
        "parameters": calibration.as_dict(defaults),
        "available_models": models,
        "refine_options": (0, 1, 2, 3),
        "chain_parameters": calibration.chain_as_dict(chain_defaults),
        "sum_history": calibration.parameter_history("word_sum", defaults.model_name),
        "chain_history": calibration.parameter_history("word_chain", chain_defaults.model_name),
    })


@staff_member_required
def preference_labeling_view(request):
    model_name = request.GET.get("model")
    chain_model = request.GET.get("chain_model")
    if model_name not in AVAILABLE_MODELS:
        model_name = None
    if chain_model not in AVAILABLE_MODELS:
        chain_model = None
    defaults = calibration.get_defaults(model_name)
    chain_defaults = calibration.get_chain_defaults(chain_model)
    queue = [
        {"id": item.pk, "example_type": item.example_type, "words": item.words}
        for item in PreferenceLabelExample.objects.filter(label__isnull=True)[:100]
    ]
    return render(request, "game/preference_labeling.html", {
        "parameters": calibration.as_dict(defaults),
        "chain_parameters": calibration.chain_as_dict(chain_defaults),
        "available_models": AVAILABLE_MODELS,
        "refine_options": (0, 1, 2, 3),
        "preference_queue": queue,
        "word_chain_count": PreferenceLabelExample.objects.filter(
            example_type=PreferenceLabelExample.WORD_CHAIN,
        ).count(),
    })


PREFERENCE_LABEL_BATCH_MAX = 50


def _preference_example_json(item):
    return {
        "id": item.pk,
        "example_type": item.example_type,
        "words": item.words,
    }


@staff_member_required
@require_POST
def generate_preference_examples(request):
    try:
        data = json.loads(request.body)
        count = int(data.get("count", 20))
        if not 1 <= count <= PREFERENCE_LABEL_BATCH_MAX:
            raise ValueError(f"Count must be between 1 and {PREFERENCE_LABEL_BATCH_MAX}.")

        kind = data.get("kind")
        if kind == PreferenceLabelExample.WORD_CHAIN:
            model_name = data.get("model_name")
            if model_name not in AVAILABLE_MODELS:
                raise ValueError("Choose a valid word-chain model.")
            stdev = float(data.get("stdev", 0.1))
            if not math.isfinite(stdev) or stdev < 0:
                raise ValueError("Standard deviation must be a non-negative number.")
            base_parameters = {
                "start_percent": float(data["start_percent"]),
                "min_previous_similarity": float(data["min_previous_similarity"]),
                "max_earlier_similarity": float(data["max_earlier_similarity"]),
            }
            candidates = []
            for _ in range(count):
                parameters = {
                    field: value + random.gauss(0, stdev) * (100 if field == "start_percent" else 1)
                    for field, value in base_parameters.items()
                }
                parameters["start_percent"] = min(100, max(0.01, parameters["start_percent"]))
                for field in ("min_previous_similarity", "max_earlier_similarity"):
                    parameters[field] = min(1, max(-1, parameters[field]))
                words = generate_word_chain(
                    length=3,
                    **parameters,
                    model_name=model_name,
                )
                if words:
                    candidates.append(words)
        elif kind == PreferenceLabelExample.WORD_SUM:
            model_name = data.get("model_name")
            if model_name not in AVAILABLE_MODELS:
                raise ValueError("Choose a valid word-sum model.")
            parameters = {
                "model_name": model_name,
                "related_pairs": data.get("related_pairs") is True,
                "sum_scoring_method": data.get("sum_scoring_method"),
                "refine_iterations": int(data["refine_iterations"]),
                "min_similarity": float(data["min_similarity"]),
                "min_refined_similarity": float(data["min_refined_similarity"]),
                "min_gap_ratio": float(data["min_gap_ratio"]),
                "max_synonym_similarity": float(data["max_synonym_similarity"]),
            }
            candidates = [
                [candidate["addend1"], candidate["addend2"], candidate["sum_word"]]
                for candidate in generate_word_sums(count=count, **parameters)
            ]
        else:
            raise ValueError("Choose word chains or word sums.")

        created = [
            PreferenceLabelExample.objects.create(
                example_type=kind,
                words=candidate,
            )
            for candidate in candidates
        ]
        return JsonResponse({
            "examples": [_preference_example_json(item) for item in created],
            "requested": count,
        })
    except (KeyError, TypeError, ValueError) as error:
        return JsonResponse({"error": str(error)}, status=400)
    except FileNotFoundError:
        return JsonResponse({"error": "The selected model file is unavailable."}, status=400)


@staff_member_required
@require_POST
def label_preference_example(request, pk):
    try:
        data = json.loads(request.body)
        label = data.get("label")
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError("Label must be 1 (good) or 0 (bad).")
        item = get_object_or_404(PreferenceLabelExample, pk=pk)
        item.label = label
        item.save(update_fields=("label",))
        return JsonResponse({"ok": True, "label": label})
    except (json.JSONDecodeError, ValueError) as error:
        return JsonResponse({"error": str(error)}, status=400)


@staff_member_required
def export_preference_labels(request):
    examples = PreferenceLabelExample.objects.filter(label__isnull=False).order_by("pk")
    lines = (
        json.dumps({
            "type": item.example_type,
            "words": item.words,
            "label": item.label,
        }, separators=(",", ":"))
        for item in examples
    )
    response = HttpResponse("\n".join(lines), content_type="application/x-ndjson")
    response["Content-Disposition"] = 'attachment; filename="preference-labels.jsonl"'
    return response


@staff_member_required
@require_POST
def start_calibration(request):
    try:
        data = json.loads(request.body)
        if data.get("model_name") not in AVAILABLE_MODELS:
            raise ValueError("Unknown embedding model.")
        defaults = calibration.get_defaults(data["model_name"])
        parameters = calibration.as_dict(defaults)
        parameters.update({
            "model_name": defaults.model_name,
            "related_pairs": bool(data["related_pairs"]),
            "sum_scoring_method": data.get(
                "sum_scoring_method",
                "cosmul" if data.get("cosmul", True) else "cosadd",
            ),
            "refine_iterations": int(data["refine_iterations"]),
        })
        noise_stdev = float(data.get("noise_stdev", calibration.NOISE_STDEV))
        learning_rate = float(data.get("learning_rate", calibration.LEARNING_RATE))
        round_record = calibration.generate_round(
            request.user, parameters, noise_stdev, learning_rate,
        )
        calibration.CalibrationClick.objects.create(
            round=round_record,
            event="round_started",
            payload={
                "parameters": parameters,
                "noise_stdev": noise_stdev,
                "learning_rate": learning_rate,
            },
        )
        return JsonResponse(_calibration_round_payload(round_record))
    except (KeyError, TypeError, ValueError) as error:
        return JsonResponse({"error": str(error)}, status=400)


@staff_member_required
@require_POST
def select_calibration_candidate(request, pk):
    try:
        data = json.loads(request.body)
        round_record = calibration.select_candidate(
            request.user, pk, int(data["candidate_index"]),
        )
        return JsonResponse(_calibration_round_payload(round_record))
    except calibration.CalibrationRound.DoesNotExist:
        return JsonResponse({"error": "Calibration round not found."}, status=404)
    except (KeyError, TypeError, ValueError) as error:
        return JsonResponse({"error": str(error)}, status=400)


@staff_member_required
@require_POST
def start_chain_calibration(request):
    try:
        data = json.loads(request.body)
        selected_model = data.get("model_name")
        if not selected_model:
            selected_model = {
                "3k": WORD_CHAIN_3K_MODEL, "10k": WORD_CHAIN_10K_MODEL,
            }.get(data.get("vocabulary"))
        if selected_model not in AVAILABLE_MODELS:
            raise ValueError("Unknown embedding model.")
        parameters = calibration.chain_as_dict(calibration.get_chain_defaults(selected_model))
        parameters.update({
            "model_name": selected_model,
            "start_percent": float(data["start_percent"]),
            "min_previous_similarity": float(data["min_previous_similarity"]),
            "max_earlier_similarity": float(data["max_earlier_similarity"]),
        })
        round_record = calibration.generate_chain_round(
            request.user, parameters, length=int(data["length"]),
            noise_stdev=float(data.get("noise_stdev", calibration.NOISE_STDEV)),
            learning_rate=float(data.get("learning_rate", calibration.LEARNING_RATE)),
        )
        return JsonResponse(_calibration_round_payload(round_record))
    except (KeyError, TypeError, ValueError) as error:
        return JsonResponse({"error": str(error)}, status=400)


@staff_member_required
@require_POST
def select_chain_calibration_candidate(request, pk):
    try:
        data = json.loads(request.body)
        round_record = calibration.select_chain_candidate(
            request.user, pk, int(data["candidate_index"]),
        )
        return JsonResponse(_calibration_round_payload(round_record))
    except calibration.CalibrationRound.DoesNotExist:
        return JsonResponse({"error": "Calibration round not found."}, status=404)
    except (KeyError, TypeError, ValueError) as error:
        return JsonResponse({"error": str(error)}, status=400)


def _calibration_round_payload(round_record):
    return {
        "round_id": round_record.pk,
        "parameters": round_record.parameters_before,
        "noise_stdev": round_record.noise_stdev,
        "learning_rate": round_record.learning_rate,
        "noise_vectors": round_record.noise_vectors,
        "candidates": round_record.candidates,
        "history": calibration.parameter_history(
            round_record.kind, round_record.parameters_before.get("model_name"),
        ),
    }


@login_required
@require_POST
def generate_combinations(request):
    data = json.loads(request.body)
    gen_type = data.get("type", "word_sum")
    try:
        count = max(1, min(int(data.get("count", 5)), 5))
    except (TypeError, ValueError):
        return JsonResponse({"error": "Count must be a number from 1 to 5."}, status=400)

    model_name = data.get("model")

    related_pairs = data.get("related_pairs", True)
    sum_scoring_method = data.get("sum_scoring_method")
    if sum_scoring_method is None and "cosmul" in data:
        sum_scoring_method = "cosmul" if data["cosmul"] else "cosadd"
    refine_iterations = int(data.get("refine_iterations", 1))
    min_similarity = data.get("min_similarity")
    min_refined_similarity = data.get("min_refined_similarity")
    min_gap_ratio = data.get("min_gap_ratio")
    max_synonym_similarity = data.get("max_synonym_similarity")

    if gen_type == "word_chain":
        chain_defaults = calibration.get_chain_defaults()
        if any(key in data for key in (
            "min_previous_similarity", "max_earlier_similarity", "vocabulary", "model_name", "start_percent",
        )) and not request.user.is_staff:
            return JsonResponse({"error": "Admins only."}, status=403)
        try:
            vocabulary = data.get("vocabulary", "standard")
            model_aliases = {
                "standard": chain_defaults.model_name,
                "3k": WORD_CHAIN_3K_MODEL,
                "10k": WORD_CHAIN_10K_MODEL,
            }
            model_name = data.get("model_name") or model_aliases.get(
                vocabulary, vocabulary if vocabulary in AVAILABLE_MODELS else None,
            )
            if model_name is None:
                raise ValueError("Unknown word-chain vocabulary.")
            selected_chain_defaults = calibration.get_chain_defaults(model_name)
            chain_length = int(data.get("length", 5))
            if chain_length < 2:
                raise ValueError("A word chain must contain at least two words.")
            chain = generate_word_chain(
                chain_length,
                min_previous_similarity=data.get(
                    "min_previous_similarity", selected_chain_defaults.min_previous_similarity,
                ),
                max_earlier_similarity=data.get(
                    "max_earlier_similarity", selected_chain_defaults.max_earlier_similarity,
                ),
                model_name=model_name,
                start_percent=data.get("start_percent", selected_chain_defaults.start_percent),
            )
        except (TypeError, ValueError) as error:
            return JsonResponse({"error": str(error)}, status=400)
        except FileNotFoundError:
            return JsonResponse({
                "error": f"Missing data/{model_name}.kv.",
            }, status=400)
        if chain is None:
            return JsonResponse({"error": f"Could not generate a chain after {WORD_CHAIN_MAX_ATTEMPTS} attempts."}, status=400)
        result = {"combinations": [{
            "type": "word_chain",
            "words": [{"word": word, "kind": WordChainSlot.DRAGGABLE} for word in chain],
        }]}
        if request.user.is_staff:
            preview = validate_word_chain(chain, model_name=model_name,
                                          min_previous_similarity=selected_chain_defaults.min_previous_similarity)
            puzzle_valid = validate_word_chain(
                chain, model_name=chain_defaults.model_name,
                min_previous_similarity=chain_defaults.min_previous_similarity,
            )["valid"]
            result.update(similarities=preview["similarities"], puzzle_valid=puzzle_valid)
        return JsonResponse(result)

    if gen_type == "word_sum":
        if model_name == "curated":
            combos = generate_curated_word_sums(count)
        else:
            if model_name is not None and model_name not in AVAILABLE_MODELS:
                return JsonResponse({"error": "Unknown embedding model."}, status=400)
            kwargs = dict(
                model_name=model_name, related_pairs=related_pairs,
                sum_scoring_method=sum_scoring_method,
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

@login_required
@require_POST
def create_duel(request):
    if services.user_has_open_duel(request.user):
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

    if services.user_has_open_duel(opponent):
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

    services.send_lobby_event(opponent.id, {
        "type": "invite_received",
        "duel": services.duel_summary(duel),
    })
    return JsonResponse({"duel_id": duel.pk})


@login_required
@require_POST
def cancel_duel(request, pk):
    duel = get_object_or_404(Duel, pk=pk, inviter=request.user, status=Duel.STATUS_PENDING)
    duel.status = Duel.STATUS_CANCELLED
    duel.save(update_fields=["status"])
    services.send_lobby_event(duel.opponent_id, {"type": "invite_cancelled", "duel_id": duel.pk})
    return JsonResponse({"ok": True})


@login_required
@require_POST
def decline_duel(request, pk):
    duel = get_object_or_404(Duel, pk=pk, opponent=request.user, status=Duel.STATUS_PENDING)
    duel.status = Duel.STATUS_DECLINED
    duel.save(update_fields=["status"])
    services.send_lobby_event(duel.inviter_id, {"type": "invite_declined", "duel_id": duel.pk})
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
    services.send_lobby_event(duel.inviter_id, {
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
    words_json, combinations_json, total = services.board_context(puzzle)
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
    services.send_duel_event(duel.pk, {
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
    """Same contract as check_row (returns green_slots/yellow_slots) plus duel bookkeeping.

    If the row is fully correct, records it against the user's DuelProgress
    and (if this was the final row) marks the duel completed.
    """
    duel = get_object_or_404(Duel, pk=pk, status=Duel.STATUS_ACTIVE)
    if request.user.id not in (duel.inviter_id, duel.opponent_id):
        return JsonResponse({"error": "Not a participant."}, status=403)

    data = json.loads(request.body)
    slot_words = data.get("words") or []
    combination_id = data.get("combination_id")
    if combination_id is None:
        return JsonResponse({"error": "Invalid row."}, status=400)
    combination = get_object_or_404(duel.puzzle.combinations, pk=combination_id)
    total = len(services.puzzle_combinations(duel.puzzle))
    slot_count = 3
    try:
        chain = combination.wordchain
    except WordChain.DoesNotExist:
        if len(slot_words) != 3:
            return JsonResponse({"error": "Invalid row."}, status=400)
        green, yellow = services.check_word_sum_row(
            services.puzzle_word_sums(duel.puzzle), slot_words,
        )
        invalid_links = []
    else:
        green, yellow, invalid_links = services.check_word_chain(chain, slot_words)
        slot_count = chain.slots.count()

    progress = DuelProgress.objects.filter(duel=duel, user=request.user).first()
    my_count = progress.count if progress else 0
    finished = False

    if len(green) == slot_count and not invalid_links:
        my_count, finished = services.record_solved_combination(
            duel, request.user, combination, slot_words, total,
        )

        services.send_duel_event(duel.pk, {
            "type": "progress",
            "user_id": request.user.id,
            "username": request.user.username,
            "count": my_count,
            "total": total,
        })
        if finished:
            services.send_duel_event(duel.pk, {
                "type": "duel_ended",
                "winner_id": request.user.id,
                "winner_username": request.user.username,
            })

    return JsonResponse({
        "green_slots": green,
        "yellow_slots": yellow,
        "invalid_links": invalid_links,
        "count": my_count,
        "total": total,
        "finished": finished,
    })

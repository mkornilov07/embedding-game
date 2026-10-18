"""Calibration workflow for the database-backed generation defaults."""

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from django.db import transaction
from django.utils import timezone

from .generators import (
    DEFAULT_MODEL,
    DEFAULT_TOP_N_VOCAB,
    DEFAULT_SUM_SCORING_METHOD,
    MAX_SYNONYM_SIMILARITY,
    MIN_GAP_RATIO,
    MIN_REFINED_SIMILARITY_COSMUL,
    MIN_SIMILARITY_COSMUL,
    REFINE_ITERATIONS,
    WORD_CHAIN_MODEL, WORD_CHAIN_START_PERCENT, WORD_CHAIN_MIN_SIMILARITY,
    WORD_CHAIN_MAX_EARLIER_SIMILARITY, generate_word_chain,
    generate_word_sums,
)
from .models import CalibrationClick, CalibrationRound, GenerationDefaults, WordChainDefaults


CONTINUOUS_PARAMETERS = (
    "min_similarity",
    "min_refined_similarity",
    "min_gap_ratio",
    "max_synonym_similarity",
)
CALIBRATION_COUNT = 9
CALIBRATION_MAX_ATTEMPTS = 200
NOISE_STDEV = 0.1
LEARNING_RATE = 0.1
CHAIN_PARAMETERS = ("start_percent", "min_previous_similarity", "max_earlier_similarity")


def parameter_history(kind):
    """Values committed by each calibration choice, oldest first."""
    fields = CHAIN_PARAMETERS if kind == "word_chain" else CONTINUOUS_PARAMETERS
    clicks = (
        CalibrationClick.objects.filter(round__kind=kind, event="candidate_selected")
        .select_related("round__calibrator")
        .order_by("created_at", "pk")
    )
    return [
        {
            "at": click.created_at.isoformat(),
            "calibrator": click.round.calibrator.username,
            "parameters": {field: values[field] for field in fields},
        }
        for click in clicks
        if (values := click.payload.get("parameters_after"))
        and all(field in values for field in fields)
    ]


def get_defaults():
    defaults, _ = GenerationDefaults.objects.get_or_create(
        key="default",
        defaults={
            "model_name": DEFAULT_MODEL,
            "sum_scoring_method": DEFAULT_SUM_SCORING_METHOD,
            "top_n_vocab": DEFAULT_TOP_N_VOCAB,
            "refine_iterations": REFINE_ITERATIONS,
            "min_similarity": MIN_SIMILARITY_COSMUL,
            "min_refined_similarity": MIN_REFINED_SIMILARITY_COSMUL,
            "min_gap_ratio": MIN_GAP_RATIO,
            "max_synonym_similarity": MAX_SYNONYM_SIMILARITY,
        },
    )
    return defaults


def as_dict(defaults):
    return {
        field: getattr(defaults, field)
        for field in (
            "model_name", "related_pairs", "sum_scoring_method", "abtt", "top_n_vocab",
            "refine_iterations", *CONTINUOUS_PARAMETERS,
        )
    }


def get_chain_defaults():
    defaults, _ = WordChainDefaults.objects.get_or_create(
        key="default",
        defaults={
            "model_name": WORD_CHAIN_MODEL,
            "start_percent": WORD_CHAIN_START_PERCENT,
            "min_previous_similarity": WORD_CHAIN_MIN_SIMILARITY,
            "max_earlier_similarity": WORD_CHAIN_MAX_EARLIER_SIMILARITY,
        },
    )
    return defaults


def chain_as_dict(defaults):
    return {field: getattr(defaults, field) for field in ("model_name", *CHAIN_PARAMETERS)}


def _validate_chain_parameters(parameters):
    if not math.isfinite(parameters["start_percent"]) or not 0 < parameters["start_percent"] <= 100:
        raise ValueError("Starting vocabulary percentage must be greater than 0 and at most 100.")
    for field in CHAIN_PARAMETERS[1:]:
        if not math.isfinite(parameters[field]) or not -1 <= parameters[field] <= 1:
            raise ValueError("Similarity thresholds must be between -1 and 1.")


def _chain_with_noise(parameters, noise):
    result = dict(parameters)
    for field, offset in zip(CHAIN_PARAMETERS, noise):
        result[field] += offset * 100 if field == "start_percent" else offset
    result["start_percent"] = min(100, max(0.01, result["start_percent"]))
    for field in CHAIN_PARAMETERS[1:]:
        result[field] = min(1, max(-1, result[field]))
    return result


def generate_chain_round(user, parameters, length=5, noise_stdev=NOISE_STDEV,
                         learning_rate=LEARNING_RATE):
    if not math.isfinite(noise_stdev) or noise_stdev < 0:
        raise ValueError("Noise standard deviation must be non-negative.")
    if not math.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError("Learning rate must be non-negative.")
    if length < 2:
        raise ValueError("A word chain must contain at least two words.")
    _validate_chain_parameters(parameters)
    noise_vectors = [
        [0.0] * len(CHAIN_PARAMETERS),
        *np.random.normal(0, noise_stdev, (CALIBRATION_COUNT - 1, len(CHAIN_PARAMETERS))).tolist(),
    ]

    def generate_candidate(noise):
        settings = _chain_with_noise(parameters, noise)
        try:
            chain = generate_word_chain(length, **settings)
            return {
                "available": bool(chain),
                "combination": chain,
                "error": None if chain else "No word chain could be generated.",
            }
        except Exception as error:
            return {"available": False, "combination": None,
                    "error": f"{type(error).__name__}: {error}"}

    with ThreadPoolExecutor(max_workers=min(4, len(noise_vectors))) as executor:
        candidates = list(executor.map(generate_candidate, noise_vectors))
    return CalibrationRound.objects.create(
        calibrator=user, kind="word_chain", chain_length=length, parameters_before=parameters,
        noise_stdev=noise_stdev, learning_rate=learning_rate,
        noise_vectors=noise_vectors, candidates=candidates,
    )


@transaction.atomic
def select_chain_candidate(user, round_id, candidate_index):
    round_record = CalibrationRound.objects.select_for_update().get(
        pk=round_id, calibrator=user, kind="word_chain",
    )
    if round_record.selected_index is not None:
        raise ValueError("This calibration round has already been selected.")
    if not 0 <= candidate_index < len(round_record.candidates):
        raise ValueError("Invalid calibration choice.")
    if not round_record.candidates[candidate_index]["available"]:
        raise ValueError("That calibration choice is unavailable.")
    noise = [value * round_record.learning_rate
             for value in round_record.noise_vectors[candidate_index]]
    parameters_after = _chain_with_noise(round_record.parameters_before, noise)
    defaults = get_chain_defaults()
    for field, value in parameters_after.items():
        setattr(defaults, field, value)
    defaults.save()
    round_record.selected_index = candidate_index
    round_record.parameters_after = parameters_after
    round_record.selected_at = timezone.now()
    round_record.save(update_fields=["selected_index", "parameters_after", "selected_at"])
    CalibrationClick.objects.create(
        round=round_record, event="candidate_selected", candidate_index=candidate_index,
        payload={"parameters_after": parameters_after},
    )
    return generate_chain_round(user, parameters_after, round_record.chain_length,
                                round_record.noise_stdev, round_record.learning_rate)


def generate_round(user, parameters, noise_stdev=NOISE_STDEV, learning_rate=LEARNING_RATE):
    if not math.isfinite(noise_stdev) or noise_stdev < 0:
        raise ValueError("Noise standard deviation must be non-negative.")
    if not math.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError("Learning rate must be non-negative.")
    noise_vectors = [
        [0.0] * len(CONTINUOUS_PARAMETERS),
        *np.random.multivariate_normal(
            mean=np.zeros(len(CONTINUOUS_PARAMETERS)),
            cov=noise_stdev**2 * np.eye(len(CONTINUOUS_PARAMETERS)),
            size=CALIBRATION_COUNT - 1,
        ).tolist(),
    ]
    def generate_candidate(noise):
        candidate_parameters = _with_noise(parameters, noise)
        try:
            combinations = generate_word_sums(
                count=1,
                max_attempts_per_result=CALIBRATION_MAX_ATTEMPTS,
                **candidate_parameters,
            )
            return {
                "available": bool(combinations),
                "combination": combinations[0] if combinations else None,
                "error": None if combinations else "No word sum could be generated.",
            }
        except Exception as error:
            return {
                "available": False,
                "combination": None,
                "error": f"{type(error).__name__}: {error}",
            }

    with ThreadPoolExecutor(max_workers=min(4, len(noise_vectors))) as executor:
        candidates = list(executor.map(generate_candidate, noise_vectors))

    round_record = CalibrationRound.objects.create(
        calibrator=user,
        parameters_before=parameters,
        noise_stdev=noise_stdev,
        learning_rate=learning_rate,
        noise_vectors=noise_vectors,
        candidates=candidates,
    )
    return round_record


@transaction.atomic
def select_candidate(user, round_id, candidate_index):
    round_record = CalibrationRound.objects.select_for_update().get(
        pk=round_id,
        calibrator=user,
        kind="word_sum",
    )
    if round_record.selected_index is not None:
        raise ValueError("This calibration round has already been selected.")
    if not 0 <= candidate_index < len(round_record.candidates):
        raise ValueError("Invalid calibration choice.")
    if not round_record.candidates[candidate_index]["available"]:
        raise ValueError("That calibration choice is unavailable.")

    noise = round_record.noise_vectors[candidate_index]
    parameters_after = _with_noise(
        round_record.parameters_before,
        [value * round_record.learning_rate for value in noise],
    )
    defaults = get_defaults()
    for field, value in parameters_after.items():
        setattr(defaults, field, value)
    defaults.save()

    round_record.selected_index = candidate_index
    round_record.parameters_after = parameters_after
    round_record.selected_at = timezone.now()
    round_record.save(update_fields=[
        "selected_index", "parameters_after", "selected_at",
    ])
    CalibrationClick.objects.create(
        round=round_record,
        event="candidate_selected",
        candidate_index=candidate_index,
        payload={"parameters_after": parameters_after},
    )
    return generate_round(
        user,
        parameters_after,
        round_record.noise_stdev,
        round_record.learning_rate,
    )


def _with_noise(parameters, noise):
    result = dict(parameters)
    for field, offset in zip(CONTINUOUS_PARAMETERS, noise):
        result[field] += offset
    return result

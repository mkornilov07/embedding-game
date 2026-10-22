from django.conf import settings
from django.db import models


class Puzzle(models.Model):
    name = models.CharField(max_length=200)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name="puzzles",
    )
    for_duel = models.BooleanField(default=False)
    is_daily = models.BooleanField(default=False)
    solved = models.BooleanField(default=False)
    pinned = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)

    def __str__(self):
        return self.name


class GenerationDefaults(models.Model):
    """The active defaults used by embedding-based word-sum generation."""

    key = models.CharField(max_length=100, unique=True, default="default")
    model_name = models.CharField(max_length=100, default="conceptnet-numberbatch-en-3k")
    related_pairs = models.BooleanField(default=True)
    sum_scoring_method = models.CharField(
        max_length=16,
        choices=[
            ("cosadd", "3CosAdd"),
            ("cosmul", "3CosMul"),
            ("raw_sum", "Raw vector sum cosine"),
        ],
        default="cosmul",
    )
    refine_iterations = models.PositiveIntegerField(default=1)
    min_similarity = models.FloatField(default=0.25)
    min_refined_similarity = models.FloatField(default=0.5)
    min_gap_ratio = models.FloatField(default=1.0)
    max_synonym_similarity = models.FloatField(default=0.45)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Generation defaults ({self.model_name})"


class WordChainDefaults(models.Model):
    key = models.CharField(max_length=100, unique=True, default="default")
    model_name = models.CharField(max_length=100, default="conceptnet-numberbatch-en-10k-abtt")
    start_percent = models.FloatField(default=20)
    min_previous_similarity = models.FloatField(default=0.47)
    max_earlier_similarity = models.FloatField(default=0.31)
    updated_at = models.DateTimeField(auto_now=True)


class WordCombination(models.Model):
    puzzle = models.ForeignKey(Puzzle, on_delete=models.CASCADE, related_name="combinations")

    def __str__(self):
        return f"WordCombination #{self.pk}"


class WordSum(WordCombination):
    addend1 = models.CharField(max_length=100)
    addend2 = models.CharField(max_length=100)
    sum_word = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.addend1} + {self.addend2} = {self.sum_word}"


class WordChain(WordCombination):
    def __str__(self):
        return f"Word chain #{self.pk}"


class WordChainSlot(models.Model):
    PREFILLED = "prefilled"
    DRAGGABLE = "draggable"
    TYPED = "typed"
    KIND_CHOICES = [
        (PREFILLED, "Pre-filled"),
        (DRAGGABLE, "Drag and drop"),
        (TYPED, "Type"),
    ]

    chain = models.ForeignKey(WordChain, on_delete=models.CASCADE, related_name="slots")
    position = models.PositiveIntegerField()
    kind = models.CharField(max_length=16, choices=KIND_CHOICES)
    word = models.CharField(max_length=100)

    class Meta:
        ordering = ["position"]
        unique_together = ("chain", "position")

    def __str__(self):
        return self.word


class PuzzleCompletion(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="completions")
    puzzle = models.ForeignKey(Puzzle, on_delete=models.CASCADE, related_name="completions")
    first_opened_at = models.DateTimeField(null=True, blank=True)
    first_solved_at = models.DateTimeField(null=True, blank=True)
    used_hints = models.BooleanField(default=False)
    # Kept for compatibility with existing completion data.
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = ("user", "puzzle")

    def __str__(self):
        return f"{self.user} completed {self.puzzle}"


class CalibrationRound(models.Model):
    calibrator = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    kind = models.CharField(max_length=16, default="word_sum")
    chain_length = models.PositiveIntegerField(default=5)
    parameters_before = models.JSONField()
    noise_stdev = models.FloatField(default=0.1)
    learning_rate = models.FloatField(default=0.1)
    noise_vectors = models.JSONField()
    candidates = models.JSONField()
    selected_index = models.PositiveIntegerField(null=True, blank=True)
    parameters_after = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    selected_at = models.DateTimeField(null=True, blank=True)


class CalibrationClick(models.Model):
    round = models.ForeignKey(CalibrationRound, on_delete=models.CASCADE, related_name="clicks")
    event = models.CharField(max_length=64)
    candidate_index = models.PositiveIntegerField(null=True, blank=True)
    payload = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)


class Duel(models.Model):
    STATUS_PENDING = "pending"
    STATUS_ACTIVE = "active"
    STATUS_COMPLETED = "completed"
    STATUS_DECLINED = "declined"
    STATUS_CANCELLED = "cancelled"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_ACTIVE, "Active"),
        (STATUS_COMPLETED, "Completed"),
        (STATUS_DECLINED, "Declined"),
        (STATUS_CANCELLED, "Cancelled"),
    ]

    inviter = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="duels_sent",
    )
    opponent = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="duels_received",
    )
    puzzle = models.ForeignKey(Puzzle, on_delete=models.CASCADE, related_name="duels")
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING)
    winner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="duels_won",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def other_player(self, user):
        return self.opponent if user == self.inviter else self.inviter


class DuelProgress(models.Model):
    duel = models.ForeignKey(Duel, on_delete=models.CASCADE, related_name="progresses")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    # Sum keys retain [sorted_addends, result]; chain keys are ["combination", id].
    solved_rows = models.JSONField(default=list)

    class Meta:
        unique_together = ("duel", "user")

    @property
    def count(self):
        return len(self.solved_rows)

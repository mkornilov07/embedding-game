from django.conf import settings
from django.db import models


class Puzzle(models.Model):
    name = models.CharField(max_length=200)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name="puzzles",
    )
    for_duel = models.BooleanField(default=False)

    def __str__(self):
        return self.name


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


class PuzzleCompletion(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="completions")
    puzzle = models.ForeignKey(Puzzle, on_delete=models.CASCADE, related_name="completions")
    completed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "puzzle")

    def __str__(self):
        return f"{self.user} completed {self.puzzle}"


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
    # Each entry is [sorted_addends_list, sum_word]; used as a set-like dedupe key.
    solved_rows = models.JSONField(default=list)

    class Meta:
        unique_together = ("duel", "user")

    @property
    def count(self):
        return len(self.solved_rows)

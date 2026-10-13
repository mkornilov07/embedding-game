from django.db import migrations, models


def preserve_existing_solve_times(apps, schema_editor):
    PuzzleCompletion = apps.get_model("game", "PuzzleCompletion")
    for completion in PuzzleCompletion.objects.filter(
        completed_at__isnull=False,
        first_solved_at__isnull=True,
    ).iterator():
        completion.first_solved_at = completion.completed_at
        completion.save(update_fields=["first_solved_at"])


class Migration(migrations.Migration):

    dependencies = [
        ("game", "0010_puzzle_solved"),
    ]

    operations = [
        migrations.AddField(
            model_name="puzzlecompletion",
            name="first_opened_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="puzzlecompletion",
            name="first_solved_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="puzzlecompletion",
            name="used_hints",
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name="puzzlecompletion",
            name="completed_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.RunPython(preserve_existing_solve_times, migrations.RunPython.noop),
    ]

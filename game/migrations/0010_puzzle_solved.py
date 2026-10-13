from django.db import migrations, models


def mark_existing_solved_puzzles(apps, schema_editor):
    Puzzle = apps.get_model("game", "Puzzle")
    Puzzle.objects.filter(completions__isnull=False).update(solved=True)


class Migration(migrations.Migration):

    dependencies = [
        ("game", "0009_puzzle_is_daily"),
    ]

    operations = [
        migrations.AddField(
            model_name="puzzle",
            name="solved",
            field=models.BooleanField(default=False),
        ),
        migrations.RunPython(mark_existing_solved_puzzles, migrations.RunPython.noop),
    ]

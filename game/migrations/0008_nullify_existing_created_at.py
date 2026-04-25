from django.db import migrations


def nullify_created_at(apps, schema_editor):
    Puzzle = apps.get_model("game", "Puzzle")
    Puzzle.objects.update(created_at=None)


class Migration(migrations.Migration):

    dependencies = [
        ('game', '0007_puzzle_created_at'),
    ]

    operations = [
        migrations.RunPython(nullify_created_at, migrations.RunPython.noop),
    ]

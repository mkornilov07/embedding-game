from django.db import migrations


def restore_defaults(apps, schema_editor):
    alias = schema_editor.connection.alias
    GenerationDefaults = apps.get_model("game", "GenerationDefaults")
    WordChainDefaults = apps.get_model("game", "WordChainDefaults")

    GenerationDefaults.objects.using(alias).all().update(
        related_pairs=True,
        sum_scoring_method="cosmul",
        refine_iterations=1,
        min_similarity=0.25,
        min_refined_similarity=0.5,
        min_gap_ratio=1.0,
        max_synonym_similarity=0.45,
    )
    WordChainDefaults.objects.using(alias).all().update(
        start_percent=20,
        min_previous_similarity=0.47,
        max_earlier_similarity=0.31,
    )


class Migration(migrations.Migration):
    dependencies = [("game", "0020_alter_generationdefaults_model_name")]

    operations = [
        migrations.RunPython(restore_defaults, migrations.RunPython.noop),
    ]

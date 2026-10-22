from django.db import migrations, models


RENAMED_MODELS = (
    ("conceptnet-numberbatch-en", "conceptnet-numberbatch-en-3k"),
    ("conceptnet-numberbatch-en-abtt", "conceptnet-numberbatch-en-3k-abtt"),
)
REMOVED_MODELS = (
    "conceptnet-numberbatch-en-20k",
    "conceptnet-numberbatch-en-20k-abtt",
)
OPENAI_MODEL = "openai-text-embedding-3-large-subtlex-us-5000"
OPENAI_ABTT_MODEL = f"{OPENAI_MODEL}-abtt"


def update_model_defaults(apps, schema_editor):
    alias = schema_editor.connection.alias
    for model_name in ("GenerationDefaults", "WordChainDefaults"):
        Defaults = apps.get_model("game", model_name)
        manager = Defaults.objects.using(alias)

        for old_name, new_name in RENAMED_MODELS:
            defaults = manager.filter(key=old_name).first()
            if defaults:
                defaults.key = new_name
                defaults.model_name = new_name
                defaults.save(update_fields=("key", "model_name"))

        manager.filter(key__in=REMOVED_MODELS).delete()
        openai_defaults = manager.filter(key=OPENAI_MODEL).first()
        if openai_defaults:
            values = {
                field.name: getattr(openai_defaults, field.name)
                for field in Defaults._meta.fields
                if field.name not in ("id", "key", "model_name", "updated_at")
            }
            manager.get_or_create(
                key=OPENAI_ABTT_MODEL,
                defaults={"model_name": OPENAI_ABTT_MODEL, **values},
            )


class Migration(migrations.Migration):
    dependencies = [("game", "0018_model_specific_defaults")]

    operations = [
        migrations.AlterField(
            model_name="generationdefaults", name="model_name",
            field=models.CharField(default="conceptnet-numberbatch-en-3k", max_length=100),
        ),
        migrations.AlterField(
            model_name="wordchaindefaults", name="model_name",
            field=models.CharField(default="conceptnet-numberbatch-en-10k-abtt", max_length=100),
        ),
        migrations.RunPython(update_model_defaults, migrations.RunPython.noop),
    ]

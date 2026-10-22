from django.db import migrations, models


MODELS = (
    "conceptnet-numberbatch-en",
    "conceptnet-numberbatch-en-abtt",
    "conceptnet-numberbatch-en-10k",
    "conceptnet-numberbatch-en-10k-abtt",
    "conceptnet-numberbatch-en-20k",
    "conceptnet-numberbatch-en-20k-abtt",
    "openai-text-embedding-3-large-subtlex-us-5000",
)


def split_defaults(apps, schema_editor):
    GenerationDefaults = apps.get_model("game", "GenerationDefaults")
    WordChainDefaults = apps.get_model("game", "WordChainDefaults")

    generation = GenerationDefaults.objects.filter(key="default").first()
    if generation:
        original = {field.name: getattr(generation, field.name)
                    for field in GenerationDefaults._meta.fields
                    if field.name not in ("id", "key", "updated_at")}
        GenerationDefaults.objects.filter(pk=generation.pk).update(key=generation.model_name)
        for model_name in MODELS:
            values = dict(original, model_name=model_name)
            GenerationDefaults.objects.get_or_create(key=model_name, defaults=values)

    chain = WordChainDefaults.objects.filter(key="default").first()
    if chain:
        original = {field.name: getattr(chain, field.name)
                    for field in WordChainDefaults._meta.fields
                    if field.name not in ("id", "key", "updated_at")}
        WordChainDefaults.objects.filter(pk=chain.pk).update(key=chain.model_name)
        for model_name in MODELS:
            values = dict(original, model_name=model_name)
            WordChainDefaults.objects.get_or_create(key=model_name, defaults=values)

    for model_name in MODELS:
        GenerationDefaults.objects.get_or_create(
            key=model_name, defaults={"model_name": model_name},
        )
        WordChainDefaults.objects.get_or_create(
            key=model_name, defaults={"model_name": model_name},
        )


class Migration(migrations.Migration):
    dependencies = [("game", "0017_wordchaindefaults_calibrationround_chain_length_and_more")]

    operations = [
        migrations.AlterField(
            model_name="generationdefaults", name="key",
            field=models.CharField(default="default", max_length=100, unique=True),
        ),
        migrations.AlterField(
            model_name="wordchaindefaults", name="key",
            field=models.CharField(default="default", max_length=100, unique=True),
        ),
        migrations.AlterField(
            model_name="wordchaindefaults", name="model_name",
            field=models.CharField(default="conceptnet-numberbatch-en-10k-abtt", max_length=100),
        ),
        migrations.RunPython(split_defaults, migrations.RunPython.noop),
        migrations.RemoveField(model_name="generationdefaults", name="abtt"),
        migrations.RemoveField(model_name="generationdefaults", name="top_n_vocab"),
    ]

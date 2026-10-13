from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("game", "0012_generation_calibration"),
    ]

    operations = [
        migrations.AddField(
            model_name="calibrationround",
            name="noise_stdev",
            field=models.FloatField(default=0.1),
        ),
    ]

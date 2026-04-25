from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('game', '0006_puzzle_pinned'),
    ]

    operations = [
        migrations.AddField(
            model_name='puzzle',
            name='created_at',
            field=models.DateTimeField(auto_now_add=True, null=True),
        ),
    ]

# Generated by Django 3.1.3 on 2021-04-18 22:18

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('AppCDV', '0002_datos_fuentes_modelosclustering_student'),
    ]

    operations = [
        migrations.AddField(
            model_name='modelosclustering',
            name='estado',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='modelosclustering',
            name='fecha_creacion',
            field=models.DateTimeField(default=datetime.datetime(2021, 4, 18, 22, 18, 11, 217276, tzinfo=utc)),
        ),
    ]
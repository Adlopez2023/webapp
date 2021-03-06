# Generated by Django 3.1.3 on 2021-04-17 20:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('AppCDV', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Datos',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('fuente', models.CharField(max_length=100)),
                ('ubicacion', models.CharField(max_length=500)),
                ('atributo', models.CharField(max_length=100)),
                ('seleccionar', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='Fuentes',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('fuente', models.CharField(max_length=100)),
                ('ubicacion', models.CharField(max_length=500)),
                ('atributo', models.CharField(max_length=100)),
                ('seleccionar', models.IntegerField(choices=[(0, 'No'), (1, 'Sí')])),
            ],
        ),
        migrations.CreateModel(
            name='ModelosClustering',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('id_modelo', models.IntegerField()),
                ('nombre_modelo', models.CharField(max_length=50)),
            ],
        ),
        migrations.CreateModel(
            name='Student',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nombre', models.CharField(max_length=50)),
                ('apellidos', models.CharField(max_length=100)),
                ('edad', models.IntegerField()),
                ('email', models.EmailField(max_length=254)),
            ],
        ),
    ]

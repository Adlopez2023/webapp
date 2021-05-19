"""PFinal URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from AppCDV import views
from AppCDV.views import index, HomeModelsLists, NewModel, ResultsModel, up_file, result, \
    DeleteModel, DeleteModelT,  Training, Train#, EditModelT,
from django.conf.urls import url

urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/', index, name="index"),
    path('home/', HomeModelsLists.as_view() , name="listar_modelos"),
    path('newmodel/', NewModel.as_view() , name="nuevo_modelo"),
    path('results/<str:pk>/', ResultsModel.as_view() , name="resultados"),
    path('result/', result, name="result"),

    path('delete/<str:pk>/', DeleteModel.as_view(), name='eliminar_modelos'),
    path('<int:id_modelo>/delete_model/', views.delete_model, name='delete_model'),
    path('deleteR/<id_modelo>/', views.DeleteModelT, name='eliminar_modelosR'),
    path('train/<id_modelo>/', Train, name='entrenar_modelos'),
    path('training/<id_modelo>/', Training, name='entrenando_modelos'),
    #path('train/<id_modelo>/', Training, name='entrenar_modelos'),

    path('upFile/', up_file, name="carga_archivo"),
    #path('results/', ListResults.as_view(), name="list_results"),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

urlpatterns += [
    url(r'^accounts/', include('django.contrib.auth.urls')),
]

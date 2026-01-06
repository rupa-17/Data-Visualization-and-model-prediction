from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('',views.home, name='home'),
    path('login/', views.login_user, name='login'),
    path('register/', views.register, name = 'register'),
    path('index/', views.index, name='index'),
    path('upload/', views.upload, name='upload'),
    path('second/', views.second, name='second'),
    path('data_visualization/', views.data_visualization, name='data_visualization'),
    path('model_prediction/', views.model_prediction, name='model_prediction'),
    path('load-model-section/', views.load_model_section, name='load_model_section'),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
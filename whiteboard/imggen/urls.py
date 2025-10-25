from django.urls import path
from .views import generate_images_batch

urlpatterns = [
    path("generate/", generate_images_batch), #prompt,path_out
]

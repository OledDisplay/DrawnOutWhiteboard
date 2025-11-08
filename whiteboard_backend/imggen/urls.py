from django.urls import path
from .views import generate_images_batch, research_images, process_images

urlpatterns = [
    path("generate/", generate_images_batch), #prompt,path_out
    path("research/", research_images), #prompt, subj
    path("preprocess/", process_images ), #inputdir, outputdir
]

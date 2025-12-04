from django.urls import path
from .views import generate_images_batch, research_images, process_images,list_objects,create_image_object,create_text_object,delete_object

urlpatterns = [
    path("generate/", generate_images_batch), #prompt,path_out
    path("research/", research_images), #prompt, subj
    path("preprocess/", process_images ), #inputdir, outputdir
    path("api/whiteboard/objects/", list_objects, name="wb-list-objects"),

    # POST -> create/update image object from JSON
    path("api/whiteboard/objects/image/", create_image_object, name="wb-create-image"),

    # POST -> create/update text object
    path("api/whiteboard/objects/text/", create_text_object, name="wb-create-text"),

    # DELETE -> delete by name (in JSON body)
    path("api/whiteboard/objects/delete/", delete_object, name="wb-delete-object"),
]

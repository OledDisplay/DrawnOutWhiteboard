from django.db import models
from pgvector.django import VectorField

TEXT_DIM = 384
IMAGE_DIM = 256

class ResearchImage(models.Model):
    path = models.TextField(unique=True)

    prompt_embedding = VectorField(dimensions=TEXT_DIM, null=True, blank=True)
    clip_embedding   = VectorField(dimensions=IMAGE_DIM, null=True, blank=True)

    clip_score        = models.FloatField(null=True, blank=True)
    confidence_score  = models.FloatField(null=True, blank=True)
    final_score       = models.FloatField(null=True, blank=True)


class ImageContext(models.Model):
    image = models.ForeignKey(
        ResearchImage,
        related_name="contexts",
        on_delete=models.CASCADE
    )

    source_kind = models.CharField(max_length=100, null=True, blank=True)
    source_name = models.CharField(max_length=255, null=True, blank=True)
    page_url = models.URLField(null=True, blank=True)
    image_url = models.URLField(null=True, blank=True)
    ctx_text = models.TextField()

    ctx_embedding = VectorField(dimensions=TEXT_DIM, null=True, blank=True)

    ctx_score = models.FloatField(null=True, blank=True)
    ctx_sem_score = models.FloatField(null=True, blank=True)
    ctx_confidence = models.FloatField(null=True, blank=True)

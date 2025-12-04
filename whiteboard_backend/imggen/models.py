from django.db import models

# Create your models here.
# whiteboard_api/models.py
from django.db import models


class WhiteboardObject(models.Model):
    KIND_IMAGE = "image"
    KIND_TEXT = "text"

    KIND_CHOICES = [
        (KIND_IMAGE, "Image JSON"),
        (KIND_TEXT, "Text prompt"),
    ]

    # For images: JSON file name (e.g. "edges_0_skeleton.json")
    # For text: the prompt string (e.g. "Hello, world")
    name = models.CharField(max_length=255, unique=True)

    kind = models.CharField(max_length=10, choices=KIND_CHOICES)

    # Same semantics as your UI: board coords and image scale
    pos_x = models.FloatField(default=0.0)
    pos_y = models.FloatField(default=0.0)
    scale = models.FloatField(default=1.0)  # used for image; you can ignore for text

    # Text-only fields (optional for images)
    letter_size = models.FloatField(null=True, blank=True)
    letter_gap = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.name} ({self.kind})"

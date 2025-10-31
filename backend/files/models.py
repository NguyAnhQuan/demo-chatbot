from django.db import models

class File(models.Model):
    FILE_TYPES = [
        ('pdf', 'PDF'),
        ('docx', 'Word'),
        ('xlsx', 'Excel'),
        ('png', 'Image'),
        ('jpg', 'JPEG Image'),
        ('jpeg', 'JPEG Image'),
        ('txt', 'Text'),
        ('other', 'Other'),
    ]

    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='uploads/')
    file_type = models.CharField(max_length=10, choices=FILE_TYPES)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

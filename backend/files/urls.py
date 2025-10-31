from django.urls import path
from . import views

urlpatterns = [
    path('files/', views.list_files, name='list-files'),
    path('files/upload/', views.upload_file, name='upload-file'),
    path('files/delete/<str:filename>/', views.delete_file, name='delete-file'),
]
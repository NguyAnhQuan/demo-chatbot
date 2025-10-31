from django.urls import path
from . import views

urlpatterns = [
    path('rag/', views.chunk_file, name='chunk_file'),
    path('rag/vectorize/', views.chunk_file, name='vectorize_file'),
    path('rag/status/', views.check_vectorized_status, name='vectorized_status'),
]
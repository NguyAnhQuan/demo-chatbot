import os
from django.conf import settings
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .chunking import Chunking

@api_view(['POST'])
def chunk_file(request):
    """Vector hóa file với cấu hình tùy chỉnh"""
    try:
        file_name = request.data.get('file_name')
        if not file_name:
            return Response({"error": "Tên file không được để trống"}, status=400)
        
        # Xây dựng đường dẫn đầy đủ đến file trong thư mục uploads
        file_path = os.path.join(settings.MEDIA_ROOT, 'uploads', file_name)
        
        # Kiểm tra file có tồn tại không
        if not os.path.exists(file_path):
            return Response({"error": f"File không tồn tại: {file_name}"}, status=404)
        
        chunk_size = request.data.get('chunk_size', 200)
        chunk_overlap = request.data.get('chunk_overlap', 30)
        model_name = request.data.get('model_name', "sentence-transformers/all-MiniLM-L6-v2")
        
        # Thực hiện chunking và embedding
        chunker = Chunking().chunking(
            file_path=file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_name=model_name
        )
        
        return Response({
            "message": "Vector hóa thành công", 
            "result": chunker,
            "file_name": file_name
        }, status=200)
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['GET'])
def check_vectorized_status(request):
    """Kiểm tra trạng thái vector hóa của các file"""
    try:
        # TODO: Implement logic kiểm tra file đã được vector hóa chưa
        # Có thể check trong ChromaDB collection
        return Response({"vectorized_files": []}, status=200)
    except Exception as e:
        return Response({"error": str(e)}, status=500)

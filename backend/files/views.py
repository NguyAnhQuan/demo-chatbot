import os
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import default_storage


UPLOAD_DIR = os.path.join(settings.MEDIA_ROOT, 'uploads')

@api_view(['GET'])
def list_files(request):
    """Liệt kê tất cả file trong uploads/"""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    files = []
    for filename in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(path):
            files.append({
                "name": filename,
                "url": f"{settings.MEDIA_URL}uploads/{filename}",
            })
    return Response(files)


@api_view(['POST'])
def upload_file(request):
    """Upload file vào thư mục uploads/"""
    file = request.FILES.get('file')
    if not file:
        return Response({"error": "Không có file nào được gửi lên"}, status=400)

    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    file_path = os.path.join(UPLOAD_DIR, file.name)
    with default_storage.open(file_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    return Response({"message": f"Đã tải lên {file.name}"}, status=201)


@api_view(['DELETE'])
def delete_file(request, filename):
    """Xóa file khỏi thư mục uploads/"""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return Response({"message": f"Đã xóa file {filename}"}, status=204)
    else:
        return Response({"error": "File không tồn tại"}, status=404)
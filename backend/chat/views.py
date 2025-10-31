from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rag.augmention import Augmention
import google.generativeai as genai

@api_view(['POST'])
def chat(request):
    genai.configure(api_key="AIzaSyBqcdSb3_HR-tJREQlHR26av7Sls1_60Yk")
    model = genai.GenerativeModel('gemini-2.0-flash')
    user_message = request.data.get("message", "")
    augmention = Augmention()
    augmented_response = augmention.get_augmented_response(user_message, top_k=5)
    promt = f"Dựa vào các thông tin sau đây, hãy trả lời câu hỏi {user_message} một cách ngắn gọn và chính xác:\n"
    for idx, doc in enumerate(augmented_response):
        promt += f"Thông tin {idx+1}: {doc.document}\n" 
    print("Prompt sent to model:", promt)
    response = model.generate_content(promt).text

    return Response({"reply": response}, status=200)

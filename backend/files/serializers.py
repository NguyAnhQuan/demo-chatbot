from rest_framework import serializers
from .models import File

class FileSerializer(serializers.ModelSerializer):
    class Meta:
        model = File
        fields = '__all__'
        read_only_fields = ['uploaded_at']
    
    def create(self, validated_data):
        return File.objects.create(**validated_data)

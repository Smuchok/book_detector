from django.contrib import admin
from django.utils.html import format_html

from .models import Detection, DetectedBook, UniqueBook, UserAction


class DetectedBookInline(admin.TabularInline):
    model = DetectedBook
    extra = 0  # No extra empty rows

@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "detection_date")
    inlines = [DetectedBookInline]
    search_fields = ("user__username", "detection_date")
    ordering = ("-detection_date",)

@admin.register(DetectedBook)
class DetectedBookAdmin(admin.ModelAdmin):
    def cropped_image_preview(self, obj):
        if obj.cropped_image:
            return format_html('<img src="{}" style="max-width: 200px;"/>', obj.cropped_image.url)
        return "No Image"
    cropped_image_preview.short_description = "Cropped Image"

    list_display = ("id", "full_text", "detection", "confidence", "detected_at")
    search_fields = ("full_text",)
    ordering = ("-detected_at",)

@admin.register(UniqueBook)
class UniqueBookAdmin(admin.ModelAdmin):
    list_display = ("title", "author", "detection", "created_at")
    search_fields = ("title", "author", "detection__uploaded_image")
    ordering = ("-created_at",)

@admin.register(UserAction)
class UserActionAdmin(admin.ModelAdmin):
    list_display = ('user', 'action_type', 'timestamp', 'description')
    list_filter = ('action_type', 'timestamp')
    search_fields = ('user__username', 'description')

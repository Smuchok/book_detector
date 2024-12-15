import os
import cv2
import numpy as np
from datetime import datetime
import json
import re
import uuid

from django.shortcuts import render, redirect, get_object_or_404
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.http import JsonResponse
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.db.models import Q

from .forms import SaveBooksForm, UserUpdateForm
from .models import UniqueBook, Detection, DetectedBook, UserAction

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import easyocr
import spacy


# Configure Detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)


# Initialize EasyOCR
reader = easyocr.Reader(['en'])
def extract_text(image):
    result_texts = reader.readtext(image)
    full_text = ''
    for detection in result_texts:
        text, confidence = detection[1], detection[2]
        print(f"Text: {text}, Confidence: {confidence}")
        full_text += f"{text} " if confidence>0.3 else ''
    return full_text.strip()


# Load spaCy's English model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

def extract_author_name(full_text):
    """
    Extracts a possible author name from the given full_text.
    """
    # Step 1: Rule-Based Extraction
    # Look for patterns like "By [Author Name]"
    author_match = re.search(r'\b[Bb]y\s([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)', full_text)
    if author_match:
        return author_match.group(1)  # Return the name after "By"

    # Step 2: Named Entity Recognition (NER)
    # Use spaCy to extract PERSON entities
    doc = nlp(full_text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":  # Check if the entity is a person
            return ent.text

    # Step 3: Fallback
    # If no author is found, return a default message or None
    return None


def detect_books(request):
    # # Configure Detectron2
    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # cfg.MODEL.DEVICE = "cpu"

    # predictor = DefaultPredictor(cfg)

    if request.method == "POST" and request.FILES.get('image'):
        # Save uploaded image to Detection model
        uploaded_file = request.FILES['image']
        title = request.POST.get('title', '').strip()

        # fs = FileSystemStorage(location="media/uploads/")
        fs = FileSystemStorage(location=settings.MEDIA_ROOT / 'detection_images')
        file_path = fs.save(uploaded_file.name, uploaded_file)

        # Create a new Detection entry
        detection = Detection.objects.create(
            user=request.user if request.user.is_authenticated else None,
            uploaded_image=uploaded_file,
            title=title if title else None,
        )

        # UserAction if there is an user
        if request.user.is_authenticated:
            UserAction.objects.create(
                user=request.user,
                action_type='add_detection',
                detection=detection,
                description='User added a new detection.'
            )

        # Detect books
        image_path = os.path.join(fs.location, file_path)
        image = cv2.imread(image_path)
        outputs = predictor(image)

        instances = outputs["instances"]
        boxes = instances.pred_boxes if instances.has("pred_boxes") else None
        masks = instances.pred_masks if instances.has("pred_masks") else None
        scores = instances.scores if instances.has("scores") else None
        classes = instances.pred_classes if instances.has("pred_classes") else None

        # Detectron2 metadata
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        category_names = metadata.get("thing_classes", None)

        # Organize files by date
        current_date = datetime.now().strftime("%Y-%m-%d")
        cropped_dir = os.path.join(settings.MEDIA_ROOT, "detected_books/cropped", current_date)
        rotated_dir = os.path.join(settings.MEDIA_ROOT, "detected_books/rotated", current_date)
        os.makedirs(cropped_dir, exist_ok=True)
        os.makedirs(rotated_dir, exist_ok=True)

        # Initialize list for detected objects to pass to the UI
        detected_objects = []

        # for i in range(len(boxes)):
        for i, unused in enumerate(boxes, start=0):
            # Get class name and confidence
            class_id = classes[i].cpu().item() if classes is not None else None
            class_name = category_names[class_id] if category_names and class_id is not None else "Unknown"
            confidence = scores[i].cpu().item() if scores is not None else None

            # Filter for "book" class
            if class_name != "book":
                continue

            # Extract bounding box
            x1, y1, x2, y2 = boxes.tensor[i].cpu().int().numpy()

            # Crop the image
            cropped_image = image[y1:y2, x1:x2]
            cropped_filename = f"book_{i+1}.png"
            cropped_image_path = os.path.join(settings.MEDIA_ROOT, "common_images", cropped_filename)
            cv2.imwrite(cropped_image_path, cropped_image)

            # Rotate the cropped image
            rotated_image = cv2.rotate(cropped_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rotated_filename = f"book_{i+1}_rotated.png"
            rotated_image_path = os.path.join(settings.MEDIA_ROOT, "common_images", rotated_filename)
            cv2.imwrite(rotated_image_path, rotated_image)

            # Extract text using OCR
            result_texts = reader.readtext(rotated_image)
            full_text = " ".join(
                detection[1] for detection in result_texts if detection[2] > 0.3
            )

            possible_author = extract_author_name(full_text)  # Extract author name

            # Save DetectedBook entry
            detected_book = DetectedBook.objects.create(
                detection=detection,
                cropped_image=f"common_images/{cropped_filename}",
                rotated_image=f"common_images/{rotated_filename}",
                full_text=full_text if full_text else "No text detected",
                class_name=class_name,
                confidence=confidence,
            )

            # Add to UI list
            detected_objects.append({
                "id": str(i),  # Assign a unique ID to each book
                "cropped_image_url": detected_book.cropped_image.url,  # Relative URL to cropped image
                "rotated_image_url": detected_book.rotated_image.url,  # Relative URL to rotated image
                "full_text": detected_book.full_text,
                "class_name": detected_book.class_name,
                "confidence": f"{detected_book.confidence:.2f}",
                "author": possible_author,
            })

        # Pass results and form to the template
        return render(request, 'books/detect_books.html', {
            "detected_objects": detected_objects,
            "uploaded_image": detection.uploaded_image.url,
            "detection": detection,
            "save_books_form": SaveBooksForm(initial={"books": json.dumps(detected_objects)})
        })

    return render(request, 'books/upload.html')


def save_books(request):
    if request.method == "POST":
        form = SaveBooksForm(request.POST)
        if form.is_valid():
            books_data = json.loads(form.cleaned_data['books'])  # Detected books data
            selected_books = request.POST.getlist('selected_books')  # Selected book IDs
            detection_id = request.POST.get('detection_id')  # Pass detection ID in the form

            # Retrieve the Detection instance
            detection = Detection.objects.get(id=detection_id)

            for book_id in selected_books:
                # Find the detected book data by ID
                book_data = next((book for book in books_data if str(book['id']) == book_id), None)
                if book_data:
                    # Create a UniqueBook linked to the Detection
                    UniqueBook.objects.create(
                        title=request.POST.get(f"title_{book_id}", book_data['full_text']),  # User-provided or default
                        author=request.POST.get(f"author_{book_id}", ""),  # User-provided or empty
                        detection=detection,  # Link to Detection
                        cropped_image=book_data['cropped_image_url'].replace(settings.MEDIA_URL, ""),  # Save cropped image
                        rotated_image=book_data['rotated_image_url'].replace(settings.MEDIA_URL, ""),  # Save rotated image
                        user=request.user if request.user.is_authenticated else None,
                    )

            unique_books_queryset = UniqueBook.objects.filter(detection=detection)
            action = UserAction.objects.create(
                user=request.user,
                action_type='add_unique_books',
                detection=detection,
                description=f'User added {len(unique_books_queryset)} books from new Detection.'
            )
            action.unique_books.set(unique_books_queryset)

            return JsonResponse({"success": True, "message": "Books saved successfully!"})
        return JsonResponse({"success": False, "message": "Invalid form submission."})


def unique_books_list(request):
    """
    Displays a list of all UniqueBooks.
    """
    unique_books = UniqueBook.objects.all().order_by('-created_at')  # Retrieve all unique books
    detections   = Detection.objects.all().order_by('-detection_date')

    # [print(i.cropped_image.url) for i in unique_books]
    # [print(i.cropped_image) for i in unique_books]
    return render(request, 'books/unique_books_list.html', {
        'unique_books': unique_books,
        'detections':   detections,
    })


def unique_book_detail(request, pk):
    """
    Displays the details of a single UniqueBook with navigation to the next and previous books.
    """
    unique_book = get_object_or_404(UniqueBook, pk=pk)

    # Get the next book (by creation time)
    next_book = UniqueBook.objects.filter(created_at__gt=unique_book.created_at).order_by('created_at').first()

    # Get the previous book (by creation time)
    previous_book = UniqueBook.objects.filter(created_at__lt=unique_book.created_at).order_by('-created_at').first()

    return render(request, 'books/unique_book_detail.html', {
        'unique_book': unique_book,
        'next_book': next_book,
        'previous_book': previous_book,
    })


@login_required
def profile(request):
    if request.user.is_authenticated:
        recent_actions = UserAction.objects.filter(user=request.user).order_by('-timestamp')[:20]
        return render(request, 'books/profile.html', {
            'recent_actions': recent_actions
        })
    else:
        return redirect('login')

@login_required
def edit_profile(request):
    if request.method == 'POST':
        us = request.user
        old_user_data = {
            'username':     us.username,
            'email':        us.email,
            'first_name':   us.first_name,
            'last_name':    us.last_name,
        }

        form = UserUpdateForm(request.POST, instance=us)
        if form.is_valid():
            form.save()
            new_user_data = form.cleaned_data
            description = ""
            for k,v in old_user_data.items():
                if k in new_user_data:
                    if v != new_user_data[k]:
                        description += f'{k}: {v} --> {new_user_data[k]}\n'

            print(f"edit_profile {description=}")

            UserAction.objects.create(
                user=request.user,
                action_type='profile_update',
                description=description,
            )
            messages.success(request, 'Your profile has been updated!')
            return redirect('profile')
    else:
        form = UserUpdateForm(instance=request.user)
    return render(request, 'books/edit_profile.html', {'form': form})


def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()  # Зберігаємо нового користувача
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}! You can now log in.')
            return redirect('login')  # Перенаправлення на сторінку логіну
    else:
        form = UserCreationForm()
    return render(request, 'books/register.html', {'form': form})


def detection_detail(request, pk):
    detection = get_object_or_404(Detection, pk=pk)
    detected_books = DetectedBook.objects.filter(detection=detection)
    return render(request, 'books/detection_detail.html', {
        'detection': detection,
        'detected_books': detected_books
    })

def save_unique_books_from_detection(request, detection_id):
    if request.method == 'POST':
        selected_books = request.POST.getlist('selected_books')
        for book_id in selected_books:
            title = request.POST.get(f'title_{book_id}', 'Untitled')
            author = request.POST.get(f'author_{book_id}', 'Unknown')
            detected_book = DetectedBook.objects.get(id=book_id)
            custom_list = []

            # Створюємо UniqueBook
            unique_book = UniqueBook.objects.create(
                title=title,
                author=author,
                cropped_image=detected_book.cropped_image,
                rotated_image=detected_book.rotated_image,
                detection=detected_book.detection,
                user=request.user if request.user.is_authenticated else None,
            )
            
            custom_list.append(unique_book.id)

        # unique_books_queryset = UniqueBook.objects.filter(detection=detected_book.detection)
        unique_books_queryset = UniqueBook.objects.filter(id__in=custom_list)
        action = UserAction.objects.create(
            user=request.user,
            action_type='add_unique_books',
            detection=detected_book.detection,
            description=f'User added {len(unique_books_queryset)} books from already existed Detection.'
        )
        action.unique_books.set(unique_books_queryset)

        return redirect('unique_books_list')
    return redirect('detection_detail', pk=detection_id)


def user_unique_books(request):
    if request.user.is_authenticated:
        user_books      = UniqueBook.objects.filter(user=request.user).order_by('-created_at')
        user_detections = Detection.objects.filter(user=request.user).order_by('-detection_date')
        return render(request, 'books/user_unique_books.html', {
            'user_books': user_books,
            'user_detections': user_detections
        })
    else:
        return redirect('login')

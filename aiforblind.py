import os
import sys
import time
import urllib.request

import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import pyttsx3

# --- Load YOLOv5 model for object detection ---
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

# --- Initialize video capture object with webcam index (usually 0) ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("Could not open webcam (index 0). Check it isn't in use by another app.")

# --- Initialize pyttsx3 engine ---
engine = pyttsx3.init()

# --- Approximate real-world height (meters) per COCO class, for distance estimation ---
# Distance without camera calibration is inherently a rough estimate, not a measurement.
KNOWN_HEIGHTS_M = {
    'person': 1.7, 'bicycle': 1.0, 'car': 1.5, 'motorcycle': 1.1,
    'bus': 3.2, 'truck': 3.2, 'chair': 0.9, 'bottle': 0.25,
    'cup': 0.1, 'laptop': 0.25, 'cell phone': 0.15, 'book': 0.25, 'tv': 0.6,
}
DEFAULT_HEIGHT_M = 0.3
FOCAL_LENGTH_PX = 615  # rough default for a typical webcam; recalibrate for accuracy

# --- Avoid re-announcing the same object every frame ---
SPEECH_COOLDOWN_SEC = 4.0
last_announced = {}

# --- Load pretrained Places365 model (ResNet18) for scene recognition ---
# Places365 has 365 scene categories (indoor + outdoor) — no training required,
# unlike the EfficientNet/ImageNet model this replaces (which only knows objects, not scenes).
MODEL_PATH = 'resnet18_places365.pth'
CATEGORIES_PATH = 'categories_places365.txt'

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(
        'http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar', MODEL_PATH
    )
if not os.path.exists(CATEGORIES_PATH):
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt',
        CATEGORIES_PATH,
    )

# Build scene_mapping: class index (str) -> readable label, from the categories file
# (lines look like "/a/airport_terminal 0")
scene_mapping = {}
with open(CATEGORIES_PATH) as f:
    for line in f:
        path, idx = line.strip().split(' ')
        label = path.split('/')[2] if path.count('/') >= 2 else path.split('/')[-1]
        scene_mapping[idx] = label.replace('_', ' ')

model = models.resnet18(num_classes=365)
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

# Define transformations for input images to the model
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define default scene label
default_scene_label = "room"

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB format (OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection using YOLOv5
        results_yolo = yolo_model(frame_rgb)
        detections_yolo = results_yolo.pandas().xyxy[0]

        # Loop through detected objects from YOLOv5
        now = time.time()
        speech_queue = []
        distance_info = []
        for _, row in detections_yolo.iterrows():
            xmin, ymin, xmax, ymax, _, class_id, name = row.to_list()
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

            # Draw rectangle and label around the detected object
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(frame, name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Distance estimation (pinhole approximation, without camera calibration)
            height_in_pixels = ymax - ymin
            if height_in_pixels > 0:
                real_height_m = KNOWN_HEIGHTS_M.get(name, DEFAULT_HEIGHT_M)
                estimated_distance = (real_height_m * FOCAL_LENGTH_PX) / height_in_pixels
                distance_info.append(f"{name}: {estimated_distance:.2f} m")
                cv2.putText(
                    frame, f"Est. Distance: {estimated_distance:.2f} m", (xmin, ymax + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
                )
                # Only queue speech for an object once per cooldown window, to avoid
                # re-announcing every detection on every frame (was blocking + spammy).
                if now - last_announced.get(name, 0) > SPEECH_COOLDOWN_SEC:
                    speech_queue.append(f"{name} approximately {estimated_distance:.1f} meters away")
                    last_announced[name] = now

        if speech_queue:
            engine.say(". ".join(speech_queue))
            engine.runAndWait()

        # Perform scene recognition using the Places365 model
        frame_resized = cv2.resize(frame, (224, 224))
        frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input_tensor = transform(frame_resized).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
        predicted_index = torch.argmax(output, 1).item()
        scene_label = scene_mapping.get(str(predicted_index), default_scene_label)
        distances_str = ", ".join(distance_info) if distance_info else "none"
        print(f"Predicted index: {predicted_index} | Scene label: {scene_label} | Distances: {distances_str}")

        cv2.putText(frame, f"Scene: {scene_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the resulting frame with detections
        cv2.imshow('frame', frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    # Release capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

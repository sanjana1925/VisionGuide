# 👁️ VisionGuide - AI Assistant for the Visually Impaired

VisionGuide is a real-time computer vision assistant that helps visually impaired users navigate their surroundings. 

## ✨ Key Features

| Feature | Description | Tech Used |
|---------|-------------|-----------|
| **🎯 Real-time Object Detection** | Identifies people and objects in the live camera feed | YOLOv5 |
| **📏 Distance Estimation** | Estimates how far each detected object is, in meters | Pinhole camera approximation |
| **🏞️ Scene Recognition** | Classifies the surrounding environment into 365 scene categories | ResNet18 (Places365) |
| **🔊 Voice Feedback** | Speaks detected objects and distances aloud, with cooldown to avoid spam | pyttsx3 (Text-to-Speech) |
| **🖥️ Live Visual Overlay** | Displays bounding boxes, distances, and scene labels on the video feed | OpenCV |

<br>

![VisionGuide Architecture](/VisionGuide_architecture_Diagram.svg)

<br><br>
*Each webcam frame runs through YOLOv5 (object detection) and a Places365-trained ResNet18 (scene recognition) in parallel, feeds distance estimates and scene labels into a cooldown-limited speech queue, and outputs both a live annotated video overlay and spoken audio via pyttsx3.*

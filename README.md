# 👁️ VisionGuide - AI Assistant for the Visually Impaired

VisionGuide is a real-time computer vision assistant that helps visually impaired users navigate their surroundings. It uses a live webcam feed to detect nearby objects, estimate how far away they are, recognize the type of scene the user is in, and speaks all of this out loud.

## ✨ Key Features

| Feature | Description | Tech Used |
|---------|-------------|-----------|
| **🎯 Real-time Object Detection** | Identifies people and objects in the live camera feed | YOLOv5 |
| **📏 Distance Estimation** | Estimates how far each detected object is, in meters | Pinhole camera approximation |
| **🏞️ Scene Recognition** | Classifies the surrounding environment into 365 scene categories | ResNet18 (Places365) |
| **🔊 Voice Feedback** | Speaks detected objects and distances aloud, with cooldown to avoid spam | pyttsx3 (Text-to-Speech) |
| **🖥️ Live Visual Overlay** | Displays bounding boxes, distances, and scene labels on the video feed | OpenCV |

*Helping visually impaired users understand what's around them, in real time, through sound.*

![VisionGuide Architecture](/VisionGuide_architecture_Diagram.svg)
*Each webcam frame runs through YOLOv5 (object detection) and a Places365-trained ResNet18 (scene recognition) in parallel, feeds distance estimates and scene labels into a cooldown-limited speech queue, and outputs both a live annotated video overlay and spoken audio via pyttsx3.*

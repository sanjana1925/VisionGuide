# 👁️ VisionGuide

It is a real-time computer vision assistant that helps visually impaired users navigate their surroundings. 

## Key Features

| Feature | Description |
|---------|-------------|
| **🎯 Real-time Object Detection** | Identifies people and objects in the live camera feed |
| **📏 Distance Estimation** | Estimates how far each detected object is, in meters |
| **🏞️ Scene Recognition** | Classifies the surrounding environment into 365 scene categories |
| **🔊 Voice Feedback** | Speaks detected objects and distances aloud, with cooldown to avoid spam |
| **🖥️ Live Visual Overlay** | Displays bounding boxes, distances, and scene labels on the video feed |

<br><br>

![VisionGuide Architecture](/VisionGuide_architecture_Diagram.svg)

<br><br>
*Each webcam frame runs through YOLOv5 (object detection) and a Places365-trained ResNet18 (scene recognition) in parallel, feeds distance estimates and scene labels into a cooldown-limited speech queue, and outputs both a live annotated video overlay and spoken audio via pyttsx3.*

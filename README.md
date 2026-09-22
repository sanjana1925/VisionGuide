<div align="center">

# 👁️ VisionGuide
### AI Assistant for the Visually Impaired

Real-time computer vision assistant that helps visually impaired users navigate their surroundings.

</div>

<br>

- 🎯 **Real-time Object Detection** — identifies people and objects in the live camera feed using **YOLOv5**
- 📏 **Distance Estimation** — estimates how far each object is, via pinhole camera approximation
- 🏞️ **Scene Recognition** — classifies the surroundings into 365 scene categories with **ResNet18 (Places365)**
- 🔊 **Voice Feedback** — speaks detections aloud with a cooldown, so it never spams the same object
- 🖥️ **Live Visual Overlay** — draws bounding boxes, distances, and scene labels on the video feed

<br>

---

<br>

<div align="center">

## 🏗️ Architecture

![VisionGuide Architecture](/VisionGuide_architecture_Diagram.svg)

*Each webcam frame runs through YOLOv5 (object detection) and a Places365-trained ResNet18 (scene recognition) in parallel, feeds distance estimates and scene labels into a cooldown-limited speech queue, and outputs both a live annotated video overlay and spoken audio via pyttsx3.*

</div>

# Blind Assist App

## Overview
This project is a **voice-controlled mobile application** designed to assist blind and visually impaired users.  
The app can capture images using the phone’s camera or gallery, send them to a Flask server, and receive AI-generated scene descriptions via YOLO object detection and an LLM (Large Language Model).  

It is built with:
- **Kivy/KivyMD** for the Android app interface
- **Flask** for the backend server
- **YOLOv8** for object detection
- **Ollama LLM** for natural language scene interpretation

---

## Features
- 🎙️ **Voice control**: Trigger camera actions by voice commands (e.g., “Take picture”).  
- 📷 **Camera integration**: Automatic photo capture on command.  
- 🖼️ **Gallery support**: Upload images from the phone gallery (with auto JPEG conversion).  
- 🔗 **Bluetooth-ready architecture** for wearable integration (future work).  
- 🧠 **Real-time AI analysis**: YOLO detects objects → LLM summarizes scene → result sent back as audio to user.  

---

## Installation

Clone repository:
```bash
git clone https://github.com/DIDI5I/blind-assist-app.git
cd blind-assist-app

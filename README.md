# CV Hand Project

## Overview
A computer vision project for real-time hand gesture recognition using webcam input.

This project uses MediaPipe and OpenCV to detect hand landmarks and recognize gestures, with additional logic to improve prediction stability.

---

## Features

### Hand Gesture Recognition
- Real-time hand detection using webcam
- Gesture classification using MediaPipe Gesture Recognizer
- Displays raw prediction and stabilized output

### Stability Optimization
- Voting-based smoothing (deque + majority vote)
- Confidence threshold filtering
- Cooldown mechanism to prevent rapid label switching
- Streak-based fast detection for responsiveness

---

## Rock-Paper-Scissors Mode
- Supports two-hand gesture recognition
- Maps gestures to:
  - ROCK (Closed_Fist)
  - PAPER (Open_Palm)
  - SCISSORS (Victory)
- Determines winner in real-time

---

## Tech Stack
- Python
- OpenCV
- MediaPipe

---

## What I Did
- Implemented real-time hand landmark detection
- Built gesture recognition pipeline using MediaPipe
- Designed stabilization logic (voting + cooldown + streak)
- Extended system to support two-hand interaction (RPS game)

---

## Key Learning
- Handling noisy real-time predictions in vision systems
- Improving model output stability with heuristic logic
- Designing responsive real-time interaction systems


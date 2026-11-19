# Live Face ID & Blink Liveness Detection

A real-time Python face recognition and blink-based liveness verification system built with **OpenCV**, **`face_recognition`**, and **MediaPipe Face Mesh**. The application runs face matching and eye-blink detection in a background thread while streaming live webcam video on the main display loop.

## Features

- **Known Face Loading (`./images`):** Automatically loads `.jpg` and `.png` reference images from the `images/` directory, extracts face encodings, and maps filenames to user labels.
- **Multi-Threaded Architecture:** Separates camera capture and frame rendering (`video_show`) from face recognition and eye-blink tracking (`compare_background` -> `eye_dis`) using a `threading.Lock`.
- **Consecutive Match Confirmation:** Requires **4** matched detections before confirming recognition and transitioning to liveness verification.
- **Eye Aspect Ratio (EAR) Blink Detection:** Uses MediaPipe 478-point refined face landmarks to measure vertical-to-horizontal eye openness (`EAR < 0.17`) and verifies natural blink duration.
- **On-Screen Feedback:** Draws bounding boxes and names around recognized faces, prompts the user to `"blink"`, and displays `"live face ID has been executed successfully"` once a blink is detected.

## Project Structure

```text
.
├── images/             # Reference face images (.jpg or .png named after each person)
├── main.py             # Main script containing Live_face_compare
└── README.md           # Project documentation
```

## Prerequisites

- Python 3.8+
- A working webcam (`cv.VideoCapture(0)`)
- CMake and C++ build tools (required by `dlib` / `face_recognition` on most systems)

### Install Dependencies

```bash
pip install opencv-python face-recognition mediapipe
```

## Setup

1. Create an `images/` folder in the same directory as your script.
2. Add clear, front-facing `.jpg` or `.png` photos of known individuals into `images/` (for example, `images/alice.jpg` or `images/bob.png`).

## Usage

Run the script from your terminal:

```bash
python main.py
```

- Look at the camera until your face is recognized and a green bounding box appears with your name.
- Blink when prompted on screen (`blink`) to complete liveness verification.
- Press **`q`** while the video window is focused to stop the threads and exit.

## Configuration & Key Parameters

- **Recognition threshold (`compare_background`):** Currently requires `count >= 4` matches before locking in recognition and starting `eye_dis()`.
- **EAR threshold (`eye_dis`):** `EAR_THRESHOLD = 0.17` with `MIN_CLOSED_FRAMES = 2`.
- **Blink duration window (`eye_dis`):** Currently checks `0.05 < blink_duration < 0.1` seconds (adjust this range if normal blinks on your setup fall slightly outside `50ms–100ms`).

## Known Limitations & Notes

- **Mirror / Bounding Box Alignment:** `video_show()` horizontally flips the frame (`cv.flip(frame, 1)`) before drawing `face_locations_to_draw`, which were detected on an unflipped frame—verify box horizontal alignment on your camera setup.
- **Thread & Loop Exit:** `video_show()` checks `cap.isOpened()`, while background loops check `self.running`. Ensure both exit cleanly when pressing `q` or closing the window.
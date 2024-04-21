# License Plate Recognition

This project provides a FastAPI application designed to upload an image and extract text from it using a two-stage process involving YOLOv8 for detection and PaddleOCR for Optical Character Recognition (OCR). It's specifically tailored for scenarios where precise text extraction from images is required, leveraging the power of state-of-the-art machine learning models.

Validation + Create Dataset: 

<a href="https://colab.research.google.com/drive/1irTA_DqFmBsXHaOGUhoOcljrpe3SrBR6?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

## Features

- **Image Upload**: Allows users to upload images via a FastAPI endpoint.
- **Text Detection and Extraction**: Utilizes YOLOv8 for detecting text regions in images and PaddleOCR for extracting text.
- **Optimized Performance**: Implements efficient handling of images and temporary files to optimize performance and resource usage.

## Prerequisites

Before you can run this application, you need to ensure your environment is set up with the following:

- Python 3.8 or newer
- FastAPI
- Uvicorn (for running the FastAPI app)
- PaddleOCR
- OpenCV
- Ultralytics YOLOv8 (Installation instructions can be found at [YOLOv8 GitHub Repository](https://github.com/ultralytics/yolov8))

## Installation

1. Clone this repository to your local machine.
2. Install the required dependencies.
```bash
    pip install -r requirements.txt
```
## Running the Application

To run the application, use the following command:
```
    python main2.py
```

This will start the FastAPI application on port 5000, allowing you to upload images and extract text via the `/upload-image` endpoint.
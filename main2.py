from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from paddleocr import PaddleOCR
import shutil
import os
import tempfile
from two_stages_yolov8 import process_image_and_extract_text
import time

app = FastAPI()

# Initialize PaddleOCR with configuration
ocr = PaddleOCR(use_angle_cls=True, lang='en')

@app.post("/upload-image")
async def upload_image(image: UploadFile = File(...)):
    """
    Endpoint to upload an image and extract text using YOLOv8 and PaddleOCR.
    """
    start_time = time.time()

    # Ensure an image file is provided
    if not image.filename:
        raise HTTPException(status_code=400, detail="No image selected for uploading")
    
    # Create a temporary directory to save the uploaded image
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, image.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Process the image and extract text
        detection_results = process_image_and_extract_text(temp_path, ocr)

    execution_time = time.time() - start_time
    print('Summary: ', execution_time)
    
    return {"detections": detection_results, "execution_time": execution_time}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

from fastapi import FastAPI, File, UploadFile, HTTPException
from paddleocr import PaddleOCR
import shutil
import os
import tempfile
from PIL import Image
import pyheif  # Import the library for HEIC conversion
from two_stages_yolov8 import process_image_and_extract_text
import time

app = FastAPI()

# Initialize PaddleOCR with configuration
ocr = PaddleOCR(use_angle_cls=True, lang='en')

@app.post("/upload-image")
async def upload_image(image: UploadFile = File(...)):
    """
    Endpoint to upload an image, convert HEIC or any other format to PNG if necessary, 
    and extract text using YOLOv8 and PaddleOCR.
    """
    start_time = time.time()

    if not image.filename:
        raise HTTPException(status_code=400, detail="No image selected for uploading")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, image.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Check if the image is in HEIC format
        if temp_path.lower().endswith('.heic'):
            heif_file = pyheif.read(temp_path)
            image = Image.frombytes(
                heif_file.mode, 
                heif_file.size, 
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
            temp_path_png = temp_path.rsplit('.', 1)[0] + '.png'
            image.save(temp_path_png, "PNG")
            temp_path = temp_path_png
        # else:
        #     # If not HEIC, open with PIL for consistency
        #     image = Image.open(temp_path)
        
        # # Save the image as PNG in both cases
        # temp_path_png = temp_path.rsplit('.', 1)[0] + '.png'
        # image.save(temp_path_png, "PNG")
        # temp_path = temp_path_png  # Use the PNG file for further processing
        
        detection_results = process_image_and_extract_text(temp_path, ocr)

    execution_time = time.time() - start_time
    
    return {"detections": detection_results, "execution_time": execution_time}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

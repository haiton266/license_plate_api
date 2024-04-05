import cv2
from ultralytics import YOLO
import tempfile
import os
from paddleocr import PaddleOCR, draw_ocr
import re
import numpy as np

def process_image_and_extract_text(source_image_path, ocr):
    try:
        model = YOLO('best.pt')
        results = model([source_image_path])
        for result in results:
            boxes = result.boxes
        
        im0 = cv2.imread(source_image_path)
        max_area = 0
        largest_cropped_image = None
        h, w = 0, 0
        for bbox in boxes.xyxy:
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area > max_area:
                max_area = area
                # Adjust the bounding box coordinates to add a 20px margin
                x1, y1, x2, y2 = bbox
                w = int(x2) - int(x1)
                h = int(y2) - int(y1)
                largest_cropped_image = im0[int(y1):int(y2), int(x1):int(x2)]
        background = np.zeros((h + 60, w + 60, 3), dtype=np.uint8)
        start_y, start_x = 30, 30
        background[start_y:start_y+h, start_x:start_x+w] = largest_cropped_image
        largest_cropped_image = background
    except:
        return "No detected license plate"
        
    # cv2.imwrite('ok.jpg', largest_cropped_image)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        cv2.imwrite(tmp_file.name, largest_cropped_image)
        # cv2.imwrite('ok.jpg', largest_cropped_image)
        # Run OCR on the saved image
        import time
        start = time.time()
        try:
            result = ocr.ocr(tmp_file.name, cls=True)
            print('result ', result)
            print('ocr: ', time.time() - start)
            # Ensure to clean up the temporary file after processing
            os.remove(tmp_file.name)

            # Extract and concatenate text from OCR results
            extracted_text = ''
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    extracted_text += line[1][0]
            print('extracted_text ', extracted_text)
            cleaned_text = re.sub(r"[^a-zA-Z0-9]", "", extracted_text)
        except:
            return "No detect character in License Plate"
    return cleaned_text

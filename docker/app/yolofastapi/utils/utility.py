
from pickletools import uint8
from typing import Any
from numpy import fromstring
from fastapi import UploadFile
import cv2
import torch
from ultralytics import YOLO

# yolo_model.py

class YoloModel:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        # Load the model from the given path
        model = YOLO(model_path)
        return model

    def predict(self, image, device=0, conf=0.6, imgsz=1024):
        # Make a prediction using the model
        with torch.no_grad():
            results = self.model(source=image, device=device, conf=conf, imgsz=imgsz, verbose=False)
            boxes = results[0].boxes.data.tolist()
            kids = [box for box in boxes if int(box[5]) == 0]
            caregivers = [box for box in boxes if int(box[5]) != 0]
        return kids, caregivers, boxes
    
model = YoloModel('models/yolo8_caregiver_kids_in_daycare_10_plus.pt')

async def process_image(file: UploadFile, device: int, conf: float, imgsz: int):
    try:
        file_contents = await file.read()
        assert file_contents, "File is empty"
        image = cv2.imdecode(fromstring(file_contents, uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error reading file {file.filename}: {e}")
        return {"error": f"Error reading file {file.filename}: {e}"}

    # Preprocess the image if necessary
    # image = preprocess(image)

    kids, caregivers, boxes = model.predict(image, device, conf, imgsz)
    detections = []
    for kid in kids:
        if kid:
            detections.append([kid[0],kid[1],kid[2],kid[3],0])
            print([kid[0],kid[1],kid[2],kid[3],0])
    for caregiver in caregivers:
        detections.append([caregiver[0],caregiver[1],caregiver[2],caregiver[3],1])
        print([caregiver[0],caregiver[1],caregiver[2],caregiver[3],1])
    
    return detections
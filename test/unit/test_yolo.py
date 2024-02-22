from ultralytics import YOLO

# Load a model
# model = YOLO("/home/agent/workspace/models/YOLOv8/yolov8n.yaml")  # build a new model from scratch
model = YOLO("/home/agent/workspace/models/YOLOv8/yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format
print(results.xyxy[0])  # print class, xyxy box coordinates and confidence of each detection
from ultralytics import YOLO

def test_inference():
    # Load a model
    # model = YOLO("/home/agent/workspace/models/YOLOv8/yolov8n.yaml")  # build a new model from scratch
    model = YOLO("/home/agent/workspace/MediNoteAI/test/unit/yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    # model.train(data="coco128.yaml", epochs=3)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    results = model("https://ultralytics.com/images/zidane.jpg")
    # path = model.export(format="onnx")  # export the model to ONNX format
    print(results.xyxy[0])  # print class, xyxy box coordinates and confidence of each detection

def test_training():
    # model = YOLO('/mnt/models/yolov9/yolov9-c.pt')
    # Load a model
    # model = YOLO("/mnt/models/YOLOv8/yolov8n.yaml")  # build a new model from scratch
    model = YOLO("/mnt/models/YOLOv8/yolov8n.pt")  # load a pretrained model (recommended for training)


    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data='/mnt/datasets/medinote/images/kids_all_data/data.yaml', epochs=100, imgsz=860)
    print(results)

if __name__ == "__main__":
    test_inference()
    # test_training()
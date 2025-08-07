from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(data="crb-labels.v2i.yolov8/data.yaml", epochs=200, imgsz=640)
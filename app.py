from ultralytics import YOLO
import json

with open('local_config.json', 'r') as config_file:
    config = json.load(config_file)

model = YOLO("yolo11n.pt")
model.cuda()

rtsp_url = config["rtsp_url"]

results = model.track(source=rtsp_url, show=True, classes=[0])

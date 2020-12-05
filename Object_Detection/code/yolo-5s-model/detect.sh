venv\Scripts\activate
python detect.py --device 0 --weights runs/train/yolov5s_eagleview3/weights/best.pt --img 640 --conf 0.4 --source ./inference/images/ --name yolov5s_eagleview3

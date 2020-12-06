# python convert.py --datasets COCO --img_path ./coco/images --label ./coco/annotations/bbox-annotations.json --convert_output_path ./coco/labels --img_type ".jpg"  --cls_list_file ./coco/coco.names.txt
python train.py --img 640 --batch 16 --epochs 30 --data ./data/sampledata.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt  --name yolov5s_ev --cache 

python detect.py --device 0 --weights runs/train/yolov5s_ev/weights/best.pt --img 640 --conf 0.4 --source ./inference/images/ --name yolov5s_ev

python test.py --weights runs/train/yolov5s_ev/weights/best.pt --data sampledata.yaml --img 640 --iou 0.65 --name yolov5s_eagleview3  --save_txt
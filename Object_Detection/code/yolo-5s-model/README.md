
## Training on Custom Data and Detection ##

- We are going to use YOLO v5 model architecture which is a single-stage object detector . It got open-sourced on May 30, 2020 by Glenn Jocher from ultralytics
  There is no published paper, but [the complete project is on GitHub.](https://github.com/ultralytics/yolov5)

- YOLO v5 uses PyTorch 

- We are going to clone the repo and do necessary changes accordingly to suit our problem statement 

- We will be fine-tuning  pre-trained model version. Take a look at the overview of the [pre-trained checkpoints](https://github.com/ultralytics/yolov5/blob/f9ae460eeccd30bdc43a89a37f74b9cc7b93d52f/README.md#pretrained-checkpoints)
We’ll use the largest model YOLOv5x (89M parameters), which is also the most accurate.

- To train a model on a custom dataset, we’ll call the train.py script. We’ll pass a couple of parameters:
```
img 640 - resize the images to 640x640 pixels
batch 16 - 16 images per batch
epochs 30 - train for 30 epochs
data ./data/sampledata.yaml - path to dataset config
cfg ./models/yolov5s.yaml - model config
weights yolov5s.pt - use pre-trained weights from the YOLOv5x model
name yolov5s_ev - name of our model
cache - cache dataset images for faster training
```
- We can train either 
   1) Using pretrained by passing ```--weights yolov5s.pt``` (recommended)
   2) Randomly initialized by passing  ```--weights ''``` (not recommended)

-  Pretrained weights are auto-downloaded from the latest [YOLOv5 release](https://github.com/ultralytics/yolov5/releases)




### Requirements ###
Python 3.7 or later with all requirements.txt dependencies installed, including torch=>1.6.0 and torchvision>=0.7.0 


To install run:
```
pip install -r requirements.txt
pip install torch==1.6.0 torchvision==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```



## Evaluation ## 
Mean Average Precision : The Mean Average Precision or mAP score is calculated by taking the mean AP over all classes and/or over all IoU thresholds, 
- mAP@.5 
- mAP@[.5:.95] 
For this task , we have taken the mAP that was averaged over both the object categories and all 10 IoU thresholds. As we can see below the mAP increased as training increased .
Best mAP@.5 : 652
Best mAP@[.5:.95]  : .348
![alt text](https://github.com/rvj07ai/EV/blob/main/Object_Detection/code/yolo-5s-model/runs/train/yolov5s_ev/results.png)

## Making predictions ##
Took some images  from the validation set and some from web and move them to inference/images to see how our model does on those:

We’ll use the detect.py script to run our model on the images. Here are the parameters we’re using:
```
python detect.py --device 0 --weights runs/train/yolov5s_ev/weights/best.pt --img 640 --conf 0.4 --source ./inference/images/ --name yolov5s_ev
```

```
weights runs/train/yolov5s_ev/weights/best.pt - checkpoint of the model
img 640 - resize the images to 640x640 px
conf 0.4 - take into account predictions with confidence of 0.4 or higher
source ./inference/images/ - path to the images
```
Eg: 

![alt text](https://github.com/rvj07ai/EV/blob/main/Object_Detection/code/yolo-5s-model/runs/detect/yolov5s_ev/image_000000043.jpg)



### Train on Custom Data  ###
- The training took around 2 hour  on GeForce RTX 2060. 
- The best model checkpoint is saved to runs/train/yolo5s_ev/weights/best.pt.

```
D:\Object_Detection\code\yolo-5s-model>python train.py --img 640 --batch 16 --epochs 30 --data ./data/sampledata.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt  --name yolov5s_ev --cache
Using torch 1.6.0+cu101 CUDA:0 (GeForce RTX 2060, 6144MB)

Namespace(adam=False, batch_size=16, bucket='', cache_images=True, cfg='./models/yolov5s.yaml', data='./data/sampledata.yaml', device='', epochs=30, evolve=False, exist_ok=False, global_rank=-1, hyp='data/hyp.scratch.yaml', image_weights=False, img_size=[640, 640], local_rank=-1, log_imgs=16, multi_scale=False, name='yolov5s_ev', noautoanchor=False, nosave=False, notest=False, project='runs/train', rect=False, resume=False, save_dir='runs\\train\\yolov5s_ev', single_cls=False, sync_bn=False, total_batch_size=16, weights='yolov5s.pt', workers=8, world_size=1)
Start Tensorboard with "tensorboard --logdir runs/train", view at http://localhost:6006/
Hyperparameters {'lr0': 0.01, 'lrf': 0.2, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1, 'box': 0.05, 'cls': 0.5, 'cls_pw': 1.0, 'obj': 1.0, 'obj_pw': 1.0, 'iou_t': 0.2, 'anchor_t': 4.0, 'fl_gamma': 0.0, 'hsv_h': 
0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0}
{'train': 'coco/images/train/', 'val': 'coco/images/val/', 'nc': 2, 'names': ['person', 'car']}

                 from  n    params  module                                  arguments
  0                -1  1      3520  models.common.Focus                     [3, 32, 3]
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]
  2                -1  1     19904  models.common.BottleneckCSP             [64, 64, 1]
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]
  4                -1  1    161152  models.common.BottleneckCSP             [128, 128, 3]
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]
  6                -1  1    641792  models.common.BottleneckCSP             [256, 256, 3]
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]
  8                -1  1    656896  models.common.SPP                       [512, 512, [5, 9, 13]]        
  9                -1  1   1248768  models.common.BottleneckCSP             [512, 512, 1, False]
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']
 12           [-1, 6]  1         0  models.common.Concat                    [1]
 13                -1  1    378624  models.common.BottleneckCSP             [512, 256, 1, False]
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]
 19          [-1, 14]  1         0  models.common.Concat                    [1]
 20                -1  1    313088  models.common.BottleneckCSP             [256, 256, 1, False]
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]
 22          [-1, 10]  1         0  models.common.Concat                    [1]
 23                -1  1   1248768  models.common.BottleneckCSP             [512, 512, 1, False]
 24      [17, 20, 23]  1     18879  models.yolo.Detect                      [2, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model Summary: 283 layers, 7257791 parameters, 7257791 gradients

Transferred 362/370 items from yolov5s.pt
Optimizer groups: 62 .bias, 70 conv.weight, 59 other
Scanning 'coco\labels\train.cache' for images and labels... 1791 found, 0 missing, 0 empty, 0 corrupted: 100%|█| 1791/1791 [00:00<?, ?it/s 
Caching images (1.6GB): 100%|█████████████████████████████████████████████████████████████████████████| 1791/1791 [00:26<00:00, 67.95it/s] 
Scanning 'coco\labels\val.cache' for images and labels... 448 found, 0 missing, 0 empty, 0 corrupted: 100%|█████| 448/448 [00:00<?, ?it/s] 
Caching images (0.4GB): 100%|███████████████████████████████████████████████████████████████████████████| 448/448 [00:09<00:00, 48.49it/s] 

Analyzing anchors... anchors/target = 4.70, Best Possible Recall (BPR) = 0.9987
Image sizes 640 train, 640 test
Using 8 dataloader workers
Logging results to runs\train\yolov5s_ev
Starting training for 30 epochs...

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
      0/29     2.59G   0.09877   0.07921   0.02435    0.2023       175       640: 100%|█████████████████| 112/112 [03:58<00:00,  2.13s/it] 
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [01:06<00:00,  2.39s/it]
                 all         448    3.41e+03       0.105       0.665       0.373       0.128

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
      1/29     2.96G   0.06973   0.06546   0.01302    0.1482       176       640: 100%|█████████████████| 112/112 [03:34<00:00,  1.91s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:31<00:00,  1.14s/it]
                 all         448    3.41e+03       0.167       0.774       0.502       0.202

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
      2/29     2.96G   0.06391   0.06419  0.008507    0.1366       167       640: 100%|█████████████████| 112/112 [03:34<00:00,  1.92s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:30<00:00,  1.10s/it]
                 all         448    3.41e+03       0.202       0.788       0.486       0.188

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
      3/29     2.96G   0.05944   0.06441  0.007309    0.1312       207       640: 100%|█████████████████| 112/112 [03:37<00:00,  1.94s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:29<00:00,  1.06s/it]
                 all         448    3.41e+03       0.295       0.769       0.591       0.253

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
      4/29     2.96G   0.05906   0.06452  0.006939    0.1305       232       640: 100%|█████████████████| 112/112 [03:37<00:00,  1.94s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:30<00:00,  1.09s/it]
                 all         448    3.41e+03       0.204       0.782       0.471       0.176

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
      5/29     2.96G   0.05823    0.0653  0.006855    0.1304       183       640: 100%|█████████████████| 112/112 [03:36<00:00,  1.93s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:29<00:00,  1.04s/it]
                 all         448    3.41e+03       0.337       0.766       0.639       0.305

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
      6/29     2.96G   0.05471   0.06487  0.006479    0.1261       189       640: 100%|█████████████████| 112/112 [03:38<00:00,  1.95s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:28<00:00,  1.02s/it]
                 all         448    3.41e+03       0.361        0.76        0.64         0.3

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
      7/29     2.96G   0.05553   0.06494  0.006289    0.1268       265       640: 100%|█████████████████| 112/112 [03:39<00:00,  1.96s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:28<00:00,  1.03s/it]
                 all         448    3.41e+03       0.327       0.757       0.627       0.293

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
      8/29     2.96G   0.05409   0.06395  0.006248    0.1243       183       640: 100%|█████████████████| 112/112 [03:41<00:00,  1.98s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:28<00:00,  1.02s/it]
                 all         448    3.41e+03       0.367       0.731       0.607       0.268

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
      9/29     2.96G   0.05212   0.06339  0.006158    0.1217       246       640: 100%|█████████████████| 112/112 [03:42<00:00,  1.99s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:29<00:00,  1.04s/it]
                 all         448    3.41e+03       0.392       0.728       0.636       0.315

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     10/29     2.96G   0.05095   0.06475  0.005797    0.1215       246       640: 100%|█████████████████| 112/112 [03:47<00:00,  2.03s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:29<00:00,  1.05s/it]
                 all         448    3.41e+03       0.396       0.737       0.648       0.319

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     11/29     2.96G   0.04882   0.06288  0.005625    0.1173       198       640: 100%|█████████████████| 112/112 [03:48<00:00,  2.04s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:28<00:00,  1.03s/it]
                 all         448    3.41e+03       0.351       0.751       0.643       0.317

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     12/29     2.96G   0.04845     0.064  0.005415    0.1179       157       640: 100%|█████████████████| 112/112 [03:47<00:00,  2.03s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:28<00:00,  1.02s/it]
                 all         448    3.41e+03       0.406       0.723       0.633       0.314

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     13/29     2.96G   0.04792   0.06428  0.005425    0.1176       221       640: 100%|█████████████████| 112/112 [03:47<00:00,  2.03s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:28<00:00,  1.03s/it]
                 all         448    3.41e+03       0.366       0.739       0.641       0.326

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     14/29     2.96G   0.04683   0.06268  0.005035    0.1146       204       640: 100%|█████████████████| 112/112 [03:47<00:00,  2.03s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:28<00:00,  1.02s/it]
                 all         448    3.41e+03       0.369       0.736       0.631       0.323

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     15/29     2.96G   0.04636   0.06218  0.004943    0.1135       217       640: 100%|█████████████████| 112/112 [03:47<00:00,  2.03s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:28<00:00,  1.01s/it]
                 all         448    3.41e+03       0.372       0.735       0.641       0.331

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     16/29     2.96G   0.04576   0.06152  0.004722     0.112       214       640: 100%|█████████████████| 112/112 [03:48<00:00,  2.04s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:28<00:00,  1.03s/it]
                 all         448    3.41e+03       0.375       0.743       0.637       0.329

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     17/29     2.96G   0.04541   0.06003  0.004697    0.1101       212       640: 100%|█████████████████| 112/112 [03:51<00:00,  2.06s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:28<00:00,  1.01s/it]
                 all         448    3.41e+03       0.398       0.737       0.643       0.336

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     18/29     2.96G   0.04491   0.06213  0.004695    0.1117       160       640: 100%|█████████████████| 112/112 [03:49<00:00,  2.05s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:28<00:00,  1.02s/it]
                 all         448    3.41e+03       0.408       0.733       0.642       0.336

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     19/29     2.96G   0.04471   0.06014  0.004474    0.1093       211       640: 100%|█████████████████| 112/112 [03:50<00:00,  2.06s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:28<00:00,  1.02s/it]
                 all         448    3.41e+03       0.382       0.743       0.648       0.335

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     20/29     2.96G   0.04364   0.05949  0.004379    0.1075       259       640: 100%|█████████████████| 112/112 [03:50<00:00,  2.06s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:28<00:00,  1.03s/it]
                 all         448    3.41e+03       0.383       0.746       0.647       0.336

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     21/29     2.96G   0.04372   0.05796   0.00421    0.1059       178       640: 100%|█████████████████| 112/112 [03:49<00:00,  2.05s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:28<00:00,  1.01s/it]
                 all         448    3.41e+03       0.393       0.741       0.654       0.345

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     22/29     2.96G   0.04328   0.05784  0.004141    0.1053       174       640: 100%|█████████████████| 112/112 [03:52<00:00,  2.07s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:28<00:00,  1.01s/it]
                 all         448    3.41e+03       0.429        0.72       0.646       0.341

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     23/29     2.96G   0.04276    0.0575  0.004048    0.1043       184       640: 100%|█████████████████| 112/112 [03:49<00:00,  2.05s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:27<00:00,  1.01it/s]
                 all         448    3.41e+03       0.397       0.741       0.654       0.345

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     24/29     2.96G   0.04248   0.05721   0.00408    0.1038       189       640: 100%|█████████████████| 112/112 [03:49<00:00,  2.05s/it]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:28<00:00,  1.02s/it]
                 all         448    3.41e+03       0.415       0.728       0.647       0.344

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:27<00:00,  1.00it/s] 
                 all         448    3.41e+03       0.399       0.744       0.649       0.338

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     26/29     2.96G   0.04167    0.0559  0.004016    0.1016       213       640: 100%|█████████████████| 112/112 [03:47<00:00,  2.04s/it] 
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:28<00:00,  1.00s/it] 
                 all         448    3.41e+03       0.434       0.724       0.651       0.347

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     27/29     2.96G     0.042   0.05793  0.003683    0.1036       185       640: 100%|█████████████████| 112/112 [03:48<00:00,  2.04s/it] 
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:27<00:00,  1.01it/s] 
                 all         448    3.41e+03       0.424       0.726       0.649       0.345

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     28/29     2.96G   0.04135   0.05449  0.003759   0.09959       142       640: 100%|█████████████████| 112/112 [03:50<00:00,  2.06s/it] 
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:28<00:00,  1.01s/it] 
                 all         448    3.41e+03       0.454       0.711       0.645       0.346

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     29/29     2.96G   0.04121   0.05339  0.003721   0.09832       187       640: 100%|█████████████████| 112/112 [03:50<00:00,  2.06s/it] 
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|███████| 28/28 [00:33<00:00,  1.18s/it] 
                 all         448    3.41e+03       0.437        0.73       0.652       0.348
Optimizer stripped from runs\train\yolov5s_ev\weights\last.pt, 14.8MB
Optimizer stripped from runs\train\yolov5s_ev\weights\best.pt, 14.8MB
30 epochs completed in 2.151 hours.

```

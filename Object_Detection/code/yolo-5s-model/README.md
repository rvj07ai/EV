(venv) D:\EagleView\Object_Detection\code\yolo-5s-model>python train.py --img 640 --batch 16 --epochs 20 --data ./data/sampledata.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name yolov5s_eagleview --cache
Using torch 1.6.0+cu101 CUDA:0 (GeForce RTX 2060, 6144MB)

Namespace(adam=False, batch_size=16, bucket='', cache_images=True, cfg='./models/yolov5s.yaml', data='./data/sampledata.yaml', device='', epochs=20, evolve=False, exist_ok=False, global_rank=-1, hyp='data/hyp.scratch.yaml', image_weights=False, img_size=[640, 640], local_rank=-1, log_imgs=16, multi_scale=False, name='yolov5s_eagleview', noautoanchor=False, nosave=False, notest=False, project='runs/train', rect=False, resume=False, save_dir='runs\\train\\yolov5s_eagleview5', single_cls=False, sync_bn=False, total_batch_size=16, weights='yolov5s.pt', workers=8, world_size=1)
Start Tensorboard with "tensorboard --logdir runs/train", view at http://localhost:6006/
Hyperparameters {'lr0': 0.01, 'lrf': 0.2, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1, 'box': 0.05, 'cls': 0.5, 'cls_pw': 1.0, 'obj': 1.0, 'obj_pw': 1.0, 'iou_t': 0.2, 'anchor_t': 4.0, 'fl_gamma': 
0.0, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0}
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
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']
 16           [-1, 4]  1         0  models.common.Concat                    [1]
 17                -1  1     95104  models.common.BottleneckCSP             [256, 128, 1, False]
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
Scanning 'coco\labels\train' for images and labels... 1791 found, 0 missing, 0 empty, 0 corrupted: 100%|█| 1791/1791 [00:20<00:00, 
New cache created: coco\labels\train.cache
Scanning 'coco\labels\train.cache' for images and labels... 1791 found, 0 missing, 0 empty, 0 corrupted: 100%|█| 1791/1791 [00:00<?
Caching images (1.6GB): 100%|██████████████████████████████████████████████████████████████████| 1791/1791 [00:18<00:00, 97.39it/s]
Scanning 'coco\labels\val' for images and labels... 448 found, 0 missing, 0 empty, 0 corrupted: 100%|█| 448/448 [00:07<00:00, 59.36
New cache created: coco\labels\val.cache
Scanning 'coco\labels\val.cache' for images and labels... 448 found, 0 missing, 0 empty, 0 corrupted: 100%|█| 448/448 [00:00<?, ?it
Caching images (0.4GB): 100%|████████████████████████████████████████████████████████████████████| 448/448 [00:06<00:00, 64.72it/s]

Analyzing anchors... anchors/target = 4.70, Best Possible Recall (BPR) = 0.9987
Image sizes 640 train, 640 test
Using 8 dataloader workers
Logging results to runs\train\yolov5s_eagleview5
Starting training for 20 epochs...

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
      0/19     2.89G   0.09876   0.07923   0.02435    0.2023       175       640: 100%|██████████| 112/112 [01:16<00:00,  1.46it/s]
                 all         448    3.41e+03       0.194       0.778       0.455       0.165

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
      3/19     3.17G   0.05923    0.0645  0.007296     0.131       207       640: 100%|██████████| 112/112 [00:50<00:00,  2.21it/s] 
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|█| 28/28 [00:20<00:00,  1.37it/s 
                 all         448    3.41e+03       0.279       0.767        0.57       0.237

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
      4/19     3.17G   0.05842   0.06453  0.006927    0.1299       232       640: 100%|██████████| 112/112 [00:49<00:00,  2.26it/s] 
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|█| 28/28 [00:21<00:00,  1.33it/s 
                 all         448    3.41e+03       0.253       0.779       0.534       0.228

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
      5/19     3.17G   0.05792   0.06532  0.006831    0.1301       183       640: 100%|██████████| 112/112 [00:58<00:00,  1.92it/s] 
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|█| 28/28 [00:19<00:00,  1.41it/s 
                 all         448    3.41e+03       0.325       0.768       0.635       0.302

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
      6/19     3.17G   0.05478    0.0647  0.006494     0.126       189       640: 100%|██████████| 112/112 [00:47<00:00,  2.34it/s] 
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|█| 28/28 [00:19<00:00,  1.42it/s
                 all         448    3.41e+03       0.347       0.763       0.643       0.306

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
      7/19     3.17G   0.05498   0.06481  0.006155    0.1259       265       640: 100%|██████████| 112/112 [00:47<00:00,  2.34it/s]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|█| 28/28 [00:18<00:00,  1.55it/s
                 all         448    3.41e+03       0.319       0.764       0.623        0.29

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
      8/19     3.17G   0.05319   0.06374  0.006161    0.1231       183       640: 100%|██████████| 112/112 [00:46<00:00,  2.43it/s]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|█| 28/28 [00:16<00:00,  1.70it/s
                 all         448    3.41e+03       0.367       0.733       0.616       0.278

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
      9/19     3.17G   0.05137   0.06311  0.006029    0.1205       246       640: 100%|██████████| 112/112 [00:47<00:00,  2.37it/s]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|█| 28/28 [00:17<00:00,  1.62it/s
                 all         448    3.41e+03       0.376       0.747       0.652       0.321

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     10/19     3.17G   0.04925   0.06428  0.005613    0.1191       246       640: 100%|██████████| 112/112 [00:45<00:00,  2.44it/s]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|█| 28/28 [00:18<00:00,  1.50it/s
                 all         448    3.41e+03       0.414       0.741       0.661        0.33

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     11/19     3.17G   0.04784   0.06216  0.005491    0.1155       198       640: 100%|██████████| 112/112 [00:46<00:00,  2.42it/s]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|█| 28/28 [00:17<00:00,  1.65it/s
                 all         448    3.41e+03        0.38       0.746        0.65       0.323

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     12/19     3.17G   0.04743   0.06284  0.005161    0.1154       157       640: 100%|██████████| 112/112 [00:45<00:00,  2.45it/s]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|█| 28/28 [00:16<00:00,  1.67it/s
                 all         448    3.41e+03       0.403       0.738       0.649       0.333

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     13/19     3.17G   0.04648   0.06318  0.005285    0.1149       221       640: 100%|██████████| 112/112 [00:46<00:00,  2.43it/s]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|█| 28/28 [00:16<00:00,  1.66it/s
                 all         448    3.41e+03       0.379       0.758       0.655       0.337

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     14/19     3.17G   0.04552   0.06135  0.004746    0.1116       204       640: 100%|██████████| 112/112 [00:45<00:00,  2.44it/s]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|█| 28/28 [00:17<00:00,  1.64it/s
                 all         448    3.41e+03       0.375       0.755       0.655       0.343

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     15/19     3.17G   0.04486   0.06113  0.004656    0.1106       217       640: 100%|██████████| 112/112 [00:45<00:00,  2.45it/s]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|█| 28/28 [00:16<00:00,  1.71it/s
                 all         448    3.41e+03        0.38        0.75       0.656        0.34

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     16/19     3.17G   0.04426   0.06022  0.004466    0.1089       214       640: 100%|██████████| 112/112 [00:45<00:00,  2.44it/s]
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|█| 28/28 [00:16<00:00,  1.66it/s 
                 all         448    3.41e+03       0.401       0.757       0.661       0.345

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     17/19     3.17G   0.04402    0.0588  0.004332    0.1071       212       640: 100%|██████████| 112/112 [00:47<00:00,  2.36it/s] 
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|█| 28/28 [00:20<00:00,  1.38it/s 
                 all         448    3.41e+03       0.408       0.752       0.666       0.348

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     18/19     3.17G    0.0438   0.06094  0.004485    0.1092       160       640: 100%|██████████| 112/112 [00:45<00:00,  2.45it/s] 
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|█| 28/28 [00:16<00:00,  1.68it/s 
                 all         448    3.41e+03       0.412       0.752       0.669       0.352

     Epoch   gpu_mem       box       obj       cls     total   targets  img_size
     19/19     3.17G   0.04361   0.05906  0.004261    0.1069       211       640: 100%|██████████| 112/112 [00:45<00:00,  2.44it/s] 
               Class      Images     Targets           P           R      mAP@.5  mAP@.5:.95: 100%|█| 28/28 [00:19<00:00,  1.43it/s 
                 all         448    3.41e+03       0.427       0.744       0.666        0.35
Optimizer stripped from runs\train\yolov5s_eagleview5\weights\last.pt, 14.8MB
Optimizer stripped from runs\train\yolov5s_eagleview5\weights\best.pt, 14.8MB
20 epochs completed in 0.386 hours.
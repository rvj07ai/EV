## Pre-process the data ##
- Convert2Yolo - Convert the coco format data into Yolo format
- Split into Training and Validation set 


### convert2Yolo ###

First things First , we need to convert the coco format data into Yolo format , we are going to use the utilities in this repo to achieve the same



Object Detection annotation Convert to [Yolo Darknet](https://pjreddie.com/darknet/yolo/) Format

Support DataSet : 

1. COCO
2. VOC
3. KITTI 2D Object Detection

​    

## Pre-Requirement (Windows 10)

```
python3.7
virtualenv venv 
venv\Scripts\activate
pip3 install -r requirements.txt
```

​    

## Required Parameters

Each dataset requires some parameters :

See [convert.py](https://github.com/rvj07ai/EV/blob/main/Object_Detection/code/convertToYolo/convert.py)

1. --datasets

   - like a COCO / VOC / KITTI

     ```bash
     --datasets COCO
     ```

2. --img_path

   - it directory path. not file path

     ```bash
     --img_path ./coco/images
     ```

3. --label

   - it directory path. not file path

     (some datasets give label `*.json` or `*.csv` . this case use file path)

     ```bash
     --label ./coco/annotations/bbox-annotations.json
     ```
     
     or
     
     --label ../coco/annotations/bbox-annotations.csv
     ```

4. --convert_output_path

   - it directory path. not file path

     ```bash
     --convert_output_path ./coco/labels 
     ```

5. --img_type

   - like a `*.png`, `*.jpg`

     ```bash
     --img_type ".jpg"
     ```


6. --cla_list_file(`*.names`)

   - it is `*.names` file contain class name. refer [darknet `*.name` file](https://github.com/pjreddie/darknet/blob/master/data/voc.names)

     ```bash
     --cls_list_file ./coco/coco.names.txt
     ```

Names w.r.t to this problem statement 



### *.names file example

```
person
car
```

​    

#### Description of Input dataset directory

The base coco dataset location are `~/coco` and coco folder contains `annotations`, `images` folder


**annotations**

```bash
.
└── box-annotations.json
```

​    

**master_images**

```bash
.
├── image_000000001.jpg
├── image_000000002.jpg
├── image_000000003.jpg
├── image_000000004.jpg
...
├── image_000002236.jpg
├── image_000002237.jpg
├── image_000002238.jpg
└── image_000002239.jpg
```

#### make `names` file

now make `coco.names.txt` file in `~/coco/`

refer [darknet `coco.names` file](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

```bash
person
car
```

​    

#### COCO dataset convert to YOLO format

Now execute example code. 

Example : 

     ```bash
     python convert.py --datasets COCO --img_path ./coco/images --label ./coco/annotations/bbox-annotations.json --convert_output_path ./coco/labels --img_type ".jpg"  --cls_list_file ./coco/coco.names.txt
     ```

**COCO convert to YOLO**
COCO Parsing:  |████████████████████████████████████████| 100.0% (16772/16772)  Complete


YOLO Generating:|████████████████████████████████████████| 100.0% (2239/2239)  Complete


YOLO Saving:   |████████████████████████████████████████| 100.0% (2239/2239)  Complete

​        

#### Result

Now check result files (`~/coco/labels`)

**`~/coco/labels`**

```bash
.
├── image_000000001.txt
├── image_000000002.txt
├── image_000000003.txt
├── image_000000004.txt
...
...
├── image_000002236.txt
├── image_000002237.txt
├── image_000002238.txt
└── image_000002239.txt
```

​    

**`image_000000001.txt`**

```bash
0 0.897 0.499 0.143 0.621
0 0.914 0.64 0.171 0.717
1 0.109 0.26 0.073 0.105
1 0.471 0.58 0.643 0.837
```

​     
​    

### Split the dataset into train and valid as below

![alt text](https://github.com/rvj07ai/EV/blob/main/Object_Detection/code/convertToYolo/train_data.JPG)





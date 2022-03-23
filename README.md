# Rotate - Yolov5


## Requirements
-   CUDA >= 11.2
#### Install require package
    pip install -r requirements.txt
#### Install extensions for CUDA
    cd utils/iou_cuda
    python setup.py install
## Dataset 
- Use dataset similar to yolov5, but change in label: [cx, cy, w, h, real, imagine]
- Check Convert_labels.ipynb to convert labels and prepare dataset tree.
- Edit path and number of classes in data/rotate_cepdof.yaml
## Train
#### Config
- Edit number of classes in models/rotate_yolov5s_cepdof.yaml
- Run utils.rotate_kmean_anchors to generate predefined anchors for your specific datasets with specified image size.
#### Training from scratch
    python rotate_train.py --weights '' --data data/rotate_cepdof.yaml --cfg models/rotate_yolov5s_cepdof.yaml --rotate
    


## Detect
    python rotate_detect.py --weights rotate_best.pt --img 1024 \
    --conf 0.75 --source Lunch2_000001.jpg --iou-thres 0.4 

## Export ONNX
    export ONNX_EXPORT=1
    python models/export.py --weights weights/rotate_best.pt --include onnx --dynamic --opset-version 10


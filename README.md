# TensorRT Convert & detection

## TRT 변환

### setup

```bash
pip3 install --upgrade setuptools pip --user
pip3 install nvidia-pyindex
pip3 install --upgrade nvidia-tensorrt
pip3 install pycuda
```

### Onnx 파일 변환 (NMS 포함)

- yolov7 레포지토리의 [export.py](http://export.py) 활용

```bash
python3 export.py --weights last.pt --grid --end2end --simplify --topk-all 80 --iou-thres 0.45 --conf-thres 0.1 --img-size 640 640
```

### TensorRT 변환

- -o : onnx 파일 경로
- -e : 변환할 tensorRT 엔진파일 경로 및 파일명

```bash
python3 ./tensorRT_convert/export.py -o yolo-tiny.onnx -e yolov7-tiny-nms.trt -p fp16
```

## TRT detection

- —model : tensorRT engine 파일 경로 입력
- —data_file : data yaml 파일 경로 입력
- —video : detection 수행할 비디오 파일 경로 입력

```bash
python3 trt_video_detection.py --model yolov7-tiny-nms.trt --data_file data/coco.yaml --video city.mp4
```

실행 결과 

yolov7

![Untitled](img/yolov7.png)

yolov7-tiny

![Untitled](img/yolo-tiny.png)

import cv2
import torch
import time
import argparse
from numpy import random
import yaml

from utils.trt_yolo_plugin import TRT_engine
from utils.detect_uilts import add_camera_args, Detect, open_window

WINDOW_NAME ='TensorRT object detecion'
device = torch.device('cuda:0')


def parse_args():
    desc = ('Display real-time object detection in video file with TensorRT optimized YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--model', type=str, required=True, help='set the path of engine file ex: ./yolo.engine')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--data_file', type=str, default='coco.yaml',help='path of data yaml file')

    args = parser.parse_args()
    return args



def postprocess(boxes, ratio, dwdh):
    dwdh = torch.tensor(dwdh*2).to(boxes.device)
    boxes -= dwdh
    boxes /= ratio
    return boxes

def visualize(img,boxes, scores, classes,names, colors, ratio,dwdh,fps):
    for box, score, cl in zip(boxes, scores, classes):
        box = postprocess(box, ratio, dwdh).round().int()
        name = names[cl]
        color = colors[name]
        name += ' ' + str(round(float(score), 3))
        img = cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), color, 2)
        img =cv2.putText(img, name, (int(box[0]), int(box[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1)
        img = cv2.putText(img, 'FPS: '+str(round(fps,1)), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,255),thickness=2)
    return img


def loop_and_detect(cam, trt_yolo, args, class_list):
    colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(class_list)}

    fps = 0.0
    tic = time.time()

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break

        boxes, scores, classes, r, dwdh = trt_yolo.predict(img, threshold=0.5)
        result_image = visualize(img, boxes, scores, classes,class_list, colors, r, dwdh,fps)
        cv2.imshow(WINDOW_NAME, result_image)

        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps * 0.95 + curr_fps * 0.05)

        tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # Esc키로 quit
            break

def main():
    args = parse_args()
    cam = Detect(args)
    with open(args.data_file) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    cls_list = data_dict['names']
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')
    open_window(
        'TensorRT object detecion', 'TensorRT object detecion',
        cam.img_width, cam.img_height)
    trt_yolo = TRT_engine(args.model)
    # trt_yolo = TRT_yolo(args.model, args.img_size, args.half, cls_list)
    loop_and_detect(cam, trt_yolo, args, cls_list)
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

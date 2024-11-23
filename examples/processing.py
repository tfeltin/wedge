from utils import get_region_boxes, nms
from torch import FloatTensor
from cv2 import resize
import numpy as np


class_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

detection_information = {'num_classes': 80, 'anchors': [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828], 'num_anchors': 5}

def postprocess(preds):
    if preds.shape == (0,):
        return ""
    conf_thresh=0.5
    nms_thresh=0.4
    num_classes, anchors, num_anchors = [detection_information[k] for k in detection_information.keys()]
    
    output = FloatTensor(preds.squeeze(0))
    boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)[0]
    boxes = nms(boxes, nms_thresh)
    
    detections = []
    for box in boxes:
        x = box[0].item()
        y = box[1].item()
        w = box[2].item()
        h = box[3].item()
        conf = box[4].item()
        max_conf = box[5].item()
        cls = class_names[int(box[6].item())]
        detections.append("%s\t-\tbb:[%.2f,%.2f,%.2f,%.2f],\tconf:%.3f,\tmax_conf:%.3f" % (cls, x, y, w, h, conf, max_conf))
    
    return "\n".join(detections)

def preprocess(image, input_shape=(416,416)):
    image = resize(image, tuple(input_shape))
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)
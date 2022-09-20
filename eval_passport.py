print('Loading...')

# basic python and ML Libraries
import sys
import numpy as np

# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# We will be reading images using OpenCV
import cv2

# torchvision libraries
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# for image augmentations
from albumentations.pytorch.transforms import ToTensorV2

image_path = sys.argv[1]

# the function takes the original prediction and the iou threshold.
def apply_nms(orig_prediction, iou_thresh=0.3):
  # torchvision returns the indices of the bboxes to keep
  keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
  final_prediction = orig_prediction
  final_prediction['boxes'] = final_prediction['boxes'][keep]
  final_prediction['scores'] = final_prediction['scores'][keep]
  final_prediction['labels'] = final_prediction['labels'][keep]
  return final_prediction

# read image and convert to correct size and color  
img = cv2.imread(image_path)
h, w, c = img.shape
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
img = cv2.resize(img, (w, h), cv2.INTER_AREA) / 255
img = ToTensorV2(p=1.0)(image=img)['image']

# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# replace the pre-trained head with a new one
in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = 2
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

# load parameters that were trained on locating mrz areas
model.load_state_dict(torch.load('model.dict', map_location=torch.device('cpu')))
model.eval()

# get predictions for current image
print('Predicting...')
with torch.no_grad():
  prediction = model([img])[0]

# remove extra bounding boxes
nms_prediction = apply_nms(prediction, iou_thresh=0.01)

print(nms_prediction)

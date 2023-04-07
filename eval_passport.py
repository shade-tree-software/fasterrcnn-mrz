print('Loading...')

# basic python and ML Libraries
import sys
import numpy as np

# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

# torchvision libraries
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# for image augmentations
from albumentations.pytorch.transforms import ToTensorV2

if len(sys.argv) != 3:
  print(f'usage: {sys.argv[0]} <image file> <model dict file>')
  exit(0)

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
model.load_state_dict(torch.load(sys.argv[2], map_location=torch.device('cpu')))
model.eval()

# get predictions for current image
print('Predicting...')
with torch.no_grad():
  prediction = model([img])[0]

# remove extra bounding boxes
nms_prediction = apply_nms(prediction, iou_thresh=0.01)

print(nms_prediction)

def plot_img_bbox(img, target):
  fig, a = plt.subplots(1,1)
  fig.set_size_inches(5,5)
  a.imshow(img.numpy().transpose((1,2,0)))
  box= target['boxes'][0]
  # Convert to x-min y-min width height
  x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
  rect = patches.Rectangle(
    (x, y),
    width, height,
    linewidth = 2,
    edgecolor = 'r',
    facecolor = 'none'
  )
  # Draw the bounding box on top of the image
  a.add_patch(rect)
  plt.show()

plot_img_bbox(img, nms_prediction)

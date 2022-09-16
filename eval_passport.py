# basic python and ML Libraries
import os
import sys
import random
import numpy as np
import pandas as pd

# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# We will be reading images using OpenCV
import cv2

# torchvision libraries
import torch
import torchvision
from torchvision import transforms as torchtrans  
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# for image augmentations
from albumentations.pytorch.transforms import ToTensorV2

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
  return torchtrans.ToPILImage()(img).convert('RGB')

# Function to visualize bounding boxes in the image
def plot_img_bbox(img, target):
  # plot the image and bboxes
  # Bounding boxes are defined as follows: x-min y-min width height
  fig, a = plt.subplots(1,1)
  fig.set_size_inches(5,5)
  a.imshow(img)
  for box in (target['boxes']):
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

# read image and convert to correct size and color    
img = cv2.imread(image_path)
h, w, c = img.shape
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
img = cv2.resize(img, (w, h), cv2.INTER_AREA) / 255
img = ToTensorV2(p=1.0)(image=img)['image']

# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
num_classes = 2
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

model.load_state_dict(torch.load('model.dict', map_location=torch.device('cpu')))
model.eval()
with torch.no_grad():
  prediction = model([img])[0]
nms_prediction = apply_nms(prediction, iou_thresh=0.01)
print(nms_prediction)
#plot_img_bbox(torch_to_pil(img), nms_prediction)

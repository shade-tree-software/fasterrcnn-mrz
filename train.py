# basic python and ML Libraries
import sys
import os
import time

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# torchvision libraries
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from PIL import Image

# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

IMG_SIZE = 480
VALIDATION_RATIO = 0.1
BATCH_SIZE = 10
NUM_CLASSES = 1
NUM_EPOCHS = 5

# defining the files directory and testing directory
train_dir = sys.argv[1] + '/train/'
test_dir = sys.argv[1] + '/test/'

class ImagesDataset(torch.utils.data.Dataset):

  def __init__(self, files_dir, img_size, transforms=None):
    self.transforms = transforms
    self.files_dir = files_dir
    self.height = img_size
    self.width = img_size
    self.imgs = [image for image in sorted(os.listdir(train_dir)) if image[-4:]=='.jpg']

  def __getitem__(self, idx):
    # read image, convert to rgb, and run any transforms 
    img_name = self.imgs[idx]
    image_path = os.path.join(self.files_dir, img_name)
    img = Image.open(image_path).convert('RGB') 
    if self.transforms is not None:
      img = self.transforms(img)
    
    # read annotations and convert from percentages to pixels
    annot_filename = img_name[:-4] + '.txt'
    annot_file_path = os.path.join(self.files_dir, annot_filename)
    annotations = {'boxes': [], 'labels': []}
    with open(annot_file_path) as f:
      for line in f:      
        class_label, box_center_x_pct, box_center_y_pct, box_width_pct, box_height_pct = \
            [float(x) for x in line.split(' ')]
        #annotations['labels'].append(int(class_label) + 1)
        annotations['labels'].append(1)
        box_min_x_pct = box_center_x_pct - box_width_pct/2
        box_max_x_pct = box_center_x_pct + box_width_pct/2
        box_min_y_pct = box_center_y_pct - box_height_pct/2
        box_max_y_pct = box_center_y_pct + box_height_pct/2
        box_min_x_pixels = int(box_min_x_pct*self.width)
        box_max_x_pixels = int(box_max_x_pct*self.width)
        box_min_y_pixels = int(box_min_y_pct*self.height)
        box_max_y_pixels = int(box_max_y_pct*self.height)
        annotations['boxes'].append([box_min_x_pixels, box_min_y_pixels, box_max_x_pixels, box_max_y_pixels])
    annotations['boxes'] = torch.as_tensor(annotations['boxes'], dtype=torch.float32)   
    annotations['labels'] = torch.as_tensor(annotations['labels'], dtype=torch.int64)

    return img, annotations

  def __len__(self):
    return len(self.imgs)

# Function to visualize bounding boxes in the image
def plot_img_bbox(img, target):
  fig, a = plt.subplots(1,1)
  fig.set_size_inches(5,5)
  a.imshow(img.numpy().transpose((1,2,0)))
  for box in (target['boxes']):
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
    
transform = transforms.Compose([
  transforms.Resize((IMG_SIZE, IMG_SIZE)),
  transforms.ToTensor(),
])

def collate_fn(batch):
  return tuple(zip(*batch))

# split into training set, validation set, and test set
full_ds = ImagesDataset(train_dir, IMG_SIZE, transforms=transform)
valid_size = int(VALIDATION_RATIO * len(full_ds))
train_size = len(full_ds) - valid_size
train_ds, valid_ds = torch.utils.data.random_split(full_ds, [train_size, valid_size])
train_dl = torch.utils.data.DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
valid_dl = torch.utils.data.DataLoader(valid_ds, valid_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
img, annotations = train_ds[0]
plot_img_bbox(img, annotations)
  
# load a pre-trained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES + 1)
# put the model on the gpu if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
print(f'Running on: {device}')

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.001)

# train the model
loss_hist_train = [0] * NUM_EPOCHS
loss_hist_valid = [0] * NUM_EPOCHS
for epoch in range(NUM_EPOCHS):
    start = time.time()
    model.train()
    for imgs, annotations in train_dl:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step() 
        loss_hist_train[epoch] += losses

    model.eval()
    with torch.no_grad():
      for imgs, annotations in valid_dl:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        preds = model(imgs)
        if len(preds) > 0 and 'boxes' in preds[0] and len(preds[0]["boxes"]) > 0:
          preds = torch.stack([pred['boxes'][0] for pred in preds], dim=0)
          annotations = torch.stack([annotation['boxes'][0] for annotation in annotations], dim=0)
          loss_fn = torch.nn.MSELoss()
          loss_hist_valid[epoch] += loss_fn(preds, annotations)
        else:
          loss_hist_valid[epoch] = 'N/A (no box predictions)'

    print(f'epoch: {epoch+1}, Train loss: {loss_hist_train[epoch]}, Validation loss: {loss_hist_valid[epoch]}')

torch.save(model.state_dict(), f'./model_state_{int(time.time())}.dict')
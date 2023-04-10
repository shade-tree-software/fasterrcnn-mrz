from PIL import Image
import sys
import os
import numpy as np
from torchvision import transforms as T

if len(sys.argv) != 2:
  print()
  print('Creates additional labeled training data for object detection projects by')
  print('taking images and their yolo-formatted object detection annotation files and')
  print('padding the images in nine different ways, thereby creating nine new image')
  print('files for each original image file.  Each of the new images is given a new')
  print('annotation file in which the bounding boxes have been updated to reflect the')
  print('changes introduced by the padding.')
  print()
  print(f'usage: {sys.argv[0]} <images and annotations directory>')
  print()
  exit(0)

offsets = [
  [0.5, 0.0],
  [0.25, 0.0],
  [0.0, 0.0],
  [0.5, 0.25],
  [0.25, 0.25],
  [0.0, 0.25],
  [0.5, 0.5],
  [0.25, 0.5],
  [0.0, 0.5],
]

def paddings(w, h, offsets):
  return [[int(w*wo*2), int(h*ho*2), int(w-w*wo*2), int(h-h*ho*2)] for wo,ho in offsets]

image_dir = sys.argv[1]
for image_filename in os.listdir(image_dir):
  if image_filename[-4:] in ['.jpg','.png']:
    im = Image.open(image_dir + '/' + image_filename)
    image_name, image_ext = os.path.splitext(image_filename)
    mrz_split = open(image_dir + '/' + image_name + '.txt').readlines()[0].split(' ')
    mrz_class = mrz_split[0]
    mrz = np.array(mrz_split[1:]).astype(float)

    for index, padding in enumerate(paddings(im.width, im.height, offsets)):
      im2 = T.Pad(padding=padding,padding_mode='edge')(im)
      new_image_name = f'padded_{padding[0]}_{padding[1]}_{padding[2]}_{padding[3]}_{image_name}'
      im2.save(image_dir + '/' + new_image_name + image_ext)
      with open(image_dir + '/' + new_image_name + '.txt', 'w') as f:
        f.write(f'{mrz_class} {mrz[0]/2 + offsets[index][0]} {mrz[1]/2 + offsets[index][1]} {mrz[2]/2} {mrz[3]/2}')

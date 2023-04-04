# passports
FTA_DUCSS.ipynb is the Google Colab notebook used to train the NN.  It was developed to detect traffic cones but it works just as well to detect MRZs if you upload the right training data.  I've already trained the NN to detect MRZs, and I've saved the params as model.dict.  If you have better training data and you want to retrain, just run this notebook in Google Colab and upload your data instead of the traffic cones dataset.  Make sure you're on a GPU runtime and run through the notebook to train the NN.  After you train it, you'll have to manually save the learned params by doing a:
```
     torch.save(model.state_dict(), 'model.dict')
```
Then download the file to your machine.

The content directory contains the training data which is basically passport images (.jpg or .png) and .txt annotation files.  The annotation files indicate the MRZ locations for the corresponding .jpg or .png image.  A typical annotation file will look like:
```
     0 0.497693 0.861967 0.928011 0.140984
```
where the values are class, center-x, center-y, width, height.  The class is always 0 which means MRZ.  If we had more than one object type in addition to MRZs, there would be other values greater than 0.  The other four numbers represent a percentage of the width and height of the picture, so a center-x value of 0.497693 means that the center-x of the MRZ is roughly halfway between the left and right edges of the picture.

augment.py is an optional utility that will create nine new passport images and annotation files for each existing passport image and annotation file by padding the image in various ways.  This can help with training by teaching the NN how to detect passports that are not cropped and centered in the middle of the image.

eval_pasport.py is the main MRZ detection code.  It will load the base NN and then populate it with the learned params in model.dict, then it evaluates the image and returns one or more predictions along with confidence values.  Usually the first prediction is the correct MRZ.  The rest can be ignored.  Note that the format of the MRZ predictions is different than the annotation files.  Coordinates are [xmin, ymin, xmax, ymax] in pixels.
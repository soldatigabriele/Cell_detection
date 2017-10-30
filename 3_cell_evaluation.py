
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[ ]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO

from PIL import Image

import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from shutil import copyfile

# ## Env setup

# In[ ]:

# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

# This is needed to display the images.
# get_ipython().magic(u'matplotlib inline')

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[ ]:


from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[ ]:


# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'cell_graph_inception'
MODEL_NAME = 'cell_graph_500_mobilenet'
# MODEL_NAME = 'cell_graph'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 1
AREA_THRESH = 0.3
SCORE_THRESH = 0.1

ERRATE_PV = 'ripulito/pv/'
ERRATE_VP = 'ripulito/vp/'
CORRETTE_PP = 'ripulito/pp/'

FILE_NAME='summaries/summary_{}_senzah50_{}soglia{}.csv'.format(MODEL_NAME,AREA_THRESH,SCORE_THRESH)

with open(FILE_NAME, 'w') as f:
    f.write('dataset,model,pp,pv,pt,vp,vv,vt,perc_full,perc_empty,perc_tot\n')

c1 = 0
c2 = 0
w1 = 0
w2 = 0
ft = 0
et = 0

# for the_file in os.listdir(ERRATE_PV):
#     file_path = os.path.join(ERRATE_PV, the_file)
#     if os.path.isfile(file_path):
#         os.unlink(file_path)
#
for the_file in os.listdir(ERRATE_VP):
    file_path = os.path.join(ERRATE_VP, the_file)
    if os.path.isfile(file_path):
        os.unlink(file_path)

for the_file in os.listdir(CORRETTE_PP):
    file_path = os.path.join(CORRETTE_PP, the_file)
    if os.path.isfile(file_path):
        os.unlink(file_path)


# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[ ]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[ ]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[ ]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[ ]:


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg 
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'divise_3chann'
PATH_TO_TEST_IMAGES_DIR = 'ripulito/Ripulito(conH50)'
# PATH_TO_TEST_IMAGES_DIR = 'ripulito/test'
# PATH_TO_TEST_IMAGES_DIR = 'aaa'
TEST_IMAGE_PATHS_ONE = os.path.join(PATH_TO_TEST_IMAGES_DIR, 'empty/')

i = 1 

fig = plt.figure()

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    for image in tqdm(os.listdir(TEST_IMAGE_PATHS_ONE)):
      if not image.startswith('.'):
        image_path = os.path.join(TEST_IMAGE_PATHS_ONE,image)
        image_name = image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        p = image.shape[0]
        p2 = image.shape[1]
        image2 = np.zeros([p,p2,3])
        image2[:,:,0] = np.array(image)
        image2[:,:,1] = np.array(image)
        image2[:,:,2] = np.array(image)
        image_np = np.array(image2)

        
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        cell_detected = vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            min_score_thresh=SCORE_THRESH,
            use_normalized_coordinates=True,
            line_thickness=4)
        
        im = cv2.resize(image_np, (300,300))
        # print(cell_detected)

        if cell_detected:
            w1 = w1+1
            #copia immagini con bb
            pwd = os.path.join(ERRATE_VP,image_name)
            cv2.imwrite(pwd, im)
            # copia immagini senza bb
            # pwd = os.path.join(ERRATE_VP,image_name)
            # copyfile(image_path, os.path.join(ERRATE_VP, image_name))
        else:
            # pwd = os.path.join(ERRATE_VP,image_name)
            # cv2.imwrite(pwd, im)
            c2 = c2+1


        et = et+1
        i = i +1

TEST_IMAGE_PATHS_ONE = os.path.join(PATH_TO_TEST_IMAGES_DIR, 'one/')

i = 1 

fig = plt.figure()

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    for image in tqdm(os.listdir(TEST_IMAGE_PATHS_ONE)):
      if not image.startswith('.'):
        image_path = os.path.join(TEST_IMAGE_PATHS_ONE,image)
        image_name = image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        p = image.shape[0]
        p2 = image.shape[1]
        image2 = np.zeros([p,p2,3])
        image2[:,:,0] = np.array(image)
        image2[:,:,1] = np.array(image)
        image2[:,:,2] = np.array(image)
        image_np = np.array(image2)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        cell_detected = vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            max_boxes_to_draw=3,
            min_score_thresh=SCORE_THRESH,
            use_normalized_coordinates=True,
            line_thickness=4,
            area_thresh=AREA_THRESH)
        

        im = cv2.resize(image_np, (300,300))

        cell = np.amax(scores)

        if cell_detected:
            c1 = c1+1
            pwd = os.path.join(CORRETTE_PP,image_name)
            cv2.imwrite(pwd, im)
            # y = fig.add_subplot(4,4,i)
            # y.imshow(im,cmap='gray')
            # plt.title('C')
        else:
            # y = fig.add_subplot(4,4,i)
            # y.imshow(im,cmap='gray')
            # plt.title('W')
            # pwd = os.path.join(ERRATE_PV,image_name)
            # cv2.imwrite(pwd, im)
            w2 = w2+1
            # copyfile(image_path, os.path.join(ERRATE_PV, image_name))
                
        ft = ft+1
        i += 1

# plt.show()

perc_full = c1 * 100 / ft
perc_empty = c2 * 100 / et
perc_tot = (c1+c2)*100/(ft+et)
perc_full = float("{0:.2f}".format(perc_full))
perc_empty = float("{0:.2f}".format(perc_empty))
perc_tot = float("{0:.2f}".format(perc_tot))
with open(FILE_NAME, 'a') as f:
    f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format('test_set','ssd',c1,w2,ft,w1,c2,et,perc_full,perc_empty,perc_tot))


print('imag tot: ',i-1)
print((c1+c2),'/',(ft+et))
print('percent: ',perc_tot)
# In[ ]:





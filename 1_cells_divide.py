'''
Inserire le immagini da analizzare in '/immagini_da_dividere/input/432346.jpg'
Lo script cancellera' i risultati delle analisi precedenti
'''

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
import shutil
from shutil import copyfile
from utils import label_map_util
from utils import visualization_utils as vis_util

sys.path.append("..")

MODEL_NAME = 'cell_graph_inception'
MODEL_NAME = 'cell_graph_1'
MODEL_NAME = 'cell_graph_500_inception'
MODEL_NAME = 'cell_graph_500_mobilenet'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 1
AREA_THRESH = 0.3
SCORE_THRESH = 0.5

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

DIR = 'immagini_da_dividere'
DA_DIVIDERE = os.path.join(DIR,'input')
OUTPUT = os.path.join(DIR,'output')
SAVE_DIRECTORY_ONE =  os.path.join(OUTPUT, 'full')
SAVE_DIRECTORY_CELLS = os.path.join(OUTPUT, 'cells')
SAVE_DIRECTORY_EMPTY = os.path.join(OUTPUT, 'empty')

DIVISE = os.path.join(DIR,'output')
if os.path.exists(DIVISE):
    shutil.rmtree(DIVISE)

os.makedirs(DIVISE)
os.makedirs(SAVE_DIRECTORY_EMPTY)
os.makedirs(SAVE_DIRECTORY_CELLS)
os.makedirs(SAVE_DIRECTORY_ONE)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    for image in tqdm(os.listdir(DA_DIVIDERE)):
      if not image.startswith('.'):
        image_path = os.path.join(DA_DIVIDERE,image)
        image_name = image
        image = Image.open(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image2 = np.zeros([image.shape[0],image.shape[1],3])
        for i in range(3):
            image2[:,:,i] = np.array(image)
        image_np = np.array(image2)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        cell_detected = vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            bb_color='blue',
            fill_color='white',
            min_score_thresh=SCORE_THRESH,
            use_normalized_coordinates=True,
            line_thickness=4)
        
        im = cv2.resize(image_np, (300,300))
        if cell_detected:
            pwd = os.path.join(SAVE_DIRECTORY_CELLS,image_name)
            cv2.imwrite(pwd, im)
            copyfile(image_path, os.path.join(SAVE_DIRECTORY_ONE, image_name))
        else:
            copyfile(image_path, os.path.join(SAVE_DIRECTORY_EMPTY, image_name))

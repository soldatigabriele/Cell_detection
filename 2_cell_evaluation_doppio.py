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
import shutil
import cv2
from tqdm import tqdm
from shutil import copyfile
from utils import label_map_util
from utils import visualization_utils as vis_util

sys.path.append("..")


MODEL_NAME = 'cell_graph_500_mobilenet'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 1
AREA_THRESH = 0.3
SCORE_THRESH = 0.5

FILE_NAME='summaries/summary_both_doppio_{}soglia{}.csv'.format(AREA_THRESH,SCORE_THRESH)

with open(FILE_NAME, 'w') as f:
    f.write('dataset,model,pp,pv,pt,vp,vv,vt,perc_full,perc_empty,perc_tot\n')

pp = 0
vv = 0
pv = 0
vp = 0
pt = 0
vt = 0

PATH = 'ripulito/doppio'
VP = os.path.join(PATH,'vp')
PV = os.path.join(PATH,'pv')
PP = os.path.join(PATH,'pp')
VV = os.path.join(PATH,'vv')
EMPTY = os.path.join(PATH,'empty')
ONE = os.path.join(PATH,'one')

if os.path.exists(PATH):
    shutil.rmtree(PATH)
os.makedirs(ONE)
os.makedirs(EMPTY)
os.makedirs(VV)
os.makedirs(VP)
os.makedirs(PP)
os.makedirs(PV)
os.makedirs('ripulito/doppio/vp_nobb')

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

DA_DIVIDERE = 'ripulito/Ripulito(senzaH50)'
DIR_EMPTY = os.path.join(DA_DIVIDERE, 'empty/')


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for image in tqdm(os.listdir(DIR_EMPTY)):
            if not image.startswith('.'):
                image_path = os.path.join(DIR_EMPTY,image)
                image_name = image
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                p = image.shape[0]
                p2 = image.shape[1]
                image2 = np.zeros([p,p2,3])
                for j in range(3):
                    image2[:,:,j] = np.array(image)
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
                        min_score_thresh=SCORE_THRESH,
                        use_normalized_coordinates=True,
                        line_thickness=1)

                im = cv2.resize(image_np, (300,300))
                if cell_detected:
                    vp = vp+1
                    pwd = os.path.join(VP,image_name)
                    cv2.imwrite(pwd, im)
                    copyfile(image_path, os.path.join('ripulito/doppio/vp_nobb', image_name))
                else:
                    copyfile(image_path, os.path.join(EMPTY, image_name))

                vt = vt+1

DIR_FULL = os.path.join(DA_DIVIDERE, 'one/')
i = 1 

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for image in tqdm(os.listdir(DIR_FULL)):
            if not image.startswith('.'):
                image_path = os.path.join(DIR_FULL,image)
                image_name = image
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                p = image.shape[0]
                p2 = image.shape[1]
                image2 = np.zeros([p,p2,3])
                for j in range(3):
                    image2[:,:,j] = np.array(image)
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
                        max_boxes_to_draw=3,
                        min_score_thresh=SCORE_THRESH,
                        use_normalized_coordinates=True,
                        line_thickness=1,
                        area_thresh=AREA_THRESH)

                im = cv2.resize(image_np, (300,300))
                if cell_detected:
                    pwd = os.path.join(PP,image_name)
                    cv2.imwrite(pwd, im)
                    pp = pp+1
                else:
                    copyfile(image_path, os.path.join(ONE, image_name))
                pt = pt+1

MODEL_NAME = 'cell_graph_500_inception'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 1
AREA_THRESH = 0.3
SCORE_THRESH = 0.1

#reset the graph
tf.reset_default_graph()
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

i = 1 

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for image in tqdm(os.listdir(EMPTY)):
            if not image.startswith('.'):
                image_path = os.path.join(EMPTY,image)
                image_name = image
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                p = image.shape[0]
                p2 = image.shape[1]
                image2 = np.zeros([p,p2,3])
                for j in range(3):
                    image2[:,:,j] = np.array(image)
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
                    min_score_thresh=SCORE_THRESH,
                    use_normalized_coordinates=True,
                    line_thickness=1)
                
                im = cv2.resize(image_np, (300,300))

                if cell_detected:
                    vp = vp+1
                    pwd = os.path.join(VP,image_name)
                    cv2.imwrite(pwd, im)
                    copyfile(image_path, os.path.join('ripulito/doppio/vp_nobb', image_name))
                else:
                    copyfile(image_path, os.path.join(VV, image_name))
                    vv = vv+1

                i = i +1


fig = plt.figure()

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for image in tqdm(os.listdir(ONE)):
            if not image.startswith('.'):
                image_path = os.path.join(ONE,image)
                image_name = image
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                p = image.shape[0]
                p2 = image.shape[1]
                image2 = np.zeros([p,p2,3])
                for j in range(3):
                    image2[:,:,j] = np.array(image)
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
                    max_boxes_to_draw=3,
                    min_score_thresh=SCORE_THRESH,
                    use_normalized_coordinates=True,
                    line_thickness=1,
                    area_thresh=AREA_THRESH)
                

                im = cv2.resize(image_np, (300,300))

                if cell_detected:
                    pp = pp+1
                    pwd = os.path.join(PP,image_name)
                    cv2.imwrite(pwd, im)
                else:
                    copyfile(image_path, os.path.join(PV, image_name))
                    pv = pv+1

perc_full = pp * 100 / pt
perc_empty = vv * 100 / vt
perc_tot = (pp+vv)*100/(pt+vt)
perc_full = float("{0:.2f}".format(perc_full))
perc_empty = float("{0:.2f}".format(perc_empty))
perc_tot = float("{0:.2f}".format(perc_tot))
with open(FILE_NAME, 'a') as f:
    f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format('test_set','ssd',pp,pv,pt,vp,vv,vt,perc_full,perc_empty,perc_tot))

print('imag tot: ',i-1)
print((pp+vv),'/',(pt+vt))
print('percent: ',perc_tot)


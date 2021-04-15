import os
import pathlib
import pyautogui
import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

if "models" in pathlib.Path.cwd().parts:
  while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')
elif not pathlib.Path('models').exists():
  get_ipython().system('git clone --depth 1 https://github.com/tensorflow/models')

utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile

def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"
  model = tf.saved_model.load(str(model_dir))
  return model

PATH_TO_LABELS = 'C:/Object_detection/models-master/research/object_detection/images/labelmap.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

PATH_TO_TEST_IMAGES_DIR = pathlib.Path('C:/Object_detection/models-master/research/object_detection/test_images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
TEST_IMAGE_PATHS

detection_model = tf.saved_model.load('C:/Object_detection/models-master/research/object_detection/inference_graph/saved_model')
detection_model.signatures['serving_default'].output_dtypes
detection_model.signatures['serving_default'].output_shapes

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  
  input_tensor = tf.convert_to_tensor(image)
  input_tensor = input_tensor[tf.newaxis,...]

  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

  if 'detection_masks' in output_dict:
   
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])   
    print(image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.8,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict


def show_inference(model, image_np):

  output_dict = run_inference_for_single_image(model, image_np)

  boxes = np.squeeze(output_dict['detection_boxes'])
  scores = np.squeeze(output_dict['detection_scores'])
  classes = np.squeeze(output_dict['detection_classes'])
  #set a min thresh score, say 0.8
  min_score_thresh = 0.8
  finalBoxes = boxes[scores > min_score_thresh]
  finalScore = scores[scores > min_score_thresh]
  finalClasses = classes[scores > min_score_thresh]
  print(finalBoxes, finalScore, finalClasses)


  if len(finalBoxes)>0:
    x = (finalBoxes[0][0] + finalBoxes[0][2])/2
    y = (finalBoxes[0][1] + finalBoxes[0][3])/2

    pyautogui.moveTo(1920*y,1080*x)
    #box = output_dict['detection_scores']
   
    if finalClasses[0] == 6:
      pyautogui.click(1910*y,1070*x)

    elif finalClasses[0] == 7:
      pyautogui.doubleClick(1910*y,1070*x)

    elif finalClasses[0] == 8:
      pyautogui.scroll(amount_to_scroll, x=moveToX, y=moveToY)
      pyautogui.scroll(-500)


  final_img =vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks_reframed', None),
          use_normalized_coordinates=True,
          line_thickness=2)
  return(final_img)

cap = cv2.VideoCapture(0)
pyautogui.FAILSAFE = False
while 1:
    _,img = cap.read()
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    final_img = show_inference(detection_model,img)
    
    final_img = cv2.cvtColor(final_img,cv2.COLOR_RGB2BGR)

    cv2.imshow('img',final_img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
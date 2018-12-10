# -*- coding: utf-8 -*-

"""
    author = wuxiaochun
"""

import numpy as np
import pydensecrf.densecrf as dcrf
import time
import os
from shutil import copyfile
import argparse
import PIL
import warnings
warnings.filterwarnings("ignore")

# Get im{read,write} from somewhere.
try:
    from cv2 import imread, imwrite
except ImportError:
    # Note that, sadly, skimage unconditionally import scipy and matplotlib,
    # so you'll need them if you don't have OpenCV. But you probably have them.
    from skimage.io import imread, imsave
    imwrite = imsave
    # TODO: Use scipy instead.

from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

parser = argparse.ArgumentParser()

parser.add_argument('--image_data_dir', type=str, default='../data/2012_image/',
                    help='The directory containing the image data.')

parser.add_argument('--label_data_dir', type=str, default='../data/2012_stacking/',
                    help='The directory containing the prediction label data.')

parser.add_argument('--output_dir', type=str, default='../output/',
                    help='Path to the directory to generate the dencecrf inference result.')

parser.add_argument('--test_data_list', type=str, default='../data_list/2012_test.txt',
                    help='Path to the file listing the all images.')


def densecrf(name):
   fn_im = FLAGS.image_data_dir+name+'.jpg'
   fn_anno = FLAGS.label_data_dir+name+'.png'
   fn_output = FLAGS.output_dir+name+'.png'
   
   ##################################
   ### Read images and annotation ###
   ##################################
   img = imread(fn_im)
   
   # Convert the annotation's RGB color to a single 32-bit integer color 0xBBGGRR
   anno_rgb = imread(fn_anno).astype(np.uint32)
   MARK = [x for x in set(anno_rgb.flat)]
#   print("MARK is",MARK)
   anno_rgb[anno_rgb == 1] = 255
   anno_rgb[anno_rgb < 1] = 1
   anno_rgb[anno_rgb > 1] = 255
   #anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)
   anno_lbl = anno_rgb
   
   # Convert the 32bit integer color to 1, 2, ... labels.
   # Note that all-black, i.e. the value 0 for background will stay 0.
   colors, labels = np.unique(anno_lbl, return_inverse=True)
   
   # But remove the all-0 black, that won't exist in the MAP!
   HAS_UNK = 0 in colors
   if HAS_UNK:
       print("Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
       print("If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
       colors = colors[1:]
   #else: 
   #    print("No single full-black pixel found in annotation image. Assuming there's no 'unknown' label!")
   
   # And create a mapping back from the labels to 32bit integer colors.
   colorize = np.empty((len(colors), 3), np.uint8)
   colorize[:,0]=colors
   colorize[:,0] = (colors & 0x0000FF)
   colorize[:,1] = (colors & 0x00FF00) >> 8
   colorize[:,2] = (colors & 0xFF0000) >> 16
   
   # Compute the number of classes in the label image.
   # We subtract one because the number shouldn't include the value 0 which stands
   # for "unknown" or "unsure".
   n_labels = len(set(labels.flat)) - int(HAS_UNK)
   print(n_labels, " labels", (" plus \"unknown\" 0: " if HAS_UNK else ""), set(labels.flat))
   if n_labels == 1:
       print("ignoring this image.")
       return 0
   ###########################
   ### Setup the CRF model ###
   ###########################
   use_2d = False
   #use_2d = True
   if use_2d:
       print("Using 2D specialized functions")
   
       # Example using the DenseCRF2D code
       d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)
   
       # get unary potentials (neg log probability)
       U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
       d.setUnaryEnergy(U)
   
       # This adds the color-independent term, features are the locations only.
       d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                             normalization=dcrf.NORMALIZE_SYMMETRIC)
   
       # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
       d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                              compat=10,
                              kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)
   else:
       print("Using generic 2D functions")
   
       # Example using the DenseCRF class and the util functions
       d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)
   
       # get unary potentials (neg log probability)
       U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
       d.setUnaryEnergy(U)
   
       # This creates the color-independent features and then add them to the CRF
       feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
       d.addPairwiseEnergy(feats, compat=3,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
   
       # This creates the color-dependent features and then add them to the CRF
       feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                         img=img, chdim=2)
       d.addPairwiseEnergy(feats, compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
   
   
   ####################################
   ### Do inference and compute MAP ###
   ####################################
   num_iter = 1
   # Run five inference steps.
   Q = d.inference(num_iter)
   
   # Find out the most probable class for each pixel.
   MAP = np.argmax(Q, axis=0)
#   res = MAP.reshape(anno_lbl.shape)
   print("MAP's shape is:{}.".format(MAP.shape))
   # Convert the MAP (labels) back to the corresponding colors and save the image.
   # Note that there is no "unknown" here anymore, no matter what we had at first.
#   MAP = colorize[MAP,:]
   res = MAP.reshape(anno_lbl.shape)
   for i in range(res.shape[0]):
      for j in range(res.shape[1]):
         if res[i][j] != 0:
            res[i][j] = MARK[1]
   imwrite(fn_output, res)
   
   # Just randomly manually run inference iterations
   Q, tmp1, tmp2 = d.startInference()
#   for i in range(5):
#       print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
   d.stepInference(Q, tmp1, tmp2)
#   print(np.shape(Q),np.shape(MAP),np.shape(tmp2))

def filter():
    print("Begin to filter.")
    names = os.listdir(FLAGS.label_data_dir)
    count = 0
    max_items = 0
    count_dict = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}
    long_names=[]
    # travelse all the images to filter images containing 2 or more objects
    for name in names:
       test = PIL.Image.open(FLAGS.label_data_dir+name)
       test_arr = np.array(test)
       test_lst = []
       for i in range(len(test_arr)):
          test_lst += list(test_arr[i])
       test_list = list(set(test_lst))
       count_dict[len(test_list)] += 1
       if len(test_list)>2:
          count += 1
          long_names.append(name.split('.')[0])
       if len(test_list)>max_items:
          max_items = len(test_list)
    print("共有{}张图片包含2类及以上种分类物体".format(count))
    print("一张图片中最多含有{}种物体".format(max_items))
    return long_names

FLAGS = parser.parse_args()
start_time = time.time()
long_names = filter()
with open(FLAGS.test_data_list,'r') as file:
    all_names = file.readlines()
short_names = [x for x in all_names if x.split('\n')[0] not in long_names]
i = 0
for name in short_names:
    print("Begin to process image {}".format(i))
    densecrf(name.split('\n')[0])
    i += 1

print("There are total {} images been processed.".format(i))
ignore_names = [x for x in os.listdir(FLAGS.label_data_dir) if x not in os.listdir(FLAGS.output_dir)]
for name in ignore_names:
   copyfile(FLAGS.label_data_dir+name,FLAGS.output_dir+name)
end_time = time.time()
print('The total time is{}.'.format(end_time-start_time))
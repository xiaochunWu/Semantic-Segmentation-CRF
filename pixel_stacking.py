# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:02:44 2018

@author: wuxiaochuna
"""

import os
import numpy as np
import time
import PIL
import argparse

parser = argparse.ArgumentParser()

parser.add_agrument('--path_val', type=str, default='..\2011_trainaug\raw_segmentation_results',
                    help='The directory containing the first model inference.')
                    
parser.add_argument('--path_aug', type=str, default='..\2011_trainaug\raw_segmentation_results',
                    help='The directory containing the second model inference.')
                    
parser.add_agrument('--path_ori', type=str, default='..\..\voc-test\results\VOC2011\Segmentation\comp6_test_cls',
                    help='The directory containing the third model inference.')

parser.add_agrument('--path_stacking', type=str, default='..\2011_stacking',
                    help='Path to the directory to generate the pixel_stacking result.')
FLAGS = parser.parse_args()
names = os.listdir(FLAGS.FLAGS.path_val)

# 对result进行名称转换============================================================
# count = 0
# start_time = time.time()
# for name_old in names:
#    name = name_old.split('\'')
#    name_new = name[1]
#    os.rename(path+"\\"+name_old,path+"\\"+name_new+'.png')
#    count += 1
#    
# end_time = time.time()
# print("{} images have been renamed, the total time is {}s.".format(count,(end_time-start_time)))
#==============================================================================
def pixelSelect(pixel_0,pixel_1,pixel_2):
   pixels = [pixel_0,pixel_1,pixel_2]
   counts = np.bincount(pixels)
   return np.argmax(counts)

start_time = time.time()
count = 0
for name in names:
   img_trainval = PIL.Image.open(FLAGS.path_val+'\\'+name)
   img_trainaug = PIL.Image.open(FLAGS.path_aug+'\\'+name)
   img_original = PIL.Image.open(FLAGS.path_ori+'\\'+name)
   img_val = np.array(img_trainval)
   img_aug = np.array(img_trainaug)
   img_ori = np.array(img_original)
   height = img_val.shape[1]
   width = img_val.shape[0]
   img_stacking = np.zeros((width,height))
   for i in range(width):
      for j in range(height):
         img_stacking[i][j] = pixelSelect(img_val[i][j],img_aug[i][j],img_ori[i][j])
   img = PIL.Image.fromarray(img_stacking).convert('P')
   img.save(FLAGS.path_stacking+'\\'+name)
   count += 1

end_time = time.time()
print('stacking done!\n{} images done, {}s cost.'.format(count,(end_time-start_time)))

#image = PIL.Image.open(path+'\\2008_000006.png')
#print(pixelSelect(0,2,1))

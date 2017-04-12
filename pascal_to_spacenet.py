import os
import re
import json
from os import listdir
import argparse
import time
import cv2
from osgeo import gdal,gdalnumeric
import ogr
import numpy as np

# User-defined module
import _init_paths
import caffe
from mnc_config import cfg
from transform.bbox_transform import clip_boxes
from utils.blob import prep_im_for_blob, im_list_to_blob
from transform.mask_transform import gpu_mask_voting
import matplotlib.pyplot as plt
from utils.vis_seg import _convert_pred_to_image, _get_voc_color_map
from PIL import Image

from functions_for_pascal_to_spacenet import prepare_mnc_args, im_detect, get_vis_dict, process_jpg

parser = argparse.ArgumentParser()
parser.add_argument("jpg_dir", help="directory for raster jpegs")
parser.add_argument("tif_dir", help="directory for raster tifs")
parser.add_argument("geojson_dir", help="directory for geojson outputs")
parser.add_argument("updated_geojson_dir", help="directory for updated geojson outputs")

args = parser.parse_args()
my_jpg_dir = args.jpg_dir
my_tif_dir = args.tif_dir
my_geojson_dir = args.geojson_dir
my_updated_geojson_dir = args.updated_geojson_dir


#Setting some parameters
CLASSES = ['building']
test_prototxt = 'MNC/models/VGG16/mnc_5stage/test.prototxt'
test_model = 'MNC/output/mnc_5stage/voc_2012_train/vgg16_mnc_5stage_iter_25000.caffemodel.h5'
caffe.set_mode_gpu()
caffe.set_device(0)
cfg.GPU_ID = 0
net = caffe.Net(test_prototxt, test_model, caffe.TEST)



# Warm up for the first two images
im = 128 * np.ones((300, 500, 3), dtype=np.float32)
for i in xrange(2):
    _, _, _ = im_detect(im, net)


#Loop through jpg directory
for im_name in listdir(my_jpg_dir):

	#Get image number
	image_number_search = re.search('(?<=img)\w+', im_name)
	image_number = image_number_search.group(0)

	
	#Fill in ... with the specifics of your filenames
	my_tif_name = 'RGB-PanSharpen_..._img'+str(image_number)+'.tif'
	my_geojson_name= 'First_AOI_..._img'+str(image_number)+'.geojson'
	my_updated_geojson_name= 'Updated_AOI_..._img'+str(image_number)+'.geojson'

	process_jpg(CLASSES, net, im_name, my_jpg_dir, my_tif_name, my_tif_dir, my_geojson_name, my_geojson_dir, my_updated_geojson_name, my_updated_geojson_dir, 0.3)

print "Done processing!"





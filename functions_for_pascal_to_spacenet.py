import os
import re
import json
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
from area import area

def prepare_mnc_args(im, net):
    # Prepare image data blob
    blobs = {'data': None}
    processed_ims = []
    im, im_scale_factors = prep_im_for_blob(im, cfg.PIXEL_MEANS, cfg.TEST.SCALES[0], cfg.TRAIN.MAX_SIZE)
    processed_ims.append(im)
    blobs['data'] = im_list_to_blob(processed_ims)
    # Prepare image info blob
    im_scales = [np.array(im_scale_factors)]
    assert len(im_scales) == 1, 'Only single-image batch implemented'
    im_blob = blobs['data']
    blobs['im_info'] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        dtype=np.float32)
    # Reshape network inputs and do forward
    net.blobs['data'].reshape(*blobs['data'].shape)
    net.blobs['im_info'].reshape(*blobs['im_info'].shape)
    forward_kwargs = {
        'data': blobs['data'].astype(np.float32, copy=False),
        'im_info': blobs['im_info'].astype(np.float32, copy=False)
    }
    return forward_kwargs, im_scales


def im_detect(im, net):
    forward_kwargs, im_scales = prepare_mnc_args(im, net)
    blobs_out = net.forward(**forward_kwargs)
    # output we need to collect:
    # 1. output from phase1'
    rois_phase1 = net.blobs['rois'].data.copy()
    masks_phase1 = net.blobs['mask_proposal'].data[...]
    scores_phase1 = net.blobs['seg_cls_prob'].data[...]
    # 2. output from phase2
    rois_phase2 = net.blobs['rois_ext'].data[...]
    masks_phase2 = net.blobs['mask_proposal_ext'].data[...]
    scores_phase2 = net.blobs['seg_cls_prob_ext'].data[...]
    # Boxes are in resized space, we un-scale them back
    rois_phase1 = rois_phase1[:, 1:5] / im_scales[0]
    rois_phase2 = rois_phase2[:, 1:5] / im_scales[0]
    rois_phase1, _ = clip_boxes(rois_phase1, im.shape)
    rois_phase2, _ = clip_boxes(rois_phase2, im.shape)
    # concatenate two stages to get final network output
    masks = np.concatenate((masks_phase1, masks_phase2), axis=0)
    boxes = np.concatenate((rois_phase1, rois_phase2), axis=0)
    scores = np.concatenate((scores_phase1, scores_phase2), axis=0)
    return boxes, masks, scores


def get_vis_dict(result_box, result_mask, img_name, cls_names, vis_thresh):
    box_for_img = []
    mask_for_img = []
    cls_for_img = []
    for cls_ind, cls_name in enumerate(cls_names):
        det_for_img = result_box[cls_ind]
        seg_for_img = result_mask[cls_ind]
        keep_inds = np.where(det_for_img[:, -1] >= vis_thresh)[0]
        for keep in keep_inds:
            box_for_img.append(det_for_img[keep])
            mask_for_img.append(seg_for_img[keep][0])
            cls_for_img.append(cls_ind + 1)
    res_dict = {'image_name': img_name,
                'cls_name': cls_for_img,
                'boxes': box_for_img,
                'masks': mask_for_img}
    return res_dict


def process_jpg(CLASSES, net, im_name, jpg_dir, tif_name, tif_dir, geojson_name, geojson_dir, updated_geojson_name, updated_geojson_dir, vis_threshold):
    print "\n"
    print "Processing {}".format(jpg_dir+'/'+im_name)
	#Create initia geojson (has extra DN:0 polygons)
     
    gt_image = os.path.join(jpg_dir,im_name)
    im = cv2.imread(gt_image)
    
    img_width = im.shape[1]
    img_height = im.shape[0]
    
    boxes, masks, seg_scores = im_detect(im, net)

    result_mask, result_box = gpu_mask_voting(masks, boxes, seg_scores, len(CLASSES) + 1,300, im.shape[1], im.shape[0])
    pred_dict = get_vis_dict(result_box, result_mask, jpg_dir + im_name, CLASSES, vis_threshold)
    

    num_inst = len(pred_dict['boxes'])
    
    #Get image number
    image_number_search = re.search('(?<=img)\w+', im_name)
    image_number = image_number_search.group(0)
    
    #Open tif file
    print "Opening {}".format(tif_dir+'/'+tif_name)
    srcRas_ds = gdal.Open(tif_dir+'/'+tif_name)
    
    #Stuff to create before entering instance loop
    geom = srcRas_ds.GetGeoTransform()
    proj = srcRas_ds.GetProjection()
    memdrv = gdal.GetDriverByName('MEM')
    inst_img = np.zeros((img_height, img_width))
    src_ds = memdrv.Create('',inst_img.shape[1],inst_img.shape[0],num_inst)
    src_ds.SetGeoTransform(geom)
    src_ds.SetProjection(proj)
    
    #Create geojson data source
    drv = ogr.GetDriverByName("geojson")
    dst_ds = drv.CreateDataSource(geojson_dir+'/'+geojson_name)

    #Create layer
    dst_layername = "building_layer_name"
    dst_layer = dst_ds.CreateLayer(dst_layername, srs=None )
    dst_layer.CreateField(ogr.FieldDefn("DN",ogr.OFTInteger))
    
    for inst_index in xrange(num_inst):
        box = np.round(pred_dict['boxes'][inst_index]).astype(int)
        mask = pred_dict['masks'][inst_index]
        cls_num = pred_dict['cls_name'][inst_index]

        box[0] = min(max(box[0], 0), img_width - 1)
        box[1] = min(max(box[1], 0), img_height - 1)
        box[2] = min(max(box[2], 0), img_width - 1)
        box[3] = min(max(box[3], 0), img_height - 1)

        mask = cv2.resize(mask.astype(np.float32), (box[2]-box[0]+1, box[3]-box[1]+1))
        mask = mask >= cfg.BINARIZE_THRESH
        mask = mask.astype(int)

        part1 = (1) * mask.astype(np.float32)
        part2 = np.multiply(np.logical_not(mask), inst_img[box[1]:box[3]+1, box[0]:box[2]+1])

        #Reset instance image to 0's
        inst_img = np.zeros((img_height, img_width))
        inst_img[box[1]:box[3]+1, box[0]:box[2]+1] = part1 + part2
        inst_img = inst_img.astype(int)

        band = src_ds.GetRasterBand(1)
        band.WriteArray(inst_img)

        gdal.Polygonize(band, None, dst_layer, 0, [], callback=None) 
    dst_ds=None
    
    #Now reformat the geojson we just created
    print 'Reformat geojson {}'.format(geojson_name)
    
    #set index to 0
    i = 0
    geojson_full_name = geojson_dir+'/'+geojson_name
    with open(geojson_full_name, 'r') as f:
        data = json.load(f)
        while (i < len(data['features'])):
            if(data['features'][i]['properties']['DN'] == 0):
                data['features'].pop(i)
            elif(area(data['features'][i]['geometry']) <= 50):
                data['features'].pop(i)
            else:
                i = i+1

    #Write geojson
    updated_geojson_full_name = updated_geojson_dir + '/' + updated_geojson_name
    with open(updated_geojson_full_name, 'w') as outfile:
        json.dump(data, outfile, sort_keys=True)
        print "Geojson reformatted!"
    



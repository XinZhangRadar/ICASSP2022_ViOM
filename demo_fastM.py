# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from shapely.geometry import Polygon, MultiPoint
import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections,pos_extract
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb
sys.path.append('/home/zhangxin/Matlab/R2016b/extern/engines/python/build/lib/')
from functools import partial
import matplotlib.pyplot as plt
from pycpd import rigid_registration,affine_registration,deformable_registration
import numpy as np
import time
import matlab.engine
import scipy.io
from PIL import Image, ImageDraw
from model.utils.hungarian import Hungarian

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  #daraset
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  #config
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  #net
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  #set
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  #load dir
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',                
                      #default="/srv/share/jyang375/models")
                      default="./data/pretrained_model")
  #image dir
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="images")
  #cuda
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  #multiple gpus
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  #class_agnostic bbox regression
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  #parallel
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  #check session
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  #check epoch
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  #check point
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  #batch size                    
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  #visualization                    
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  #webcam ID
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)

  args = parser.parse_args()
  return args
#learning rate
lr = cfg.TRAIN.LEARNING_RATE
#momentum
momentum = cfg.TRAIN.MOMENTUM
#weight_decay
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  #
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)
def visualize(iteration, error, X, Y, ax):
    '''
    plt.cla()
    ax.scatter(X[:,0] ,  X[:,1], color='red', label='Target')
    ax.scatter(Y[:,0] ,  Y[:,1], color='blue', label='Source')
    plt.text(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)
    '''
    return X,Y
def polygon_iou(list1, list2):
    """
    Intersection over union between two shapely polygons.
    """
    polygon_points1 = np.array(list1).reshape(4, 2)
    poly1 = Polygon(polygon_points1).convex_hull
    polygon_points2 = np.array(list2).reshape(4, 2)
    poly2 = Polygon(polygon_points2).convex_hull
    union_poly = np.concatenate((polygon_points1, polygon_points2))
    if not poly1.intersects(poly2):  # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                return 0
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou
def draw_rectangle(image, bounds, width=1, outline='white', antialias=4):
    """Improved ellipse drawing function, based on PIL.ImageDraw."""

    # Use a single channel image (mode='L') as mask.
    # The size of the mask can be increased relative to the imput image
    # to get smoother looking results. 
    mask = Image.new(
        size=[int(dim * antialias) for dim in image.size],
        mode='L', color='black')

    draw = ImageDraw.Draw(mask)
    #pdb.set_trace();

    # draw outer shape in white (color) and inner shape in black (transparent)
    for offset, fill in (width/-2.0, 'white'), (width/2.0, 'black'):
        left, top = [(value + offset) * antialias for value in bounds[:2]]
        right, bottom = [(value - offset) * antialias for value in bounds[2:]]
        draw.rectangle([left, top, right, bottom], fill=fill)

    # downsample the mask using PIL.Image.LANCZOS 
    # (a high-quality downsampling filter).
    mask = mask.resize(image.size, Image.LANCZOS)
    # paste outline color to input image through the mask
    image.paste(outline, mask=mask)

if __name__ == '__main__':
  
  args = parse_args()
  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  cfg.USE_GPU_NMS = args.cuda

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)
  #os.environ["CUDA_VISIBLE_DEVICES"] = "7"

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  #input_dir = args.load_dir
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    #'faster_rcnn_attention_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    'faster_rcnn_attention_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
  '''
  pascal_classes = np.asarray(['__background__',
                       'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor'])
  '''
  pascal_classes = np.asarray(['__background__','plane'])
  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  if args.cuda > 0:
    checkpoint = torch.load(load_name)
  else:
    checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']
  cfg.POOLING_MODE = 'prp'

  print('load model successfully!')

  # pdb.set_trace()

  #print("load checkpoint %s" % (load_name))

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()
  #pdb.set_trace()
  # make variable
  with torch.no_grad():
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
  
    if args.cuda > 0:
      cfg.CUDA = True
  
    if args.cuda > 0:
      fasterRCNN.cuda()
  
    fasterRCNN.eval()
  
    start = time.time()
    max_per_image = 100
    thresh = 0.5
    vis = True
  
    webcam_num = args.webcam_num
    # Set up webcam or get image directories
    if webcam_num >= 0 :
      cap = cv2.VideoCapture(webcam_num)
      num_images = 0
    else:
      imglist = os.listdir(args.image_dir)
      num_images = len(imglist)
  
    print('Loaded Photo: {} images.'.format(num_images))
    image_size = np.zeros(2);
    im_save = [];
    pos_allimg = [];
    area_allimg =[];
    bbox_pos = [];
    #pdb.set_trace();
    while (num_images > 0):
        total_tic = time.time()
        if webcam_num == -1:
          num_images -= 1
  
        # Get image from the webcam
        if webcam_num >= 0:
          if not cap.isOpened():
            raise RuntimeError("Webcam could not open. Please check connection.")
          ret, frame = cap.read()
          im_in = np.array(frame)
        # Load the demo image
        else:
          im_file = os.path.join(args.image_dir, imglist[num_images])
          #im = cv2.imread(im_file)
          im_in = np.array(imread(im_file))
        #pdb.set_trace()
        if len(im_in.shape) == 2:
          im_in = im_in[:,:,np.newaxis]
          im_in = np.concatenate((im_in,im_in,im_in), axis=2)
        # rgb -> bgr
        #pdb.set_trace()
        im = im_in[:,:,::-1]
        #image list,image pyrimd scale
        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        #image list
        im_blob = blobs
        #image shape
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
        #chang image type:numpy->tensor
        im_data_pt = torch.from_numpy(im_blob)
        #chang image dim order
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        #chang image information type:numpy->tensor
        im_info_pt = torch.from_numpy(im_info_np)
        #assign data to placehoder
        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.data.resize_(1, 1, 5).zero_()
        num_boxes.data.resize_(1).zero_()
  
        # pdb.set_trace()
        det_tic = time.time()
        # faster-RCnn, produce bbox,bbox class score,bbox regression result,
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label,name = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)


        #class scores and boxes
        #pdb.set_trace()
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
  
        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
              if args.class_agnostic:
                  if args.cuda > 0:
                      box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                  else:
                      box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
  
                  box_deltas = box_deltas.view(1, -1, 4)
              else:
                  if args.cuda > 0:
                      box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                  else:
                      box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                  box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))
  
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))
  
        pred_boxes /= im_scales[0]
  
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        #print(pred_boxes) 
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        #pdb.set_trace()
        #imshow result
        if vis:
            im2show = np.copy(im)
        for j in xrange(1, len(pascal_classes)):
            inds = torch.nonzero(scores[:,j]>thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
              cls_scores = scores[:,j][inds]
              _, order = torch.sort(cls_scores, 0, True)
              if args.class_agnostic:
                cls_boxes = pred_boxes[inds, :]
              else:
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
              #combine class score and boxes
              cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
              # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
              #order the boxes by class scores
              cls_dets = cls_dets[order]
              #pdb.set_trace();
              keep = nms(cls_dets, cfg.TEST.NMS, force_cpu= not cfg.USE_GPU_NMS)
              #print(len(keep))
              cls_dets = cls_dets[keep.view(-1).long()]

              pos,area = pos_extract(cls_dets.cpu().numpy(), 0)

              
              #im2show,pos,area = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0)
                #pdb.set_trace();
              bbox_pos.append(cls_dets.cpu().numpy());
              pos_allimg.append(pos);
              area_allimg.append(area);
                #pdb.set_trace();
              im_save.append(im2show);
              image_size[0] = int(np.max([im2show.shape[0],image_size[0]]));
              image_size[1] = int(np.max([im2show.shape[1],image_size[1]]));
              #cv2.imshow("test",im2show)
              #cv2.waitKey(0)
                #pdb.set_trace()
        
        
        #print(pos_allimg)
        misc_toc = time.time()
        nms_time = misc_toc - misc_tic
        #position = cls_dets
        #pdb.set_trace()
        if webcam_num == -1:
            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                             .format(num_images + 1, len(imglist), detect_time, nms_time))
            sys.stdout.flush()
  
        if vis and webcam_num == -1:
            # cv2.imshow('test', im2show)
            # cv2.waitKey(0)
            result_path = os.path.join(args.image_dir, imglist[num_images][:-4] + "_det.jpg")
            cv2.imwrite(result_path, im2show)
        else:
            im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
            cv2.imshow("frame", im2showRGB)
            total_toc = time.time()
            total_time = total_toc - total_tic
            frame_rate = 1 / total_time
            print('Frame rate:', frame_rate)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    


    
  
    #callback = partial(visualize, ax=fig.axes[0])
    #pdb.set_trace()
    eng = matlab.engine.start_matlab()
    eng.eval("addpath('/home/zhangxin/FastMatch_v3/')");
    eng.eval("addpath('/home/zhangxin/FastMatch_v3/matlab/')");
    eng.eval("addpath('/home/zhangxin/FastMatch_v3/mex/')");    


    #pdb.set_trace();
    image_size = image_size.astype(np.int32);
    target = Image.new('RGB', (image_size[1]*2, image_size[0]));
    left = 0;
    #pdb.set_trace();
    ind = 0; 
    for im2pil in im_save:
        target.paste(Image.fromarray(cv2.cvtColor(im2pil,cv2.COLOR_BGR2RGB)),(left,0));
        left = left + im2pil.shape[1];
        if ind == 0:
          scipy.io.savemat('img.mat', dict(img=im2pil))
          ind +=1;
        else:
          scipy.io.savemat('template_img.mat', dict(template_img=im2pil))
    
    
    pp = []; 
    t1 = 0;
    label1 = np.zeros(len(bbox_pos[0]));
    label2 = np.zeros(len(bbox_pos[1]));
    print(bbox_pos[0]);
    print(bbox_pos[1]);


    for roi2 in bbox_pos[1] :
      scipy.io.savemat('templatep.mat', dict(templatep=roi2))
      print("template roi is :")
      print(roi2)
      t2 = 0;
      position = eng.FastMatch_demo();#pdb.set_trace()
      position = np.array(position,dtype=np.int32).T;
      print(" computed position is:")
      print(position);
      print

      
      for roi1 in bbox_pos[0]:
        roi = roi1[:-1];
        position2 = np.vstack((np.append(roi[0],roi[3]),np.append(roi[2],roi[1]),np.append(roi[0],roi[1]),np.append(roi[0],roi[1])));
        iou = polygon_iou(position,position2);
        print(iou);

        if iou>=0.2:
          label1[t1] = 1;
          label2[t2] = 1;
          t2 = t2+1;
          break;
        else :
          t1 = t1 +1;
          continue;
     
    pdb.set_trace();
    
    list1 = np.where(label1 == 0);
    list2 = np.where(label2 == 0);
    for r in list1[0]:
      draw_rectangle(target,(bbox_pos[0][r][0],bbox_pos[0][r][1],bbox_pos[0][r][2],bbox_pos[0][r][3]), width=3,outline='red')
        #draw.line((X[match[0][r]-1][0],X[match[0][r]-1][1],Y[r][0]+im_save[0].shape[1],Y[r][1]),fill = 128);
    for r in list2[0]:
      draw_rectangle(target,(bbox_pos[1][r][0]+im_save[0].shape[1],bbox_pos[1][r][1],bbox_pos[1][r][2]+im_save[0].shape[1],bbox_pos[1][r][3]), width=3,outline='red')
    im_name = imglist[num_images]+'RPM-fastMatching.jpg'
    target.save(im_name)
    

 


    #target.show();

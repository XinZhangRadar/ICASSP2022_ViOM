import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
import sys
sys.path.append('/home/zhangxin/faster-rcnn.pytorch/PreciseRoIPooling/pytorch/prroi_pool')

from prroi_pool import PrRoIPool2D
from model.rpn.bbox_transform import bbox_overlaps_batch
import matplotlib.pyplot as plt


class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        #self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()
        #self.attn2 = Self_Attn( 512,  'relu')

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        #pdb.set_trace();
        batch_size = im_data.size(0)
        avg_pool = PrRoIPool2D(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        #pdb.set_trace()

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)
        #pdb.set_trace()

        # feed base feature map tp RPN to obtain rois
        rois,rois_bef_nms, rpn_loss_cls, rpn_loss_bbox,score = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        #pdb.set_trace()
        

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            #pdb.set_trace();
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes,score)

            #roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws,overlaps_indece_batch = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
            #gt = torch.cat( (gt_boxes.view(-1,5)[:,-1].reshape(-1,1),gt_boxes.view(-1,5)[:,:-1]) ,dim = 1)

        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        rois_bef_nms = Variable(rois_bef_nms)
        


        
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())

            #if self.training:
                #grid_gt_xy = _affine_grid_gen(gt, base_feat.size()[2:], self.grid_size)
                #grid_gt_yx = torch.stack([grid_gt_xy.data[:,:,:,1], grid_gt_xy.data[:,:,:,0]], 3).contiguous()
                #pooled_gt_feat = self.RCNN_roi_crop(base_feat, Variable(grid_gt_yx).detach())

            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
                #pooled_gt_feat = F.max_pool2d(pooled_gt_feat, 2, 2)
            name = 'crop'
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            #if self.training:
                #pooled_gt_feat =self.RCNN_roi_align(base_feat, gt)
            name = 'align'
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))
            #if self.training:
                #pooled_gt_feat =self.RCNN_roi_pool(base_feat, gt)
            name = 'pool'
        elif cfg.POOLING_MODE == 'prp':
            pooled_feat = avg_pool(base_feat, rois.view(-1,5))
            #if self.training:
                #pooled_gt_feat = avg_pool(base_feat, gt)

            name = 'prp'
        #pdb.set_trace()
        '''
        for i in range(batch_size):
            indice_roi = np.where(rois_label.view(batch_size,rois_label.size(0))[i,:])[0];#select positive sample index
            pooled_feat_at = pooled_feat.index_select(0, torch.LongTensor(indice_roi).cuda());#obtain positive sample rois pooled feature map ,shape: [pos_num x c x w x h]
            indice_gt = overlaps_indece_batch[i,:][indice_roi];#select positive sample correspoding gt_box index 
            pooled_gt_feat_at = pooled_gt_feat.index_select(0, indice_gt.cuda());#obtain positive sample rois pooled feature map
            pooled_feat_at,p1 = self.attn1(pooled_gt_feat_at,pooled_feat_at)
            pooled_feat = torch.cat( (pooled_feat_at,pooled_feat[pooled_feat_at.size(0):]) ,dim = 0)

        	#pooled_feat[0:pooled_feat_at.size(0)].copy_(pooled_feat_at);
        '''
        #for i in range(batch_size):
            #pooled_feat_at,p1 = self.attn1(pooled_feat[0].expand(pooled_feat.shape),pooled_feat)



        





        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)
        #pdb.set_trace()
#///////////////////////////////////////////////
        #roi_data = self.RCNN_proposal_target_OM(rois, gt_boxes, num_boxes,cls_prob)
#############################
        #pdb.set_trace()
        #overlaps = bbox_overlaps_batch(rois, gt_boxes)
        #max_overlaps, gt_assignment = torch.max(overlaps, 2)

        #X = max_overlaps.cpu().numpy();
        #Y = cls_prob[:,1].cpu().detach().numpy();
        #plt.scatter(X, Y)
        #plt.show()
##############################
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
            #print('RCNN_loss_cls:'+str(RCNN_loss_cls));
            #print('RCNN_loss_bbox:'+str(RCNN_loss_bbox));


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        #return rois,rois_bef_nms, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label,name
        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label,name

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

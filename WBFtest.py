import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader, create_dataloader_rgb_ir
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized

from utils.callbacks import Callbacks
import cv2
from models.WBF.examples.example import example_wbf_3_models, example_wbf_2_models, example_wbf_1_model


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct



def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001, # for NMS
         iou_thres=0.45,  # for NMS  #0.6 --0.45
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=True,  # save auto-label confidences
         plots=False,
         callbacks=Callbacks(),
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         is_coco=False,
         opt=None):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()    #模型测试模式，固定住dropout层和Batch Normalization层
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.safe_load(f)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        # if device.type != 'cpu':
        #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        print(opt.task)
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        val_path_rgb = data['val_rgb']
        val_path_ir = data['val_ir']
        dataloader = create_dataloader_rgb_ir(val_path_rgb, val_path_ir, imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 7) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.75', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map75, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0, 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    
    ## 开始验证 =================================================
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
    # for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(testloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        img_rgb = img[:, :3, :, :]
        img_ir = img[:, 3:, :, :]

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            
            ## 前向推理
            # out:       推理结果 1个 [bs, anchor_num*grid_w*grid_h, xywh+c+20classes] = [1, 19200+4800+1200, 25]
            # train_out: 训练结果 3个 [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
            #                    如: [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
            
            #out, train_out = model(img_rgb, img_ir, augment=augment)  # inference and training outputs
            
            out1, dout = model(img_rgb, img_ir, augment=augment)  
            #out1为最后一个detect，dout为所有detect
            ##
            #3个元组 #3detect
            out = []  ##推理out
            for k in range(0,len(dout)):
                out.append(dout[k][0])
            #for k in range(1,len(dout)):
                #out[0]=torch.cat((out[0],out[k]),1) 
            #out = out[0] #将三个detect结果concat
            
#             train_out = []  ##训练train_out
#             for k in range(0,len(dout)):
#                 train_out.append(dout[k][1])
#             for j in range(3): #3个特征图
#                 for k in range(1,len(dout)):
#                     train_out[0][j]=torch.cat((train_out[0][j],train_out[k][j]),1)           
#             train_out = train_out[0] #将三个detect结果concat
            
            
            ## Inference
            #model1_out = model1(im, augment=augment, val=False)  # inference, loss outputs
            #model2_out = model2(im, augment=opt.augment)[0]
            model1_out = out[0]  #depth
            model2_out = out[1]  #rgb
            model3_out = out[2]  #add
            
            #print(len(model1_out))  #16
            #print(len(model2_out))  #16
            #print(len(model3_out))  #16
            
            t0 += time_synchronized() - t  #模型时间

#             # 计算验证损失
#             # compute_loss不为空 说明正在执行train.py  根据传入的compute_loss计算损失值
#             if compute_loss:
#                 loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls
            
            train_out = []  ##训练train_out
            for k in range(0,len(dout)):
                train_out.append(dout[k][1])
                
            loss_items1 = compute_loss([x.float() for x in train_out[0]], targets.to(device))[1][:3]
            loss_items2 = compute_loss([x.float() for x in train_out[1]], targets.to(device))[1][:3]
            #
            loss_items3 = compute_loss([x.float() for x in train_out[2]], targets.to(device))[1][:3]
            
            loss = []
            #list元素相加
            for m,n,l in zip(loss_items1,loss_items2, loss_items3):
                loss_items= m *0.2 + n *0.3 + l* 0.5
                loss.append(loss_items)

            #loss = loss_items1 * 0.2 + loss_items2 * 0.3 + loss_items3 * 0.5
                
            
            # Run NMS
            # 将真实框target的xywh(因为target是在labelimg中做了归一化的)映射到img(test)尺寸
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            # 是在NMS之前将数据集标签targets添加到模型预测中
            # 这允许在数据集中自动标记(for autolabelling)其他对象(在pred中混入gt) 并且mAP反映了新的混合标签
            # targets: [num_target, img_index+class_index+xywh] = [31, 6]
            # lb: {list: bs} 第一张图片的target[17, 5] 第二张[1, 5] 第三张[7, 5] 第四张[6, 5]
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            model1_out = non_max_suppression(model1_out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
            model2_out = non_max_suppression(model2_out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
            model3_out = non_max_suppression(model3_out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
            
            #print(len(model1_out))  #16
            #print(len(model2_out))  #16
            #print(len(model3_out))  #16
            
            #model2_out = model2_out + model1_out
            #print(len(model2_out))  #32!
            

        # 6.5、统计每张图片的真实框、预测框信息  Statistics per image
        # 为每张图片做统计，写入预测信息到txt文件，生成json文件字典，统计tp等
        # out: list{bs}  [300, 6] [42, 6] [300, 6] [300, 6]  [pred_obj_num, x1y1x2y2+object_conf+cls]
        
        ### 3个模型融合？
        for si, (im0, model1_dets, model2_dets, model3_dets) in enumerate(zip(img_rgb, model1_out, model2_out, model3_out)):
            #
            #print(im0.shape)  #用img torch.Size([6, 384, 672])
            #用img_rgb torch.Size([3, 384, 672])
            im0 = im0.detach().cpu().numpy() * 255
            im0 = im0.transpose((1,2,0)).astype(np.uint8).copy()
            #concat two model's outputs
            if len(model3_dets):
                model3_dets[:, :4] = scale_coords(img.shape[2:], model3_dets[:, :4], im0.shape).round()
                
            if len(model2_dets):  #若model2不为空
                model2_dets[:, :4] = scale_coords(img.shape[2:], model2_dets[:, :4], im0.shape).round()

            if len(model1_dets):
                model1_dets[:, :4] = scale_coords(img.shape[2:], model1_dets[:, :4], im0.shape).round()
            
            # Flag for indicating detection success 检测成功标志
            #detect_success = False
            
            iou_thres = 0.55
            if len(model3_dets)>0 and len(model2_dets)>0 and len(model1_dets)>0:
                #print(333)  
                ##example_wbf_3_models默认iou_thr=0.55，大于该值的框才融合？改为开头设置的iou_thres？
                boxes, scores, labels = example_wbf_3_models(model3_dets.detach().cpu().numpy(), model2_dets.detach().cpu().numpy(), model1_dets.detach().cpu().numpy(), im0, iou_thr=iou_thres)
                boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
                boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height
                
            elif len(model3_dets)>0 and len(model2_dets)>0:
                boxes, scores, labels = example_wbf_2_models(model3_dets.detach().cpu().numpy(), model2_dets.detach().cpu().numpy(), im0, iou_thr=iou_thres)
                boxes2[:,0], boxes2[:,2] = boxes[:,0] * width, boxes[:,2] * width
                boxes2[:,1], boxes2[:,3] = boxes[:,1] * height, boxes[:,3] * height
                
            elif len(model3_dets)>0 and len(model1_dets)>0:
                boxes, scores, labels = example_wbf_2_models(model3_dets.detach().cpu().numpy(), model1_dets.detach().cpu().numpy(), im0, iou_thr=iou_thres)
                boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
                boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height
            
            elif len(model2_dets)>0 and len(model1_dets)>0:
                boxes, scores, labels = example_wbf_2_models(model2_dets.detach().cpu().numpy(), model1_dets.detach().cpu().numpy(), im0, iou_thr=iou_thres)
                boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
                boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height
                
            elif len(model3_dets)>0:
                boxes, scores, labels = example_wbf_1_model(model3_dets.detach().cpu().numpy(), im0, iou_thr=iou_thres)
                boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
                boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height
                
            elif len(model2_dets)>0:
                boxes, scores, labels = example_wbf_1_model(model2_dets.detach().cpu().numpy(), im0, iou_thr=iou_thres)
                boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
                boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height
                
            elif len(model1_dets)>0:
                boxes, scores, labels = example_wbf_1_model(model1_dets.detach().cpu().numpy(), im0, iou_thr=iou_thres)
                boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
                boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height
                
            else: ##没有时返回0
                boxes, scores, labels = np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
                #boxes = boxes + boxes2
                #scores = scores + scores2
                #labels = labels + labels2
                
            for box in boxes:
                cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 3)
               
                
                
#         ## 2个模型
#         ## zip将迭代元素打包成元组，若长度不一舍去长的部分。
#         for si, (im0, model2_dets, model1_dets) in enumerate(zip(img_rgb, model3_out, model2_out)):
#             #
#             #print(im0.shape)  #用img torch.Size([6, 384, 672])
#             #用img_rgb torch.Size([3, 384, 672])
#             im0 = im0.detach().cpu().numpy() * 255
#             im0 = im0.transpose((1,2,0)).astype(np.uint8).copy()
#             #print(im0.shape)  #(384, 672, 3)
#             #后面example_wbf_1_models里面 img_height, img_width = img.shape[1:]
            
#             #model2为空?
#             if len(model2_dets):  #若model2不为空
#                 #scale_coords将坐标coords(x1y1x2y2)从img_shape缩放到im0_shape尺寸（尺寸一致）
#                 model2_dets[:, :4] = scale_coords(img.shape[2:], model2_dets[:, :4], im0.shape).round()
#                 ##归一化坐标到[0,1]，后面example_wbf_2_models里面会归一化
#                 ##为什么坐标会超过1？
#                 #model2_dets[:, 0],  model2_dets[:, 2] = model2_dets[:, 0]/ width,  model2_dets[:, 2]/ width
#                 #model2_dets[:, 1],  model2_dets[:, 3] = model2_dets[:, 0]/ height,  model2_dets[:, 2]/ height

#             if len(model1_dets):
#                 model1_dets[:, :4] = scale_coords(img.shape[2:], model1_dets[:, :4], im0.shape).round()
#                 #model1_dets[:, 0],  model1_dets[:, 2] = model1_dets[:, 0]/ width,  model1_dets[:, 2]/ width
#                 #model1_dets[:, 1],  model1_dets[:, 3] = model1_dets[:, 0]/ height,  model1_dets[:, 2]/ height
            
#             # Flag for indicating detection success 检测成功标志
#             detect_success = False
            
             
#             #print(len(model2_dets))  #0?  #model2为空?
#             #print(len(model1_dets))  #1
                
#             if len(model2_dets)>0 and len(model1_dets)>0:
#                 boxes, scores, labels = example_wbf_2_models(model2_dets.detach().cpu().numpy(), model1_dets.detach().cpu().numpy(), im0)
#                 #通过im0获取图片width、height
#                 boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
#                 boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height
#                 for box in boxes:
#                     cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 3)
#                 detect_success = True
#             elif len(model2_dets)>0:
#                 boxes, scores, labels = example_wbf_1_model(model2_dets.detach().cpu().numpy(), im0)
#                 boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
#                 boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height
#                 for box in boxes:
#                     cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 3)
#                 detect_success = True
#             elif len(model1_dets)>0:
#                 boxes, scores, labels = example_wbf_1_model(model1_dets.detach().cpu().numpy(), im0)
#                 boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
#                 boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height
#                 for box in boxes:
#                     cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 3)
#                 detect_success = True
#             else: ##没有时返回0
#                 boxes, scores, labels = np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
            
            ###
            p_boxes, p_scores, p_labels = boxes, scores, labels
            # Result visualization
            #if detect_success is True:
                #cv2.imshow("detected_image", im0)
                #cv2.waitKey(0)
            
            t1 += time_synchronized() - t  # 累计NMS时间
            
            # 获取第si张图片的gt标签信息 包括class, x, y, w, h    target[:, 0]为标签属于哪张图片的编号
            labels = targets[targets[:, 0] == si, 1:]   # [:, class+xywh]
            nl = len(labels)    # 第si张图片的gt个数
            # 获取标签类别
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path,shape = Path(paths[si]), shapes[si][0]
            # 统计测试图片数量 +1
            seen += 1
            
            # 如果预测为空，则添加空的信息到stats里
            if len(boxes) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            #print(p_boxes.shape) #(1,4)
            if len(p_labels.shape)>1: #(1,) #中途报错维度不一致
                #print(p_labels.shape) #torch.Size([27, 5]) ？怎么会变成二维
                #print(p_labels[0])    #tensor([ 15.00000, 248.25023, 158.49985,  41.16672,  37.00008], device='cuda:0')
                ##只取第1列
                n0 = p_boxes.shape[0]  #中途报错size不一致
                p_labels = p_labels[:n0,0]
            
            ## 报错，/home/ubuntu/miniconda3/envs/WBF/lib/python3.6/site-packages/torch/_tensor.py in __array__
            ## 将报错代码 return self.numpy()改为self.cpu().numpy()即可
            pred = np.concatenate([p_boxes, np.expand_dims(p_scores, axis=1), np.expand_dims(p_labels, axis=1)], axis=1)
            pred = torch.from_numpy(pred).to(device)
            predn = pred.clone()
            
            # 将预测坐标映射到原图img中
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Evaluate 评估
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                #scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                scale_coords(img_rgb[si].shape[1:], tbox, shape, shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            # 将每张图片的预测结果统计到stats中 Append statistics
            # stats: correct, conf, pcls, tcls   bs个 correct, conf, pcls, tcls
            # correct: [pred_num, 10] bool 当前图片每一个预测框在每一个iou条件下是否是TP
            # pred[:, 4]: [pred_num, 1] 当前图片每一个预测框的conf
            # pred[:, 5]: [pred_num, 1] 当前图片每一个预测框的类别
            # tcls: [gt_num, 1] 当前图片所有gt框的class    
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # Append to text file  保存预测信息到txt文件
            if save_txt:
                # gn = [w, h, w, h] 对应图片的宽高  用于后面归一化
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    # xyxy -> xywh 并作归一化处理
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    # 保存预测类别和坐标值到对应图片id.txt文件中
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
            # Append to pycocotools JSON dictionary  将预测信息保存到coco格式的json字典(后面存入json文件)
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})
            callbacks.run('on_val_image_end', pred, predn, path, names, img_rgb[si]) ##早停训练？

            
        
    # 统计stats中所有图片的统计结果 将stats列表的信息拼接到一起
    # stats(concat后): list{4} correct, conf, pcls, tcls  统计出的整个数据集的GT
    # correct [img_sum, 10] 整个数据集所有图片中所有预测框在每一个iou条件下是否是TP  [1905, 10]
    # conf [img_sum] 整个数据集所有图片中所有预测框的conf  [1905]
    # pcls [img_sum] 整个数据集所有图片中所有预测框的类别   [1905]
    # tcls [gt_sum] 整个数据集所有图片所有gt框的class     [929]
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    # stats[0].any(): stats[0]是否全部为False, 是则返回 False, 如果有一个为 True, 则返回 True
    if len(stats) and stats[0].any():
        # 根据上面的统计预测结果计算p, r, ap, f1, ap_class（ap_per_class函数是计算每个类的mAP等指标的）等指标
        # p: [nc] 最大平均f1时每个类别的precision
        # r: [nc] 最大平均f1时每个类别的recall
        # ap: [71, 10] 数据集每个类别在10个iou阈值下的mAP
        # f1 [nc] 最大平均f1时每个类别的f1
        # ap_class: [nc] 返回数据集中所有的类别index
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        # print("mAP75", ap[:, 5].mean(-1))
        # ap50: [nc] 所有类别的mAP@0.5   ap: [nc] 所有类别的mAP@0.5:0.95
        ap50, ap75, ap = ap[:, 0], ap[:, 5], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        # mp: [1] 所有类别的平均precision(最大f1时)
        # mr: [1] 所有类别的平均recall(最大f1时)
        # map50: [1] 所有类别的平均mAP@0.5
        # map: [1] 所有类别的平均mAP@0.5:0.95
        mp, mr, map50, map75, map = p.mean(), r.mean(), ap50.mean(), ap75.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class 统计整个数据集的gt框中数据集各个类别的个数
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 5  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map75, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap75[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)

        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results  返回测试指标结果
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map  # [nc] nc个平均mAP@0.5:0.95
    for i, c in enumerate(ap_class):
        maps[c] = ap[i] # 所有类别的mAP@0.5:0.95
    # (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()): {tuple:7}
    #      0: mp [1] 所有类别的平均precision(最大f1时)
    #      1: mr [1] 所有类别的平均recall(最大f1时)
    #      2: map50 [1] 所有类别的平均mAP@0.5
    #      3: map [1] 所有类别的平均mAP@0.5:0.95
    #      4: val_box_loss [1] 验证集回归损失
    #      5: val_obj_loss [1] 验证集置信度损失
    #      6: val_cls_loss [1] 验证集分类损失
    # maps: [80] 所有类别的mAP@0.5:0.95
    # t: {tuple: 3} 0: 打印前向传播耗费的总时间   1: nms耗费总时间   2: 总时间
    return (mp, mr, map50, map75, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='/home/fqy/proj/multispectral-object-detection/best.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='./data/multispectral/FLIR_aligned.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=64, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', default=False, action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', default=True, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    print(opt.data)
    check_requirements()

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             opt=opt
             )
    # results, maps, times = test.test(data_dict,
    #                                  batch_size=batch_size * 2,
    #                                  imgsz=imgsz_test,
    #                                  model=ema.ema,
    #                                  single_cls=opt.single_cls,
    #                                  dataloader=testloader,
    #                                  save_dir=save_dir,
    #                                  verbose=nc < 50 and final_epoch,
    #                                  plots=plots and final_epoch,
    #                                  wandb_logger=wandb_logger,
    #                                  compute_loss=compute_loss,
    #                                  is_coco=is_coco)

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, opt=opt)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, opt=opt)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot

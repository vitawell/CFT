import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box, xywh2xyxy
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.WBF.examples.example import example_wbf_3_models, example_wbf_2_models, example_wbf_1_model


def detect(opt):
    source1, source2, source3, weights, view_img, save_txt, imgsz = opt.source1, opt.source2, opt.source3, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    save_img = not opt.nosave and not source1.endswith('.txt')  # save inference images
    webcam = source1.isnumeric() or source1.endswith('.txt') or source1.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source1, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source1, img_size=imgsz, stride=stride)
        dataset2 = LoadImages(source2, img_size=imgsz, stride=stride)
        dataset3 = LoadImages(source3, img_size=imgsz, stride=stride)

    # # Run inference
    # if device.type != 'cpu':
    #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    t0 = time.time()
    img_num = 0
    fps_sum = 0
    for (path, img, im0s, vid_cap), (path_, img2, im0s_, vid_cap_), (path3, img3, im3s, vid_cap3) in zip(dataset, dataset2, dataset3):
        # print(path)
        # print(path_)
        img = torch.from_numpy(img).to(device)
        img2 = torch.from_numpy(img2).to(device)
        img3 = torch.from_numpy(img3).to(device)

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        img2 = img2.half() if half else img2.float()  # uint8 to fp16/32
        img2 /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img2.ndimension() == 3:
            img2 = img2.unsqueeze(0)
        img3 = img3.half() if half else img3.float()  # uint8 to fp16/32
        img3 /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img3.ndimension() == 3:
            img3 = img3.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        #pred = model(img, img2, augment=opt.augment)[0]
        ##pred为输出的推理out
        pred = model(img, img2, img3, augment=opt.augment)[1]
        out = []
        for k in range(0,len(pred)):
            out.append(pred[k][0])
        #for k in range(1,len(pred)):
            #out[0]=torch.cat((out[0],out[k]),1) 
        #pred = out[0] #将三个detect结果concat
        
        if len(out)==3:
            pred1 = out[0]  #
            pred2 = out[1]  #
            pred3 = out[2]

            # Apply NMS
            #pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            pred1 = non_max_suppression(pred1, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            pred2 = non_max_suppression(pred2, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            pred3 = non_max_suppression(pred3, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

            t2 = time_synchronized()

            ###
            # Apply Classifier
            if classify:
                pred1 = apply_classifier(pred1, modelc, img, im0s)
                pred2 = apply_classifier(pred2, modelc, img, im0s)
                pred3 = apply_classifier(pred3, modelc, img, im0s)

            # Process detections
            ### 3个detect融合？
            for si, (im0, model1_dets, model2_dets, model3_dets) in enumerate(zip(img, pred1, pred2, pred3)):
                #
                #print(im0.shape)  #用img torch.Size([6, 384, 672])
                #用img_rgb torch.Size([3, 384, 672])
                im0 = im0.detach().cpu().numpy() * 255
                im0 = im0.transpose((1,2,0)).astype(np.uint8).copy()

                width = im0.shape[1]
                height = im0.shape[0]


                if len(model3_dets):
                    model3_dets[:, :4] = scale_coords(img.shape[2:], model3_dets[:, :4], im0.shape).round()

                if len(model2_dets):  #若model2不为空
                    model2_dets[:, :4] = scale_coords(img.shape[2:], model2_dets[:, :4], im0.shape).round()

                if len(model1_dets):
                    model1_dets[:, :4] = scale_coords(img.shape[2:], model1_dets[:, :4], im0.shape).round()

                # Flag for indicating detection success 检测成功标志
                #detect_success = False

                #iou_thres = 0.55
                if len(model3_dets)>0 and len(model2_dets)>0 and len(model1_dets)>0:
                    #print(333)  
                    ##example_wbf_3_models默认iou_thr=0.55，大于该值的框才融合？改为开头设置的iou_thres？
                    boxes, scores, labels = example_wbf_3_models(model3_dets.detach().cpu().numpy(), model2_dets.detach().cpu().numpy(), model1_dets.detach().cpu().numpy(), im0, iou_thr=0.55)
                    boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
                    boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height

                elif len(model3_dets)>0 and len(model2_dets)>0:
                    boxes, scores, labels = example_wbf_2_models(model3_dets.detach().cpu().numpy(), model2_dets.detach().cpu().numpy(), im0, iou_thr=0.55)
                    boxes2[:,0], boxes2[:,2] = boxes[:,0] * width, boxes[:,2] * width
                    boxes2[:,1], boxes2[:,3] = boxes[:,1] * height, boxes[:,3] * height

                elif len(model3_dets)>0 and len(model1_dets)>0:
                    boxes, scores, labels = example_wbf_2_models(model3_dets.detach().cpu().numpy(), model1_dets.detach().cpu().numpy(), im0, iou_thr=0.55)
                    boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
                    boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height

                elif len(model2_dets)>0 and len(model1_dets)>0:
                    boxes, scores, labels = example_wbf_2_models(model2_dets.detach().cpu().numpy(), model1_dets.detach().cpu().numpy(), im0, iou_thr=0.55)
                    boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
                    boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height

                elif len(model3_dets)>0:
                    boxes, scores, labels = example_wbf_1_model(model3_dets.detach().cpu().numpy(), im0, iou_thr=0.55)
                    boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
                    boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height

                elif len(model2_dets)>0:
                    boxes, scores, labels = example_wbf_1_model(model2_dets.detach().cpu().numpy(), im0, iou_thr=0.55)
                    boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
                    boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height

                elif len(model1_dets)>0:
                    boxes, scores, labels = example_wbf_1_model(model1_dets.detach().cpu().numpy(), im0, iou_thr=0.55)
                    boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
                    boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height

                else: ##没有时返回0
                    boxes, scores, labels = np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
                    #boxes = boxes + boxes2
                    #scores = scores + scores2
                    #labels = labels + labels2

                for box in boxes:
                    cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 3)

                ###
                p_boxes, p_scores, p_labels = boxes, scores, labels

            #for i, det in enumerate(pred):  # detections per image

                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                    p, s, im0_, frame = path, '', im0s_.copy(), getattr(dataset2, 'frame', 0)
                    p, s, im3, frame = path, '', im3s.copy(), getattr(dataset3, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                # print(gn)

                # print(det)
                if len(p_boxes):
                    p_boxes = torch.from_numpy(p_boxes)
                    # Rescale boxes from img_size to im0 size
                    p_boxes[:, :4] = scale_coords(img.shape[2:], p_boxes[:, :4], im0.shape).round()

                    # Print results
                    for c in p_labels:
                        n = (p_labels == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    # xyxy是预测框左上角和右下角坐标
                    for si, (*xyxy, conf, cls) in enumerate(zip(p_boxes, p_scores, p_labels)):
                        xyxy = xyxy[0].tolist()
                        #print(xyxy)
                        #print(conf)
                        #print(cls)
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or opt.save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')

                            # rgb图像是im0
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                            # ir图像是im0_
                            plot_one_box(xyxy, im0_, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                            # 3图像是im3
                            plot_one_box(xyxy, im3, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                            if opt.save_crop:
                                save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.6f}s, {1/(t2 - t1):.6f}Hz)')
                # add all the fps
                img_num += 1
                fps_sum += 1/(t2 - t1)

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        save_path_rgb = save_path.split('.')[0] + '_rgb.' + save_path.split('.')[1]
                        save_path_ir = save_path.split('.')[0] + '_ir.' + save_path.split('.')[1]
                        save_path_3 = save_path.split('.')[0] + '_3.' + save_path.split('.')[1]
                        print(save_path_rgb)
                        cv2.imwrite(save_path_rgb, im0)
                        cv2.imwrite(save_path_ir, im0_)
                        cv2.imwrite(save_path_3, im3)
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)
                    
                    
        if len(out)==2:
            pred1 = out[0]  #
            pred2 = out[1]  #
            pred3 = out[2]

            # Apply NMS
            #pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            pred1 = non_max_suppression(pred1, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            pred2 = non_max_suppression(pred2, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            pred3 = non_max_suppression(pred3, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

            t2 = time_synchronized()

            ###
            # Apply Classifier
            if classify:
                pred1 = apply_classifier(pred1, modelc, img, im0s)
                pred2 = apply_classifier(pred2, modelc, img, im0s)
                pred3 = apply_classifier(pred3, modelc, img, im0s)

            # Process detections
            ### 3个detect融合？
            for si, (im0, model1_dets, model2_dets, model3_dets) in enumerate(zip(img, pred1, pred2, pred3)):
                #
                #print(im0.shape)  #用img torch.Size([6, 384, 672])
                #用img_rgb torch.Size([3, 384, 672])
                im0 = im0.detach().cpu().numpy() * 255
                im0 = im0.transpose((1,2,0)).astype(np.uint8).copy()

                width = im0.shape[1]
                height = im0.shape[0]


                if len(model3_dets):
                    model3_dets[:, :4] = scale_coords(img.shape[2:], model3_dets[:, :4], im0.shape).round()

                if len(model2_dets):  #若model2不为空
                    model2_dets[:, :4] = scale_coords(img.shape[2:], model2_dets[:, :4], im0.shape).round()

                if len(model1_dets):
                    model1_dets[:, :4] = scale_coords(img.shape[2:], model1_dets[:, :4], im0.shape).round()

                # Flag for indicating detection success 检测成功标志
                #detect_success = False

                #iou_thres = 0.55
                if len(model3_dets)>0 and len(model2_dets)>0 and len(model1_dets)>0:
                    #print(333)  
                    ##example_wbf_3_models默认iou_thr=0.55，大于该值的框才融合？改为开头设置的iou_thres？
                    boxes, scores, labels = example_wbf_3_models(model3_dets.detach().cpu().numpy(), model2_dets.detach().cpu().numpy(), model1_dets.detach().cpu().numpy(), im0, iou_thr=0.55)
                    boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
                    boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height

                elif len(model3_dets)>0 and len(model2_dets)>0:
                    boxes, scores, labels = example_wbf_2_models(model3_dets.detach().cpu().numpy(), model2_dets.detach().cpu().numpy(), im0, iou_thr=0.55)
                    boxes2[:,0], boxes2[:,2] = boxes[:,0] * width, boxes[:,2] * width
                    boxes2[:,1], boxes2[:,3] = boxes[:,1] * height, boxes[:,3] * height

                elif len(model3_dets)>0 and len(model1_dets)>0:
                    boxes, scores, labels = example_wbf_2_models(model3_dets.detach().cpu().numpy(), model1_dets.detach().cpu().numpy(), im0, iou_thr=0.55)
                    boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
                    boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height

                elif len(model2_dets)>0 and len(model1_dets)>0:
                    boxes, scores, labels = example_wbf_2_models(model2_dets.detach().cpu().numpy(), model1_dets.detach().cpu().numpy(), im0, iou_thr=0.55)
                    boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
                    boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height

                elif len(model3_dets)>0:
                    boxes, scores, labels = example_wbf_1_model(model3_dets.detach().cpu().numpy(), im0, iou_thr=0.55)
                    boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
                    boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height

                elif len(model2_dets)>0:
                    boxes, scores, labels = example_wbf_1_model(model2_dets.detach().cpu().numpy(), im0, iou_thr=0.55)
                    boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
                    boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height

                elif len(model1_dets)>0:
                    boxes, scores, labels = example_wbf_1_model(model1_dets.detach().cpu().numpy(), im0, iou_thr=0.55)
                    boxes[:,0], boxes[:,2] = boxes[:,0] * width, boxes[:,2] * width
                    boxes[:,1], boxes[:,3] = boxes[:,1] * height, boxes[:,3] * height

                else: ##没有时返回0
                    boxes, scores, labels = np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
                    #boxes = boxes + boxes2
                    #scores = scores + scores2
                    #labels = labels + labels2

                for box in boxes:
                    cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 3)

                ###
                p_boxes, p_scores, p_labels = boxes, scores, labels

            #for i, det in enumerate(pred):  # detections per image

                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                    p, s, im0_, frame = path, '', im0s_.copy(), getattr(dataset2, 'frame', 0)
                    p, s, im3, frame = path, '', im3s.copy(), getattr(dataset3, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                # print(gn)

                # print(det)
                if len(p_boxes):
                    p_boxes = torch.from_numpy(p_boxes)
                    # Rescale boxes from img_size to im0 size
                    p_boxes[:, :4] = scale_coords(img.shape[2:], p_boxes[:, :4], im0.shape).round()

                    # Print results
                    for c in p_labels:
                        n = (p_labels == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    # xyxy是预测框左上角和右下角坐标
                    for si, (*xyxy, conf, cls) in enumerate(zip(p_boxes, p_scores, p_labels)):
                        xyxy = xyxy[0].tolist()
                        #print(xyxy)
                        #print(conf)
                        #print(cls)
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or opt.save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')

                            # rgb图像是im0
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                            # ir图像是im0_
                            plot_one_box(xyxy, im0_, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                            # 3图像是im3
                            plot_one_box(xyxy, im3, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                            if opt.save_crop:
                                save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.6f}s, {1/(t2 - t1):.6f}Hz)')
                # add all the fps
                img_num += 1
                fps_sum += 1/(t2 - t1)

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        save_path_rgb = save_path.split('.')[0] + '_rgb.' + save_path.split('.')[1]
                        save_path_ir = save_path.split('.')[0] + '_ir.' + save_path.split('.')[1]
                        save_path_3 = save_path.split('.')[0] + '_3.' + save_path.split('.')[1]
                        print(save_path_rgb)
                        cv2.imwrite(save_path_rgb, im0)
                        cv2.imwrite(save_path_ir, im0_)
                        cv2.imwrite(save_path_3, im3)
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)
        ###
#         # Apply Classifier
#         if classify:
#             pred = apply_classifier(pred, modelc, img, im0s)

#         # Process detections
#         for i, det in enumerate(pred):  # detections per image

#             if webcam:  # batch_size >= 1
#                 p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
#             else:
#                 p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
#                 p, s, im0_, frame = path, '', im0s_.copy(), getattr(dataset2, 'frame', 0)

#             p = Path(p)  # to Path
#             save_path = str(save_dir / p.name)  # img.jpg
#             txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
#             s += '%gx%g ' % img.shape[2:]  # print string
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#             # print(gn)

#             # print(det)
#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

#                 # Print results
#                 for c in det[:, -1].unique():
#                     n = (det[:, -1] == c).sum()  # detections per class
#                     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

#                 # Write results
#                 # xyxy是预测框左上角和右下角坐标
#                 for *xyxy, conf, cls in reversed(det):
#                     if save_txt:  # Write to file
#                         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                         line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
#                         with open(txt_path + '.txt', 'a') as f:
#                             f.write(('%g ' * len(line)).rstrip() % line + '\n')

#                     if save_img or opt.save_crop or view_img:  # Add bbox to image
#                         c = int(cls)  # integer class
#                         label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        
#                         # rgb图像是im0
#                         plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
#                         # ir图像是im0_
#                         plot_one_box(xyxy, im0_, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
#                         if opt.save_crop:
#                             save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

#             # Print time (inference + NMS)
#             print(f'{s}Done. ({t2 - t1:.6f}s, {1/(t2 - t1):.6f}Hz)')
#             # add all the fps
#             img_num += 1
#             fps_sum += 1/(t2 - t1)

#             # Stream results
#             if view_img:
#                 cv2.imshow(str(p), im0)
#                 cv2.waitKey(1)  # 1 millisecond

#             # Save results (image with detections)
#             if save_img:
#                 if dataset.mode == 'image':
#                     save_path_rgb = save_path.split('.')[0] + '_rgb.' + save_path.split('.')[1]
#                     save_path_ir = save_path.split('.')[0] + '_ir.' + save_path.split('.')[1]
#                     print(save_path_rgb)
#                     cv2.imwrite(save_path_rgb, im0)
#                     cv2.imwrite(save_path_ir, im0_)
#                 else:  # 'video' or 'stream'
#                     if vid_path != save_path:  # new video
#                         vid_path = save_path
#                         if isinstance(vid_writer, cv2.VideoWriter):
#                             vid_writer.release()  # release previous video writer
#                         if vid_cap:  # video
#                             fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                         else:  # stream
#                             fps, w, h = 30, im0.shape[1], im0.shape[0]
#                             save_path += '.mp4'
#                         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#                     vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    print(f'Average Speed: {fps_sum/img_num:.6f}Hz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/home/fqy/proj/multispectral-object-detection/best.pt', help='model.pt path(s)')
    parser.add_argument('--source1', type=str, default='/home/fqy/DATA/FLIR_ADAS_1_3/align/yolo/test/rgb/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source2', type=str, default='/home/fqy/DATA/FLIR_ADAS_1_3/align/yolo/test/ir', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source3', type=str, default='/home/fqy/DATA/FLIR_ADAS_1_3/align/yolo/test/3', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=False, action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt=opt)
                strip_optimizer(opt.weights)
        else:
            print("helloxxxxxxxxxxxxxxxxxxxx")
            detect(opt=opt)

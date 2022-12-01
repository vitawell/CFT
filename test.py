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


def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
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
    model.eval()
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
            ##不输出dout时：
            ##print(len(out))  #16?
            ##print(out.size)  #<built-in method size of Tensor object at 0x7f5d21856f98>
            ##print(out.size())  #torch.Size([16, 14553, 21])
            ##print(len(train_out))  #3
            ##print(train_out.size())  #list no size
            
            out1, dout = model(img_rgb, img_ir, augment=augment) 
            ##输出dout时：
            out = out1[0]
            train_out = out1[1]
            ##print(len(out1))  #2
            ##print(out1.size())  #tuple no size
            ##print(out.size())  #torch.Size([16, 14553, 21])
            ##print(len(train_out))  #3
            ##print(dout.size()) #list no size
            
            t0 += time_synchronized() - t

            # 计算验证损失
            # compute_loss不为空 说明正在执行train.py  根据传入的compute_loss计算损失值
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

            # Run NMS
            # 将真实框target的xywh(因为target是在labelimg中做了归一化的)映射到img(test)尺寸
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            # 是在NMS之前将数据集标签targets添加到模型预测中
            # 这允许在数据集中自动标记(for autolabelling)其他对象(在pred中混入gt) 并且mAP反映了新的混合标签
            # targets: [num_target, img_index+class_index+xywh] = [31, 6]
            # lb: {list: bs} 第一张图片的target[17, 5] 第二张[1, 5] 第三张[7, 5] 第四张[6, 5]
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
            t1 += time_synchronized() - t  # 累计NMS时间

        # 6.5、统计每张图片的真实框、预测框信息  Statistics per image
        # 为每张图片做统计，写入预测信息到txt文件，生成json文件字典，统计tp等
        # out: list{bs}  [300, 6] [42, 6] [300, 6] [300, 6]  [pred_obj_num, x1y1x2y2+object_conf+cls]
        for si, pred in enumerate(out):
            # 获取第si张图片的gt标签信息 包括class, x, y, w, h    target[:, 0]为标签属于哪张图片的编号
            labels = targets[targets[:, 0] == si, 1:]   # [:, class+xywh]
            nl = len(labels)    # 第si张图片的gt个数
            # 获取标签类别
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])  # 第si张图片的地址
            # 统计测试图片数量 +1
            seen += 1
            
            # 如果预测为空，则添加空的信息到stats里
            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            # 将预测坐标映射到原图img中
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

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

            # W&B logging - Media Panel Plots  保存预测信息到wandb_logger(类似tensorboard)中
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

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

            # 计算混淆矩阵、计算correct、生成stats
            # Assign all predictions as incorrect  初始化预测评定 niou为iou阈值的个数
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices  用于存放已检测到的目标
                tcls_tensor = labels[:, 0]  # 当前图片的所有gt的类别

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])    #获得xyxy格式的框
                # 将预测框映射到原图img
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    # 计算混淆矩阵 confusion_matrix
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class  对图片中每个类别单独处理
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices 预测框中该类别的索引
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices  

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        # predn[pi, :4]: 属于该类的预测框[144, 4]  tbox[ti]: 属于该类的gt框[13, 4]
                        # box_iou: [144, 4] + [13, 4] => [144, 13]  计算属于该类的预测框与属于该类的gt框的iou
                        # .max(1): [144] 选出每个预测框与所有gt box中最大的iou值, i为最大iou值时对应的gt索引
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):  # j: ious中>0.5的索引 只有iou>=0.5才是TP
                            d = ti[i[j]]  # detected target  获得检测到的目标
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)  # 将当前检测到的gt框d添加到detected()
                                # iouv为以0.05为步长  0.5-0.95的序列
                                # 从所有TP中获取不同iou阈值下的TP true positive  并在correct中记录下哪个预测框是哪个iou阈值下的TP
                                # correct: [pred_num, 10] = [300, 10]  记录着哪个预测框在哪个iou阈值下是TP
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                # 如果检测到的目标值等于gt框的个数 就结束
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # 将每张图片的预测结果统计到stats中 Append statistics
            # stats: correct, conf, pcls, tcls   bs个 correct, conf, pcls, tcls
            # correct: [pred_num, 10] bool 当前图片每一个预测框在每一个iou条件下是否是TP
            # pred[:, 4]: [pred_num, 1] 当前图片每一个预测框的conf
            # pred[:, 5]: [pred_num, 1] 当前图片每一个预测框的类别
            # tcls: [gt_num, 1] 当前图片所有gt框的class
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        # 画出前三个batch的图片的ground truth和预测框predictions(两个图)一起保存
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels  # ground truth
            # Thread  表示在单独的控制线程中运行的活动 创建一个单线程(子线程)来执行函数 由这个子进程全权负责这个函数
            # target: 执行的函数  args: 传入的函数参数  daemon: 当主线程结束后, 由他创建的子线程Thread也已经自动结束了
            # .start(): 启动线程  当thread一启动的时候, 就会运行我们自己定义的这个函数plot_images
            # 如果在plot_images里面打开断点调试, 可以发现子线程暂停, 但是主线程还是在正常的训练(还是正常的跑)
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

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

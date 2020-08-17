"""
(CRNN) models training

Training strategy:
------------------
1. the following pairs of classes will be treated the same:
    (RBBB, CRBBB), (PAC, SVPB), (PVC, VPB)
    normalization of labels (classes) will be
    CRBB ---> RBBB, SVPB --- > PAC, VPB ---> PVC

2. the following classes will be determined by the special detectors:
    PR, LAD, RAD, LQRSV, Brady,

3. models will be trained for each tranche separatly:
    tranche A and B are from the same source, hence will be treated one during training,
    the distribution of the classes for each tranche are as follows:
        A+B: {'IAVB': 828, 'AF': 1374, 'AFL': 54, 'Brady': 271, 'CRBBB': 113, 'IRBBB': 86, 'LBBB': 274, 'NSIVCB': 4, 'PR': 3, 'PAC': 689, 'PVC': 188, 'LQT': 4, 'QAb': 1, 'RAD': 1, 'RBBB': 1858, 'SA': 11, 'SB': 45, 'NSR': 922, 'STach': 303, 'SVPB': 53, 'TAb': 22, 'TInv': 5, 'VPB': 8}
        C: {'AF': 2, 'Brady': 11, 'NSIVCB': 1, 'PAC': 3, 'RBBB': 2, 'SA': 2, 'STach': 11, 'SVPB': 4, 'TInv': 1}
        D: {'AF': 15, 'AFL': 1, 'NSR': 80, 'STach': 1}
        E: {'IAVB': 797, 'AF': 1514, 'AFL': 73, 'CRBBB': 542, 'IRBBB': 1118, 'LAnFB': 1626, 'LAD': 5146, 'LBBB': 536, 'LQRSV': 182, 'NSIVCB': 789, 'PR': 296, 'PAC': 398, 'LPR': 340, 'LQT': 118, 'QAb': 548, 'RAD': 343, 'SA': 772, 'SB': 637, 'NSR': 18092, 'STach': 826, 'SVPB': 157, 'TAb': 2345, 'TInv': 294}
        F: {'IAVB': 769, 'AF': 570, 'AFL': 186, 'Brady': 6, 'CRBBB': 28, 'IRBBB': 407, 'LAnFB': 180, 'LAD': 940, 'LBBB': 231, 'LQRSV': 374, 'NSIVCB': 203, 'PAC': 639, 'LQT': 1391, 'QAb': 464, 'RAD': 83, 'RBBB': 542, 'SA': 455, 'SB': 1677, 'NSR': 1752, 'STach': 1261, 'SVPB': 1, 'TAb': 2306, 'TInv': 812, 'VPB': 357}
    hence in this manner, training classes for each tranche are as follows:
        A+B: [IAVB', 'AF', 'AFL',  'IRBBB', 'LBBB', 'PAC', 'PVC', 'RBBB', 'SB', 'NSR', 'STach', 'TAb']
        E: ['IAVB', 'AF', 'AFL', 'IRBBB', 'LAnFB', 'LBBB', 'NSIVCB', 'PAC', 'LPR', 'LQT', 'QAb', 'SA', 'SB', 'NSR', 'STach', 'TAb', 'TInv']
        F: ['IAVB', 'AF', 'AFL', 'IRBBB', 'LAnFB', 'LBBB', 'NSIVCB', 'PAC', 'LQT', 'QAb', 'RBBB', 'SA', 'SB', 'NSR', 'STach', 'TAb', 'TInv', 'VPB']
    tranches C, D have too few recordings (recordings of C are long), which shall not be used to train separate models?

4. one model will be trained using the whole dataset

"""
import os
import time
import logging
import argparse
from copy import deepcopy
from collections import deque
import datetime
from typing import Union, Optional, Sequence, NoReturn
from numbers import Real, Number

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from easydict import EasyDict as ED

from models.ecg_crnn import ATI_CNN
from model_configs.ati_cnn import ATI_CNN_CONFIG
from cfg import ModelCfg, TrainCfg


__all__ = []


def train(model:nn.Module, device:torch.device, config:dict, epochs:int=5, batch_size:int=1, save_ckpt:bool=True, log_step:int=20, logger:Optional[logging.Logger]=None):
    """
    """
    train_dataset = ACNE04(label_path=config.train_label, cfg=config, train=True)
    val_dataset = ACNE04(label_path=config.val_label, cfg=config, train=False)

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch // config.subdivisions,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,  # setting False would result in error
        collate_fn=collate,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch // config.subdivisions,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,  # setting False would result in error
        collate_fn=val_collate,
    )

    writer = SummaryWriter(
        log_dir=config.TRAIN_TENSORBOARD_DIR,
        filename_suffix=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}',
        comment=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}',
    )
    
    max_itr = config.TRAIN_EPOCHS * n_train
    # global_step = cfg.TRAIN_MINEPOCH * n_train
    global_step = 0
    if logger:
        logger.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {config.batch}
            Subdivisions:    {config.subdivisions}
            Learning rate:   {config.learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_ckpt}
            Device:          {device.type}
            Images size:     {config.width}
            Optimizer:       {config.TRAIN_OPTIMIZER}
            Dataset classes: {config.classes}
            Train label path:{config.train_label}
            Pretrained:      {config.pretrained}
        ''')

    # learning rate setup
    def burnin_schedule(i):
        if i < config.burn_in:
            factor = pow(i / config.burn_in, 4)
        elif i < config.steps[0]:
            factor = 1.0
        elif i < config.steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    if config.TRAIN_OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config.learning_rate / config.batch,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    elif config.TRAIN_OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config.learning_rate / config.batch,
            momentum=config.momentum,
            weight_decay=config.decay,
        )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    criterion = Yolo_loss(
        n_classes=config.classes,
        device=device,
        batch=config.batch // config.subdivisions,
        iou_type=config.iou_type,
    )
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=6, min_lr=1e-7)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, 0.001, 1e-6, 20)

    save_prefix = 'Yolov4_epoch'
    saved_models = deque()
    model.train()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_step = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=100) as pbar:
            for i, batch in enumerate(train_loader):
                global_step += 1
                epoch_step += 1
                images = batch[0]
                bboxes = batch[1]

                images = images.to(device=device, dtype=torch.float32)
                bboxes = bboxes.to(device=device)

                bboxes_pred = model(images)
                loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(bboxes_pred, bboxes)
                # loss = loss / config.subdivisions
                loss.backward()

                epoch_loss += loss.item()

                if global_step  % config.subdivisions == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                if global_step % (log_step * config.subdivisions) == 0:
                    writer.add_scalar('train/Loss', loss.item(), global_step)
                    writer.add_scalar('train/loss_xy', loss_xy.item(), global_step)
                    writer.add_scalar('train/loss_wh', loss_wh.item(), global_step)
                    writer.add_scalar('train/loss_obj', loss_obj.item(), global_step)
                    writer.add_scalar('train/loss_cls', loss_cls.item(), global_step)
                    writer.add_scalar('train/loss_l2', loss_l2.item(), global_step)
                    writer.add_scalar('lr', scheduler.get_lr()[0] * config.batch, global_step)
                    pbar.set_postfix(**{
                        'loss (batch)': loss.item(),
                        'loss_xy': loss_xy.item(),
                        'loss_wh': loss_wh.item(),
                        'loss_obj': loss_obj.item(),
                        'loss_cls': loss_cls.item(),
                        'loss_l2': loss_l2.item(),
                        'lr': scheduler.get_lr()[0] * config.batch
                    })
                    if logger:
                        logger.info(f'Train step_{global_step}: loss : {loss.item()},loss xy : {loss_xy.item()}, loss wh : {loss_wh.item()}, loss obj : {loss_obj.item()}, loss cls : {loss_cls.item()}, loss l2 : {loss_l2.item()}, lr : {scheduler.get_lr()[0] * config.batch}')

                pbar.update(images.shape[0])
                
            # TODO: eval for each epoch using `evaluate`
            eval_model = Yolov4(yolov4conv137weight=None, n_classes=config.classes, inference=True)
            eval_model.load_state_dict(model.state_dict())
            eval_model.to(device)
            evaluator = evaluate(eval_model, val_loader, config, device, logger)
            del eval_model

            stats = evaluator.coco_eval['bbox'].stats
            writer.add_scalar('train/AP', stats[0], global_step)
            writer.add_scalar('train/AP50', stats[1], global_step)
            writer.add_scalar('train/AP75', stats[2], global_step)
            writer.add_scalar('train/AP_small', stats[3], global_step)
            writer.add_scalar('train/AP_medium', stats[4], global_step)
            writer.add_scalar('train/AP_large', stats[5], global_step)
            writer.add_scalar('train/AR1', stats[6], global_step)
            writer.add_scalar('train/AR10', stats[7], global_step)
            writer.add_scalar('train/AR100', stats[8], global_step)
            writer.add_scalar('train/AR_small', stats[9], global_step)
            writer.add_scalar('train/AR_medium', stats[10], global_step)
            writer.add_scalar('train/AR_large', stats[11], global_step)

            if save_ckpt:
                try:
                    os.mkdir(config.checkpoints)
                    if logger:
                        logger.info('Created checkpoint directory')
                except OSError:
                    pass
                save_path = os.path.join(config.checkpoints, f'{save_prefix}{epoch + 1}_{_get_date_str()}.pth')
                torch.save(model.state_dict(), save_path)
                if logger:
                    logger.info(f'Checkpoint {epoch + 1} saved!')
                saved_models.append(save_path)
                # remove outdated models
                if len(saved_models) > config.keep_checkpoint_max > 0:
                    model_to_remove = saved_models.popleft()
                    try:
                        os.remove(model_to_remove)
                    except:
                        logger.info(f'failed to remove {model_to_remove}')

    writer.close()


@torch.no_grad()
def evaluate(model, data_loader, cfg, device, logger=None, **kwargs):
    """ finished, tested
    """
    # cpu_device = torch.device("cpu")
    model.eval()
    # header = 'Test:'

    coco = convert_to_coco_api(data_loader.dataset, bbox_fmt='coco')
    coco_evaluator = CocoEvaluator(coco, iou_types = ["bbox"], bbox_fmt='coco')

    for images, targets in data_loader:
        model_input = [[cv2.resize(img, (cfg.w, cfg.h))] for img in images]
        model_input = np.concatenate(model_input, axis=0)
        model_input = model_input.transpose(0, 3, 1, 2)
        model_input = torch.from_numpy(model_input).div(255.0)
        model_input = model_input.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(model_input)

        # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        # outputs = outputs.cpu().detach().numpy()
        res = {}
        # for img, target, output in zip(images, targets, outputs):
        for img, target, (boxes, confs) in zip(images, targets, outputs):
            img_height, img_width = img.shape[:2]
            # boxes = output[...,:4].copy()  # output boxes in yolo format
            boxes = boxes.squeeze(2).cpu().detach().numpy()
            boxes[...,2:] = boxes[...,2:] - boxes[...,:2] # Transform [x1, y1, x2, y2] to [x1, y1, w, h]
            boxes[...,0] = boxes[...,0]*img_width
            boxes[...,1] = boxes[...,1]*img_height
            boxes[...,2] = boxes[...,2]*img_width
            boxes[...,3] = boxes[...,3]*img_height
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # confs = output[...,4:].copy()
            confs = confs.cpu().detach().numpy()
            labels = np.argmax(confs, axis=1).flatten()
            labels = torch.as_tensor(labels, dtype=torch.int64)
            scores = np.max(confs, axis=1).flatten()
            scores = torch.as_tensor(scores, dtype=torch.float32)
            res[target["image_id"].item()] = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }

        debug = kwargs.get("debug", [])
        if isinstance(debug, str):
            debug = [debug]
        debug = [item.lower() for item in debug]
        if 'iou' in debug:
            from tool.utils_iou_test import bboxes_iou_test
            ouput_boxes = np.array(post_processing(None, 0.5, 0.5, outputs)[0])[...,:4]
            img_height, img_width = images[0].shape[:2]
            ouput_boxes[...,0] = ouput_boxes[...,0] * img_width
            ouput_boxes[...,1] = ouput_boxes[...,1] * img_height
            ouput_boxes[...,2] = ouput_boxes[...,2] * img_width
            ouput_boxes[...,3] = ouput_boxes[...,3] * img_height
            # coco format to yolo format
            truth_boxes = targets[0]['boxes'].numpy().copy()
            truth_boxes[...,:2] = truth_boxes[...,:2] + truth_boxes[...,2:]/2
            iou = bboxes_iou_test(torch.Tensor(ouput_boxes), torch.Tensor(truth_boxes), fmt='yolo')
            print(f"iou of first image = {iou}")
        if len(debug) > 0:
            return
        
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator


def get_args(**kwargs):
    """
    """
    pretrained_detector = '/mnt/wenhao71/workspace/yolov4_acne_torch/pretrained/yolov4.pth'
    cfg = kwargs
    parser = argparse.ArgumentParser(
        description='Train the Model on images and target masks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument(
    #     '-b', '--batch-size',
    #     metavar='B', type=int, nargs='?', default=2,
    #     help='Batch size',
    #     dest='batchsize')
    parser.add_argument(
        '-l', '--learning-rate',
        metavar='LR', type=float, nargs='?', default=0.001,
        help='Learning rate',
        dest='learning_rate')
    parser.add_argument(
        '-f', '--load',
        dest='load', type=str, default=pretrained_detector,
        help='Load model from a .pth file')
    parser.add_argument(
        '-g', '--gpu',
        metavar='G', type=str, default='0',
        help='GPU',
        dest='gpu')
    # `dataset_dir` and `pretrained` already set in cfg_acne04.py
    # parser.add_argument(
    #     '-dir', '--data-dir',
    #     type=str, default=None,
    #     help='dataset dir', dest='dataset_dir')
    # parser.add_argument(
    #     '-pretrained',
    #     type=str, default=None,
    #     help='pretrained yolov4.conv.137')
    parser.add_argument(
        '-classes',
        type=int, default=1,
        help='dataset classes')
    # parser.add_argument(
    #     '-train_label_path',
    #     dest='train_label', type=str, default='train.txt',
    #     help="train label path")
    parser.add_argument(
        '-iou-type', type=str, default='iou',
        help='iou type (iou, giou, diou, ciou)',
        dest='iou_type')
    parser.add_argument(
        '-keep-checkpoint-max', type=int, default=10,
        help='maximum number of checkpoints to keep. If set 0, all checkpoints will be kept',
        dest='keep_checkpoint_max')
    parser.add_argument(
        '-optimizer', type=str, default='adam',
        help='training optimizer',
        dest='TRAIN_OPTIMIZER')
    
    args = vars(parser.parse_args())

    cfg.update(args)
    
    return ED(cfg)


def init_logger(log_file=None, log_dir=None, mode='a', verbose=0):
    """
    """
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = f'log_{_get_date_str()}.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    print(f'log file path: {log_file}')

    logger = logging.getLogger('Yolov4-ACNE04')

    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(log_file)

    if verbose >= 2:
        print("levels of c_handler and f_handler are set DEBUG")
        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    elif verbose >= 1:
        print("level of c_handler is set INFO, level of f_handler is set DEBUG")
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        print("levels of c_handler and f_handler are set WARNING")
        c_handler.setLevel(logging.WARNING)
        f_handler.setLevel(logging.WARNING)
        logger.setLevel(logging.WARNING)

    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def _get_date_str():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M')


if __name__ == "__main__":
    cfg = get_args(**Cfg)
    # os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    if not DAS:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda')
    log_dir = cfg.TRAIN_TENSORBOARD_DIR
    logger = init_logger(log_dir=log_dir)
    logger.info(f"\n{'*'*20}   Start Training   {'*'*20}\n")
    logger.info(f'Using device {device}')
    logger.info(f"Using torch of version {torch.__version__}")
    logger.info(f'with configuration {cfg}')
    print(f"\n{'*'*20}   Start Training   {'*'*20}\n")
    print(f'Using device {device}')
    print(f"Using torch of version {torch.__version__}")
    print(f'with configuration {cfg}')

    model = Yolov4(cfg.pretrained, n_classes=cfg.classes)

    if not DAS and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    if not DAS:
        model.to(device=device)
    else:
        model.cuda()

    try:
        train(
            model=model,
            config=cfg,
            epochs=cfg.TRAIN_EPOCHS,
            device=device,
            logger=logger,
        )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), os.path.join(cfg.checkpoints, 'INTERRUPTED.pth'))
        logger.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


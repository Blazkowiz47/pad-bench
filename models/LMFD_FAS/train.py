from __future__ import print_function, division
import argparse
from logging import getLogger
import os
import random

from tqdm import tqdm
import pandas as pd
import numpy as np
import math
from collections import defaultdict


import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from utils import AvgrageMeter, performances_cross_db
from model.mfad import FAD_HAM_Net
from losses import WeightedFocalLoss

from dataset import StandardWrapper

# This train.py is used for cross-domain training and evaluation
# The evaluation metrics for cross-domain and inter-dataset is different.
# For inter dataset, the threshold is obtained from development set.


def main(args):
    # create log file

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "train" + ".txt")

    # create directory for save trained models
    checkpoint_save_dir = os.path.join(args.log_dir, "checkpoints")
    best_weights_path = os.path.join(checkpoint_save_dir, "best_weights.pth")
    if os.path.isdir(checkpoint_save_dir):
        return

    os.makedirs(checkpoint_save_dir, exist_ok=True)

    # initialize model
    model = FAD_HAM_Net(pretrain=args.pretrain, variant=args.backbone).cuda()

    print("-------------- train ------------------------")
    log_file = open(os.path.join(log_path), "w")
    log = getLogger()
    # load data

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0001
    )
    # two loss function: smooth L1 and focal loss
    smotthl1_criterion = torch.nn.SmoothL1Loss().cuda()
    FL_criterion = WeightedFocalLoss(alpha=0.5, gamma=2).cuda()

    scheduler = ExponentialLR(optimizer, gamma=0.995)
    scaler = GradScaler()

    # parameters for early stopping
    epochs_no_improvement = 0
    max_auc = -1
    min_hter = 1000
    best_weights = None
    best_epoch = -1

    # start training
    for epoch in range(1, args.epochs + 1):
        loss_total = AvgrageMeter()
        loss_1_total = AvgrageMeter()
        loss_2_total = AvgrageMeter()

        ###########################################
        """                train             """
        ###########################################
        model.train()

        # loss weight update
        if epoch >= 5:
            w1 = 100  # 10-75
        else:
            w1 = 1

        wrapper = StandardWrapper(
            {"facedetect": "Face_Detect"}, log, 14, attack=args.attack, rdir=args.rdir
        )
        train_loader = DataLoader(
            wrapper.get_split("train"),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_works,
            pin_memory=True,
            drop_last=True,
        )

        test_loader = DataLoader(
            wrapper.get_split("test"),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_works,
            pin_memory=True,
        )

        progress_bar = tqdm(train_loader)
        for i, (images, labels, map_x) in enumerate(progress_bar):
            progress_bar.set_description("Epoch " + str(epoch))

            images = images.cuda()
            labels = labels.cuda()
            map_x = map_x.cuda()
            model.zero_grad()

            with autocast():
                pred, map_y = model(images)
                pred = torch.squeeze(pred)
                loss_1 = FL_criterion(
                    pred.cuda().float(),
                    labels.cuda().float(),
                )
                loss_2 = smotthl1_criterion(map_y, map_x) * w1
                loss = loss_1 + loss_2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_total.update(loss.data, images.shape[0])
            loss_1_total.update(loss_1.data, images.shape[0])
            loss_2_total.update(loss_2.data, images.shape[0])

            progress_bar.set_postfix(
                loss_1="%.5f" % (loss_1_total.avg),
                loss_2="%.5f" % (loss_2_total.avg),
                loss="%.5f" % (loss_total.avg),
            )

        ###########################################
        """                Test             """
        ###########################################
        print("------------ test -------------------")
        if args.early_stop:
            model.eval()
            # torch.save(model.state_dict(), os.path.join(checkpoint_save_dir, 'epoch_{}.pth'.format(epoch)))

            predictions, gt_labels, video_ids = [], [], []
            vid = 0
            with torch.no_grad():
                for images, labels, _ in tqdm(test_loader):
                    images = images.cuda()
                    labels = labels.cuda()
                    optimizer.zero_grad()

                    pred, _ = model(images)
                    pred = torch.sigmoid(pred)

                    for j in range(images.shape[0]):
                        predictions.append(pred[j].detach().cpu())
                        # mean_pred = torch.mean(map_y[j])
                        gt_labels.append(labels[j].detach().cpu())
                        video_ids.append(str(vid))
                        vid += 1

            # fuse prediction scores (mean value) of all frames for each video
            predictions, gt_labels, _ = compute_video_score(
                video_ids, predictions, gt_labels
            )

            # evaluation
            test_auc, _, _, test_hter = performances_cross_db(predictions, gt_labels)
            if test_auc == math.nan:
                return

            if test_hter < min_hter:
                min_hter = test_hter
                max_auc = test_auc
                epochs_no_improvement = 0
                best_weights = model.state_dict()
                best_epoch = epoch
            else:
                epochs_no_improvement += 1

            if epochs_no_improvement >= args.patience:
                print(
                    f"EARLY STOPPING at {best_epoch}: {min_hter}, {max_auc}"
                )  # , {max_auc}
                break

            tqdm.write("Test: auc=%.4f, hter= %.4f \n" % (test_auc, test_hter))
            log_file.write("Test: auc=%.4f, hter= %.4f \n" % (test_auc, test_hter))
            log_file.flush()

    torch.save(best_weights, best_weights_path)


def compute_video_score(video_ids, predictions, labels):
    predictions_dict, labels_dict = defaultdict(list), defaultdict(list)

    for i in range(len(video_ids)):
        video_key = video_ids[i]
        predictions_dict[video_key].append(predictions[i])
        labels_dict[video_key].append(labels[i])

    new_predictions, new_labels, new_video_ids = [], [], []

    for video_indx in list(set(video_ids)):
        new_video_ids.append(video_indx)
        scores = np.mean(predictions_dict[video_indx])
        label = labels_dict[video_indx][0]
        new_predictions.append(scores)
        new_labels.append(label)

    return new_predictions, new_labels, new_video_ids


if __name__ == "__main__":
    torch.cuda.empty_cache()
    cudnn.benchmark = True

    if torch.cuda.is_available():
        print("GPU is available")
        torch.cuda.manual_seed(0)
    else:
        print("GPU is not available")
        torch.manual_seed(0)

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        choices=["resnet101", "resnet50", "resnet34"],
    )
    parser.add_argument(
        "--pretrain",
        default=True,
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
    )
    parser.add_argument(
        "--early_stop",
        default=True,
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
    )

    parser.add_argument("--lr", default=0.001, type=float)

    parser.add_argument("--epochs", default=100, type=int, help="maximum epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="train batch size")
    parser.add_argument("--num_works", default=8, type=int, help="train batch size")
    parser.add_argument("--patience", default=15, type=int)
    parser.add_argument("--log_dir", type=str, default="training_logs/")

    args = parser.parse_args()
    ATTACKS = [
        "display",
        "print",
        "hard_plastic",
        "latex_mask",
        "paper_mask",
        "silicone_mask",
        "soft_plastic",
        "wrap",
    ]

    for iphone in ["iPhone12", "iPhone11"]:
        for attack in ATTACKS:
            rdir = f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/2D_Face_Databases_PAD/{iphone}/Data_Split/"
            edir = f"../../tmp/LMFD_FAS/{iphone}/{attack}"
            args.attack = attack
            args.rdir = rdir
            args.log_dir = edir
            main(args)

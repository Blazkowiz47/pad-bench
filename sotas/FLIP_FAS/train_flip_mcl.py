import argparse
import os
import time
from datetime import datetime
from logging import getLogger
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import config_A
from fas import flip_mcl
from utils.dataset import StandardWrapper
from utils.evaluate import eval
from utils.utils import (
    AverageMeter,
    Logger,
    accuracy,
    save_checkpoint,
    time_to_str,
)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda"


def train(config):
    wrapper = StandardWrapper(
        {"batch_size": 6, "num_workers": 4, "facedetect": "Face_Detect"},
        log=getLogger(),
        rdir=config.rdir,
        attack=config.attack,
    )

    src1_train_dataloader_fake, src1_train_dataloader_real = wrapper.get_split("train")
    test_dataloader = wrapper.get_split("test")

    best_model_ACC = 0.0
    best_model_HTER = 1.0
    best_model_ACER = 1.0
    best_model_AUC = 0.0
    best_TPR_FPR = 0.0

    valid_args = [np.inf, 0, 0, 0, 0, 0, 0, 0]

    loss_simclr = AverageMeter()
    loss_l2_euclid = AverageMeter()
    loss_total = AverageMeter()
    loss_classifier = AverageMeter()
    classifer_top1 = AverageMeter()

    log = Logger()
    log.write(
        "\n----------------------------------------------- [START %s] %s\n\n"
        % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "-" * 51)
    )
    log.write("** start training target model! **\n")
    log.write(
        "--------|------------- VALID -------------|--- classifier ---|----------------SimCLR loss-------------|------ Current Best ------|--------------|\n"
    )
    log.write(
        "  iter  |   loss   top-1   HTER    AUC    |   loss   top-1   |   SimCLR-loss   l2-loss   total-loss   |   top-1   HTER    AUC    |    time      |\n"
    )
    log.write(
        "------------------------------------------------------------------------------------------------------------------------------------------------|\n"
    )
    start = timer()
    criterion = {"softmax": nn.CrossEntropyLoss().cuda()}

    net1 = flip_mcl(
        in_dim=512, ssl_mlp_dim=4096, ssl_emb_dim=256
    ).to(
        device
    )  # ssl applied to image, and euclidean distance applied to image and text cosine similarity

    # Fine-tune all the layers
    for name, param in net1.named_parameters():
        param.requires_grad = True

    # Load if checkpoint is provided
    if config.checkpoint:
        ckpt = torch.load(config.checkpoint)
        net1.load_state_dict(ckpt["state_dict"])
        epoch = ckpt["epoch"]
        iter_num_start = epoch * 100
        print(f"Loaded checkpoint from epoch {epoch} at iteration : {iter_num_start}")
    else:
        epoch = 1
        iter_num_start = 0
        print(f"Starting training from epoch {epoch} at iteration : {iter_num_start}")

    iter_per_epoch = 100

    optimizer_dict = [
        {
            "params": filter(lambda p: p.requires_grad, net1.parameters()),
            "lr": 0.000001,
        },
    ]

    optimizer1 = optim.Adam(optimizer_dict, lr=0.000001, weight_decay=0.000001)

    src1_train_iter_real = iter(src1_train_dataloader_real)
    src1_iter_per_epoch_real = len(src1_train_iter_real)

    src1_train_iter_fake = iter(src1_train_dataloader_fake)
    src1_iter_per_epoch_fake = len(src1_train_iter_fake)

    for iter_num in range(iter_num_start, 4000 + 1):
        if iter_num % src1_iter_per_epoch_real == 0:
            src1_train_iter_real = iter(src1_train_dataloader_real)

        if iter_num % src1_iter_per_epoch_fake == 0:
            src1_train_iter_fake = iter(src1_train_dataloader_fake)

        if iter_num != 0 and iter_num % iter_per_epoch == 0:
            epoch = epoch + 1

        net1.train(True)
        optimizer1.zero_grad()
        ######### data prepare #########
        src1_img_real, src1_img_real_view_1, src1_img_real_view_2, src1_label_real = (
            src1_train_iter_real.next()
        )
        src1_img_real = src1_img_real.cuda()
        src1_label_real = src1_label_real.cuda()
        src1_img_real_view_1 = src1_img_real_view_1.cuda()
        src1_img_real_view_2 = src1_img_real_view_2.cuda()
        input1_real_shape = src1_img_real.shape[0]

        src1_img_fake, src1_img_fake_view_1, src1_img_fake_view_2, src1_label_fake = (
            src1_train_iter_fake.next()
        )
        src1_img_fake = src1_img_fake.cuda()
        src1_label_fake = src1_label_fake.cuda()
        src1_img_fake_view_1 = src1_img_fake_view_1.cuda()
        src1_img_fake_view_2 = src1_img_fake_view_2.cuda()
        input1_fake_shape = src1_img_fake.shape[0]

        if config.tgt_data in ["cefa", "surf", "wmca"]:
            input_data = torch.cat(
                [
                    src1_img_real,
                    src1_img_fake,
                ],
                dim=0,
            )

            input_data_view_1 = torch.cat(
                [
                    src1_img_real_view_1,
                    src1_img_fake_view_1,
                ],
                dim=0,
            )

            input_data_view_2 = torch.cat(
                [
                    src1_img_real_view_2,
                    src1_img_fake_view_2,
                ],
                dim=0,
            )

        else:
            input_data = torch.cat(
                [
                    src1_img_real,
                    src1_img_fake,
                ],
                dim=0,
            )

            input_data_view_1 = torch.cat(
                [
                    src1_img_real_view_1,
                    src1_img_fake_view_1,
                ],
                dim=0,
            )

            input_data_view_2 = torch.cat(
                [
                    src1_img_real_view_2,
                    src1_img_fake_view_2,
                ],
                dim=0,
            )

        if config.tgt_data in ["cefa", "surf", "wmca"]:
            source_label = torch.cat(
                [
                    src1_label_real.fill_(1),
                    src1_label_fake.fill_(0),
                ],
                dim=0,
            )
        else:
            source_label = torch.cat(
                [
                    src1_label_real.fill_(1),
                    src1_label_fake.fill_(0),
                ],
                dim=0,
            )

        ######### forward #########
        classifier_label_out, logits_ssl, labels_ssl, l2_euclid_loss = net1(
            input_data, input_data_view_1, input_data_view_2, source_label, True
        )  # ce on I-T, SSL for image and l2 loss for image-view-text dot product
        cls_loss = criterion["softmax"](
            classifier_label_out.narrow(0, 0, input_data.size(0)), source_label
        )
        sim_loss = criterion["softmax"](logits_ssl, labels_ssl)

        fac = 1.0
        total_loss = cls_loss + fac * sim_loss + fac * l2_euclid_loss

        total_loss.backward()
        optimizer1.step()
        optimizer1.zero_grad()

        loss_classifier.update(cls_loss.item())
        loss_l2_euclid.update(l2_euclid_loss.item())
        loss_simclr.update(sim_loss.item())
        loss_total.update(total_loss.item())

        acc = accuracy(
            classifier_label_out.narrow(0, 0, input_data.size(0)),
            source_label,
            topk=(1,),
        )
        classifer_top1.update(acc[0])

        if iter_num != 0 and (iter_num + 1) % (iter_per_epoch) == 0:
            valid_args = eval(test_dataloader, net1, True)
            # judge model according to HTER
            is_best = valid_args[3] <= best_model_HTER
            best_model_HTER = min(valid_args[3], best_model_HTER)
            threshold = valid_args[5]
            if valid_args[3] <= best_model_HTER:
                best_model_ACC = valid_args[6]
                best_model_AUC = valid_args[4]
                best_TPR_FPR = valid_args[-1]

            save_list = [
                epoch,
                valid_args,
                best_model_HTER,
                best_model_ACC,
                best_model_ACER,
                threshold,
            ]

            save_checkpoint(
                save_list,
                is_best,
                net1,
                os.path.join(
                    config.op_dir,
                    config.attack
                    + f"_flip_mcl_checkpoint_run_{str(config.run)}.pth.tar",
                ),
            )

            print("\r", end="", flush=True)
            log.write(
                "  %4.1f  |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |     %6.3f     %6.3f     %6.3f     |  %6.3f  %6.3f  %6.3f  | %s   %s"
                % (
                    (iter_num + 1) / iter_per_epoch,
                    valid_args[0],
                    valid_args[6],
                    valid_args[3] * 100,
                    valid_args[4] * 100,
                    loss_classifier.avg,
                    classifer_top1.avg,
                    loss_simclr.avg,
                    loss_l2_euclid.avg,
                    loss_total.avg,
                    float(best_model_ACC),
                    float(best_model_HTER * 100),
                    float(best_model_AUC * 100),
                    time_to_str(timer() - start, "min"),
                    0,
                )
            )
            log.write("\n")

            time.sleep(0.01)

    return best_model_HTER * 100.0, best_model_AUC * 100.0, best_TPR_FPR * 100.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--op_dir", type=str, default=None)
    parser.add_argument("--report_logger_path", type=str, default=None)
    parser.add_argument("--rdir", type=str, default=None)
    parser.add_argument("--attack", type=str, default=None)

    args = parser.parse_args()
    config = config_A

    config.rdir = args.rdir
    config.attack = args.attack

    for attr in dir(config):
        if attr.find("__") == -1:
            print("%s = %r" % (attr, getattr(config, attr)))

    config.op_dir = str(args.op_dir)

    with open(args.report_logger_path, "w") as f:
        f.write("Run, HTER, AUC, TPR@FPR=1%\n")
        hter_avg = []
        auc_avg = []
        tpr_fpr_avg = []

        for i in range(5):
            # To reproduce results
            torch.manual_seed(i)
            np.random.seed(i)

            config.run = i
            config.checkpoint = args.ckpt
            hter, auc, tpr_fpr = train(config)

            hter_avg.append(hter)
            auc_avg.append(auc)
            tpr_fpr_avg.append(tpr_fpr)

            f.write(f"{i},{hter},{auc},{tpr_fpr}\n")

        hter_mean = np.mean(hter_avg)
        auc_mean = np.mean(auc_avg)
        tpr_fpr_mean = np.mean(tpr_fpr_avg)
        f.write(f"Mean,{hter_mean},{auc_mean},{tpr_fpr_mean}\n")

        hter_std = np.std(hter_avg)
        auc_std = np.std(auc_avg)
        tpr_fpr_std = np.std(tpr_fpr_avg)
        f.write(f"Std dev,{hter_std},{auc_std},{tpr_fpr_std}\n")

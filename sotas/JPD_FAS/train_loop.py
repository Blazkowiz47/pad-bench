import os

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

if __name__ == "__main__":
    args = []
    port = 12345
    for iphone in ["iPhone11", "iPhone12"]:
        for attack in reversed(ATTACKS):
            rdir = f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/2D_Face_Databases_PAD/{iphone}/Data_Split/"
            edir = f"../../tmp/JPD_FAS/{iphone}/{attack}"
            cmd = f"python -m torch.distributed.run --nproc_per_node=1 --master_port={port} \
                  train_dist.py \
                  --attack {attack} \
                  --rdir {rdir} \
                  --protocol 'p2.2' \
                  --live_weight 1.0 \
                  --train_root 'xxx/cvpr2024/data' \
                  --train_list 'xxx/cvpr2024/data/p2.2/train_dev_label.txt' \
                  --val_root   'xxx/cvpr2024/data' \
                  --val_list   'xxx/cvpr2024/data/p2.2/dev_label.txt' \
                  --cos \
                  --syncbn \
                  --arch resnet50 \
                  --num_classes 2 \
                  --input_size 224 \
                  --batch_size 32 \
                  --workers 8 \
                  --optimizer AdamW \
                  --learning_rate 0.001 \
                  --weight_decay 0.0005 \
                  --epochs 20 \
                  --print_freq 1 \
                  --save_freq 1 \
                  --saved_model_dir '{edir}' \
                "
            port += 1
            os.system(cmd)

class DefaultConfigs(object):
    seed = 666
    # SGD
    weight_decay = 1e-6
    momentum = 0.9
    # learning rate
    init_lr = 1e-4
    lr_epoch_1 = 0
    lr_epoch_2 = 150
    # model
    pretrained = True
    model = "dgua_fas"
    # training parameters
    gpus = "0"
    batch_size = 4
    norm_flag = True
    max_iter = 4000
    lambda_triplet = 1
    lambda_adreal = 0.5
    # test model name
    tgt_best_model_name = "best_model.pth.tar"  #'model_best_0.08_29.pth.tar'
    # source data information
    src1_data = "celebaspoof"
    src1_train_num_frames = 1
    src2_data = "trainingPro"
    src2_train_num_frames = 1
    src3_data = "oulu"
    src3_train_num_frames = 1
    src4_data = "unidata"
    src4_train_num_frames = 1
    # target data information
    tgt_data = "msu_mfsd"
    tgt_test_num_frames = 3
    # paths information
    checkpoint_path = "./test_checkpoint/" + model + "_1/DGFANet/"
    best_model_path = "./test_checkpoint/" + model + "_1/best_model/"
    logs = "./logs/"


config = DefaultConfigs()

__all__ = ["config"]

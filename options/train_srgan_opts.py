"""
The options to train Super-Resolution Wasserstein GAN (SR WGAN).
"""
import argparse


def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--channels", type=int, default=3,
        help="The number of image channels."
    )
    parser.add_argument(
        "--scale_factor", type=int, default=4,
        choices=[2, 3, 4, 8],
        help="Upscale factor for super-resolution."
    )

    parser.add_argument(
        "--st_epoch", type=int, default=1,
        help="Epoch to start the formal training."
    )
    parser.add_argument(
        "--ed_epoch", type=int, default=10,
        help="Epoch to end the formal training."
    )
    parser.add_argument(
        "--n_critic", type=int, default=5,
        help="The number of training steps for discriminator per iter."
    )

    parser.add_argument(
        "--sample_interval", type=int, default=100,
        help="The interval between saving training results (LR/SR/HR images)."
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=-1,
        help="The interval between two model checkpoints."
    )

    parser.add_argument(
        "--output_path", type=str, default="./output/",
        help="The path to save outputs."
    )

    # region -------- Models --------
    parser.add_argument(
        "--G", type=str, default=None,
        choices=["RRDBNet", "MSRResNet", "EnhanceNet"],
        help="The choice of generator."
    )
    parser.add_argument(
        "--G_model", type=str, default=None,
        help="The generator model file to be loaded."
    )

    parser.add_argument(
        "--D", type=str, default=None,
        choices=["VGG", "PatchVGG",
                 "DeeperVGG", "ResNet18"],
        help="The choice of discriminator."
    )
    parser.add_argument(
        "--D_model", type=str, default=None,
        help="The discriminator model file to be loaded."
    )
    # endregion

    # region -------- Training Set --------
    parser.add_argument(
        "--train_set", type=str, default=None,
        help="The path of training set."
             "For training, you must give a path, it cannot be None."
    )
    parser.add_argument(
        "--train_set_form", type=str, default=None,
        choices=["Pair", "Single"],
        help="The form of the images in training set, support 'Pair' or 'Single'."
    )
    parser.add_argument(
        "--train_set_guider", type=str, default=None,
        help="The guide file of training set. "
             "It will help the dataloader locate the images."
    )
    parser.add_argument(
        "--lr_size", type=int, default=48,
        help="The size of LR patches."
    )
    parser.add_argument(
        "--augment_data", type=int, default=1,
        choices=[0, 1],
        help="If conduct data augmentation when load training set images."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="The size of each mini-batch during training."
    )
    # endregion

    # region -------- SR Valid Set --------
    parser.add_argument(
        "--sr_valid_set", type=str, default=None,
        help="The path of validation set."
    )
    parser.add_argument(
        "--sr_valid_set_form", type=str, default=None,
        choices=["SR_Pair", "SR_Single"],
        help="The form of the images in validation set, support 'SR_Pair' and 'SR_Single'."
    )
    parser.add_argument(
        "--sr_valid_set_guider", type=str, default=None,
        help="The guide file of validation set. "
             "It will help the dataloader locate the images."
    )
    parser.add_argument(
        "--sr_valid_interval", type=int, default=-1,
        help="The interval between validations."
    )
    parser.add_argument(
        "--sr_valid_save_images", type=int, nargs='*', default=None,
        help="The indices of images which will be saved in validation. "
             "Default value None means to save all."
    )
    # endregion

    # region -------- IQA Valid Set --------
    parser.add_argument(
        "--iqa_valid_set", type=str, default=None,
        help="The path of validation set."
    )
    parser.add_argument(
        "--iqa_valid_set_form", type=str, default="IQA",
        choices=["IQA"],
        help="Now only have one form."
    )
    parser.add_argument(
        "--iqa_valid_set_guider", type=str, default=None,
        help="The guide file of validation set."
    )
    parser.add_argument(
        "--iqa_valid_interval", type=int, default=-1,
        help="The interval between validations."
    )
    # endregion

    parser.add_argument(
        "--n_cpu", type=int, default=2,
        help="The number of cpu threads to be used during batch generation."
    )

    # region -------- Optimizer --------
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Adam: learning rate."
    )
    parser.add_argument(
        "--b1", type=float, default=0.9,
        help="Adam: decay of first order momentum of gradient."
    )
    parser.add_argument(
        "--b2", type=float, default=0.999,
        help="Adam: decay of first order momentum of gradient."
    )

    parser.add_argument(
        "--adjust_lr", type=int, default=1,
        choices=[0, 1],
        help="If use learning rate adjustment in formal training."
    )
    parser.add_argument(
        "--decay_iters", type=int, nargs='*', default=[50000, 100000, 200000, 300000],
        help="Decay the learning rate when reach these iterations."
    )

    parser.add_argument(
        "--G_optim", type=str, default=None,
        help="The G optimizer state file to be loaded."
    )
    parser.add_argument(
        "--D_optim", type=str, default=None,
        help="The D optimizer state file to be loaded."
    )
    parser.add_argument(
        "--G_sched", type=str, default=None,
        help="The G scheduler state file to be loaded."
    )
    parser.add_argument(
        "--D_sched", type=str, default=None,
        help="The D scheduler state file to be loaded."
    )
    # endregion

    # region -------- Pre-train --------
    parser.add_argument(
        "--pretrain_epochs", type=int, default=10,
        help="The number of epochs of pre-training. Effective only when st_epoch == 1."
    )
    parser.add_argument(
        "--pretrain_lr", type=float, default=2e-4,
        help="The learning rate in pre-training."
    )
    parser.add_argument(
        "--pretrain_adjust_lr", type=int, default=1,
        choices=[0, 1],
        help="If use learning rate adjustment in pre-training."
    )
    parser.add_argument(
        "--pretrain_decay_step", type=int, default=2000,
        help="Decay the learning rate every n mini-batches."
    )
    parser.add_argument(
        "--pretrain_chkpt_intrvl", type=int, default=-1,
        help="The interval between model checkpoints during pre-training."
    )

    parser.add_argument(
        "--pretrain_optim", type=str, default=None,
        help="The optimizer state file to be loaded during pre-training."
    )
    parser.add_argument(
        "--pretrain_sched", type=str, default=None,
        help="The scheduler state file to be loaded during pre-training."
    )
    # endregion

    # region -------- Loss --------
    parser.add_argument(
        "--w_adv_loss", type=float, default=5e-3,
        help="The weight of adversarial loss."
    )
    parser.add_argument(
        "--w_percept_loss", type=float, default=1.0,
        help="The weight of perceptual loss."
    )
    parser.add_argument(
        "--w_content_loss", type=float, default=1.0,
        help="The weight of content loss."
    )

    parser.add_argument(
        "--content_criterion", type=str, default="L1",
        choices=["L1", "L2"],
        help="The criterion for content loss."
    )

    parser.add_argument(
        "--percept_criterion", type=str, default="L1",
        choices=["L1", "L2"],
        help="The criterion for perceptual loss."
    )
    parser.add_argument(
        "--percept_vgg_model", type=str, default=None,
        help="For feature extraction in perceptual loss, the pretrained VGG model file to be loaded."
    )
    parser.add_argument(
        "--percept_last_layer", type=int, default=35,
        help="The last layer in VGG for feature extraction."
    )
    # endregion

    return parser.parse_args()


if __name__ == '__main__':
    pass

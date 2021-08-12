"""
The options to test discriminator on IQA datasets.
"""
import argparse


def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--channels", type=int, default=3,
        help="The number of image channels."
    )

    # region -------- Model --------
    parser.add_argument(
        "--D", type=str, default=None,
        choices=["VGG", "PatchVGG",
                 "VGG_old", "PatchVGG_old",  # Now these old versions have been abandoned.
                 "DeeperVGG", "ResNet18"],
        help="The choice of discriminator."
    )
    parser.add_argument(
        "--D_model", type=str, default=None,
        help="The discriminator model file to load."
    )
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="The epoch number of the trained discriminator model, used to check."
    )

    parser.add_argument(
        "--fusion_mode", type=str, default="naive",
        choices=["naive", "Naive",
                 "SOD BASNet + SR residual",
                 "SOD U2Net + SR residual"],
        help="The fusion mode of PatchGAN discriminator output, "
             "it will be useful only when use PatchGAN discriminator."
             "Here naive = simply average the output of PatchGAN discriminator."
    )

    parser.add_argument(
        "--coef_sod", type=float, default=None,
        help="The coefficient of SOD weight matrix."
             "Only when use SOD weight mat + IR weight mat."
    )
    parser.add_argument(
        "--coef_res", type=float, default=None,
        help="The coefficient of image residual weight matrix."
             "Only when use SOD weight mat + IR weight mat."
    )
    parser.add_argument(
        "--coef_avg", type=float, default=None,
        help="The coefficient of all-one weight matrix."
             "Only when use SOD weight mat + IR weight mat."
    )
    parser.add_argument(
        "--norm_type", type=str, default=None,
        choices=["mean", "set_sum_1"],
        help="The norm type of total weight matrix."
             "Effective only when use SOD weight matrix + IR weight matrix."
    )
    # endregion

    parser.add_argument(
        "--n_cpu", type=int, default=2,
        help="The number of cpu threads to use during batch generation."
    )

    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="The batch size of one test iteration."
    )

    # region -------- Test Set --------
    parser.add_argument(
        "--test_set", type=str, default=None,
        help="The path of test set."
    )
    parser.add_argument(
        "--test_set_guider", type=str, default=None,
        help="The guide file of test set."
    )
    # endregion

    # region -------- Results --------
    parser.add_argument(
        "--save_pred", type=int, default=0,
        choices=[0, 1],
        help="If save the predicted scores."
    )

    parser.add_argument(
        "--res_file", type=str, default=None,
        help="The file to save pred results."
    )
    # endregion

    return parser.parse_args()


if __name__ == '__main__':
    pass

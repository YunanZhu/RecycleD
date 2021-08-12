"""
Used to test discriminator on the 2afc->val->superres part in BAPPS dataset (proposed in LPIPS).
Zhang et al. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR 2018.

About the 2afc->val->superres part of BAPPS dataset:
    https://github.com/richzhang/PerceptualSimilarity#c-about-the-dataset
    2AFC Evaluators were given a patch triplet (1 reference + 2 distorted).
    val/superres: [10.9k triplets].
    Each 2AFC subdirectory contains the following folders:
        ref: original reference patches.
        p0,p1: two distorted patches.
        judge: human judgments - 0 if all preferred p0, 1 if all humans preferred p1.
"""
import torch
import torch.nn as nn

from models import srgan, optim_predictor, access
from dataloader import test_set
from metrics import binary_metrics


def main(opts):
    print(opts)

    # region -------- CUDA State --------
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"CUDA state: {'available' if use_cuda else 'unavailable'}.")  # To check if CUDA is available.
    # endregion

    # region -------- Initialize Discriminator --------
    D = srgan.define_discriminator(
        opts.D,  # The choice of discriminator.
        in_channels=opts.channels,
        count_params=True  # Count and print the number of parameters of discriminator.
    )
    D.to(device)

    if opts.D_model is not None:
        access.load_model(D, opts.epoch, opts.D_model)
    else:
        raise ValueError("The values of opts.D_model cannot be None.")

    # Try to optimize D to get better quality assess ability by using weight matrix.
    if ("patch" in opts.D.lower()) and (opts.fusion_mode.lower() != "naive"):
        predictor = optim_predictor.define_predictor(
            d_choice=opts.D,  # The choice of discriminator.
            d_model=D,  # The PatchGAN discriminator model.
            in_channels=opts.channels,
            fusion_mode=opts.fusion_mode,
            coef_sod=opts.coef_sod,
            coef_res=opts.coef_res,
            coef_avg=opts.coef_avg,
            norm_type=opts.norm_type
        )
    else:
        predictor = D

    predictor = nn.DataParallel(predictor)
    predictor.to(device)
    # endregion

    # region -------- Dataloader --------
    test_loader = test_set.get_dataloader(
        choice="BappsSuperres",
        path=opts.test_set,
        guide_file=opts.test_set_guider,
        scale_factor=-1,  # Upscale factor is useless here.
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.n_cpu
    )
    print(f"The batch size: {test_loader.batch_size}.")
    print(f"{len(test_loader)} batches in test set.")
    # endregion

    # region -------- Test --------
    test_res = {"pred_prefer": [], "GT_prefer": []}
    with torch.no_grad():
        predictor.eval()
        for i, samples in enumerate(test_loader):
            p0, p1 = samples["p0"].to(device), samples["p1"].to(device)
            gt_prefer = samples["prefer"]

            p0_score, p1_score = predictor(p0), predictor(p1)
            pred_prefer = (p0_score <= p1_score).type(torch.int)

            pred_prefer = pred_prefer.view(pred_prefer.size(0))  # Reshape to [N].
            assert len(pred_prefer.size()) == 1 and len(gt_prefer.size()) == 1  # Check the tensor shapes.

            test_res["pred_prefer"].extend(pred_prefer.tolist())
            test_res["GT_prefer"].extend(gt_prefer.tolist())

    # Calculate accuracy, precision and recall.
    assert len(test_res["pred_prefer"]) == len(test_res["GT_prefer"])
    accuracy = binary_metrics.accuracy(test_res["pred_prefer"], test_res["GT_prefer"])
    precision = binary_metrics.precision(test_res["pred_prefer"], test_res["GT_prefer"])
    recall = binary_metrics.recall(test_res["pred_prefer"], test_res["GT_prefer"])
    print(f"Test results: "
          f"accuracy = {accuracy:.6f}, "
          f"precision = {precision:.6f}, "
          f"recall = {recall:.6f}.")

    # Save predictions.
    if bool(opts.save_pred):
        with open(opts.res_file, 'w') as test_logger:
            print("pred_prefer,GT_prefer", file=test_logger)
            for i in range(len(test_res["GT_prefer"])):
                pred_prefer = test_res["pred_prefer"][i]
                gt_prefer = test_res["GT_prefer"][i]
                print(f"{pred_prefer},{gt_prefer}", file=test_logger)
            print(f"accuracy,{accuracy}", file=test_logger)
            print(f"precision,{precision}", file=test_logger)
            print(f"recall,{recall}", file=test_logger)

    # endregion


if __name__ == '__main__':
    from options import test_iqa_opts  # Use the same options as `test_iqa.py`.

    main(test_iqa_opts.get_options())

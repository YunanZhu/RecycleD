"""
Used to test discriminator on the IQA dataset.

Support:
    1. Test ImageGAN discriminator on IQA dataset.
    2. Test PatchGAN discriminator on IQA dataset.
    3. Test optimized PatchGAN discriminator (optimized by using weight matrix) on IQA dataset.
"""
import time

import torch
import torch.nn as nn

from models import srgan, optim_predictor, access
from dataloader import test_set
from metrics import correlation as corr


def main(opts):
    print(opts)

    # region -------- CUDA State --------
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"CUDA state: {'available' if use_cuda else 'unavailable'}.")  # To check if CUDA is available.
    # endregion

    # region -------- Initialize Discriminator (predictor) --------
    D = srgan.define_discriminator(
        opts.D,
        in_channels=opts.channels,
        count_params=True
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
        choice="IQA",
        path=opts.test_set,
        guide_file=opts.test_set_guider,
        scale_factor=-1,  # Upscale factor is useless in IQA test.
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.n_cpu
    )
    print(f"There are {len(test_loader)} batches in IQA test set. "
          f"The batch size: {test_loader.batch_size}.")
    # endregion

    # region -------- Test --------
    test_res = {"image_name": [], "MOS": [], "pred_score": []}
    with torch.no_grad():
        predictor.eval()
        for i, samples in enumerate(test_loader):
            image = samples["image"].to(device)  # The tensor with shape = [N, C, H, W].
            mos = samples["MOS"]  # The tensor with shape = [N].
            name = samples["image_name"]  # The list with shape = [N].

            pred_score = predictor(image)

            mos = mos.view(mos.size(0))  # Reshape to [N].
            pred_score = pred_score.view(pred_score.size(0))  # Reshape to [N].
            assert len(mos.size()) == 1 and len(pred_score.size()) == 1  # Check the tensor shapes.

            test_res["MOS"].extend(mos.tolist())
            test_res["pred_score"].extend(pred_score.tolist())
            test_res["image_name"].extend(name)

    # Calculate PLCC, SRCC and KRCC.
    assert len(test_res["MOS"]) == len(test_res["pred_score"]) == len(test_res["image_name"])
    plcc = corr.plcc(test_res["pred_score"], test_res["MOS"])
    fitted_plcc = corr.fitted_plcc(test_res["pred_score"], test_res["MOS"])
    srcc = corr.srcc(test_res["pred_score"], test_res["MOS"])
    krcc = corr.krcc(test_res["pred_score"], test_res["MOS"])
    print(f"PLCC, fitted_PLCC, SRCC, KRCC:")
    print(f"{plcc:.8f},{fitted_plcc:.8f},{srcc:.8f},{krcc:.8f}")

    # Save predictions.
    if bool(opts.save_pred):
        with open(opts.res_file, 'w') as iqa_test_logger:
            print("image_name,mos,pred", file=iqa_test_logger)
            for i in range(len(test_res["image_name"])):
                img = test_res["image_name"][i]
                mos = test_res["MOS"][i]
                pred = test_res["pred_score"][i]
                print(f"{img},{mos},{pred}", file=iqa_test_logger)
            print(f"PLCC,fitted_PLCC,SRCC,KRCC", file=iqa_test_logger)
            print(f"{plcc:.12f},{fitted_plcc:.12f},{srcc:.12f},{krcc:.12f}", file=iqa_test_logger)

    # endregion


if __name__ == '__main__':
    from options import test_iqa_opts

    main(test_iqa_opts.get_options())

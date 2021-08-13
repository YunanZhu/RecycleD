"""
Train Super-Resolution Wasserstein GAN (SR WGAN).
"""
import os
import math
from datetime import datetime

import numpy as np

import torch
from torch import nn
from torch.nn import functional as nn_f
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from tensorboardX import SummaryWriter

from models import srgan, access, losses
from models.archs import arch_utils
from dataloader import train_set, test_set
from metrics import psnr_ssim, correlation


def main(opts):
    print(opts)

    # region -------- Path --------
    SR_TRAIN_RESULTS_PATH = opts.output_path + "/sr_train_results/"
    os.makedirs(SR_TRAIN_RESULTS_PATH, exist_ok=True)

    SR_VALID_RESULTS_PATH = opts.output_path + "/sr_valid_results/"
    os.makedirs(SR_VALID_RESULTS_PATH, exist_ok=True)

    IQA_VALID_RESULTS_PATH = opts.output_path + "/iqa_valid_results/"
    os.makedirs(IQA_VALID_RESULTS_PATH, exist_ok=True)

    SAVE_MODELS_PATH = opts.output_path + "/save_models/"
    os.makedirs(SAVE_MODELS_PATH, exist_ok=True)
    # endregion

    # region -------- CUDA State --------
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"CUDA state: {'available' if use_cuda else 'unavailable'}.")  # To check if CUDA is available.
    # endregion

    # region -------- Initialize Generator & Discriminator --------
    G = srgan.define_generator(
        opts.G,  # To control the generator architecture type.
        in_channels=opts.channels,
        out_channels=opts.channels,
        upscale_factor=opts.scale_factor,
        count_params=True  # Set true to show the number of parameters.
    )
    G = nn.DataParallel(G)
    G.to(device)
    if opts.G_model is not None:
        access.load_model(G, opts.st_epoch - 1, opts.G_model)

    D = srgan.define_discriminator(
        opts.D,  # To control the discriminator architecture type.
        in_channels=opts.channels,
        count_params=True  # Set true to show the number of parameters.
    )
    D = nn.DataParallel(D)
    D.to(device)
    if opts.D_model is not None:
        access.load_model(D, opts.st_epoch - 1, opts.D_model)
    # endregion

    # region -------- Losses --------
    if opts.content_criterion == "L1":
        content_loss = nn.L1Loss()
    elif opts.content_criterion == "L2":
        content_loss = nn.MSELoss()
    else:
        raise NotImplementedError(f"Cannot recognized content criterion {opts.content_criterion}.")
    content_loss.to(device)

    percept_loss = losses.PerceptualLoss(
        opts.percept_vgg_model,
        last_layer=opts.percept_last_layer,
        criterion_type=opts.percept_criterion,
        loss_weight=1.0
    )
    percept_loss.to(device)
    # endregion

    # region -------- Optimizers --------
    optimizer_G = torch.optim.Adam(G.parameters(), lr=opts.lr, betas=(opts.b1, opts.b2))
    scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer_G, opts.decay_iters, 0.5)
    if opts.st_epoch > 1:
        if opts.G_optim is not None:
            access.load_model(optimizer_G, opts.st_epoch - 1, opts.G_optim)
        if opts.G_sched is not None:
            access.load_model(scheduler_G, opts.st_epoch - 1, opts.G_sched)

    optimizer_D = torch.optim.Adam(D.parameters(), lr=opts.lr, betas=(opts.b1, opts.b2))
    scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer_D, opts.decay_iters, 0.5)
    if opts.st_epoch > 1:
        if opts.D_optim is not None:
            access.load_model(optimizer_D, opts.st_epoch - 1, opts.D_optim)
        if opts.D_sched is not None:
            access.load_model(scheduler_D, opts.st_epoch - 1, opts.D_sched)
    # endregion

    # region -------- Dataloader --------
    train_loader = train_set.get_dataloader(
        choice=opts.train_set_form,
        path=opts.train_set,
        guide_file=opts.train_set_guider,
        lr_patch_size=opts.lr_size,
        scale_factor=opts.scale_factor,
        use_data_augmentation=bool(opts.augment_data),
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.n_cpu
    )
    print(f"There are {len(train_loader)} batches in training set. "
          f"The batch size: {train_loader.batch_size}.")

    if opts.sr_valid_interval >= 1:
        sr_valid_loader = test_set.get_dataloader(
            choice=opts.sr_valid_set_form,
            path=opts.sr_valid_set,
            guide_file=opts.sr_valid_set_guider,
            scale_factor=opts.scale_factor,
            num_workers=opts.n_cpu
        )
        print(f"There are {len(sr_valid_loader)} batches in SR validation set. "
              f"The batch size: {sr_valid_loader.batch_size}.")
    else:
        sr_valid_loader = None
        print("No SR validation.")

    if opts.iqa_valid_interval >= 1:
        iqa_valid_loader = test_set.get_dataloader(
            choice=opts.iqa_valid_set_form,
            path=opts.iqa_valid_set,
            guide_file=opts.iqa_valid_set_guider,
            scale_factor=0,  # Upscale factor is useless in IQA validation.
            num_workers=opts.n_cpu
        )
        print(f"There are {len(iqa_valid_loader)} batches in IQA validation set. "
              f"The batch size: {iqa_valid_loader.batch_size}.")
    else:
        iqa_valid_loader = None
        print("No IQA validation set.")
    # endregion

    # region -------- Build Logger --------
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(opts.output_path, "logs", current_time)
    logger = SummaryWriter(log_dir)
    # endregion

    # region -------- Pretraining --------
    # Train SR generator without adversarial loss or discriminator.
    if opts.st_epoch == 1 and opts.pretrain_epochs >= 1:
        print("Start to pretrain...")
        G.train()

        optimizer = torch.optim.Adam(G.parameters(), lr=opts.pretrain_lr, betas=(opts.b1, opts.b2))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opts.pretrain_decay_step, 0.5)
        if opts.pretrain_optim is not None:
            access.load_model(optimizer, 0, opts.pretrain_optim)
        if opts.pretrain_sched is not None:
            access.load_model(scheduler, 0, opts.pretrain_sched)

        for epoch in range(1, opts.pretrain_epochs + 1):  # epoch = [1, opts.pretrain_epochs]
            for i, images in enumerate(train_loader):
                iteration = (epoch - 1) * len(train_loader) + i

                # Configure model input.
                hr_images = images["HR"].to(device)
                lr_images = images["LR"].to(device)

                optimizer.zero_grad()

                # Generate SR images from LR inputs.
                sr_images = G(lr_images)

                # Content loss between SR and HR images.
                ctt_loss = content_loss(sr_images, hr_images)

                ctt_loss.backward()
                optimizer.step()
                scheduler.step()

                # Log content loss.
                logger.add_scalar("pretrain/iter content loss", ctt_loss.item(), iteration)

            if opts.pretrain_chkpt_intrvl >= 1 and epoch % opts.pretrain_chkpt_intrvl == 0:
                path_to_save = SAVE_MODELS_PATH + f"/pretrain_epoch{epoch:04d}/"
                os.makedirs(path_to_save, exist_ok=True)
                # Designate epoch=0 when save model because of pre-training.
                access.save_model(G, 0, path_to_save + "G.pth")
                access.save_model(optimizer, 0, path_to_save + "optimizer.pth")
                access.save_model(scheduler, 0, path_to_save + "scheduler.pth")

            print(f"The epoch {epoch} has been finished in pretraining.")

        print("Pretraining complete.")
    # endregion

    # region -------- Formal Training --------
    print("Start to train formally...")
    for epoch in range(opts.st_epoch, opts.ed_epoch + 1):  # epoch = [opts.st_epoch, opts.ed_epoch]
        # region -------- Training --------
        G.train(), D.train()

        for i, images in enumerate(train_loader):
            iteration = (epoch - 1) * len(train_loader) + i

            # Configure model input.
            hr_images = images["HR"].to(device)
            lr_images = images["LR"].to(device)

            # region -------- Train Discriminator --------
            optimizer_D.zero_grad()

            # Generate SR images from LR inputs.
            sr_images = G(lr_images)

            loss_D_wo_GP = torch.mean(D(sr_images.detach())) - torch.mean(D(hr_images))
            gradient_penalty = losses.gradient_penalty(D, hr_images, sr_images.detach())

            loss_D = loss_D_wo_GP + gradient_penalty
            loss_D.backward()
            optimizer_D.step()
            # endregion

            if i % opts.n_critic == 0:
                # region -------- Train Generator --------
                optimizer_G.zero_grad()

                # adversarial loss
                adv_loss = -torch.mean(D(sr_images))

                # perceptual loss
                pcp_loss = percept_loss(sr_images, hr_images)

                # content loss
                ctt_loss = content_loss(sr_images, hr_images)

                # total loss
                loss_G = \
                    opts.w_adv_loss * adv_loss + \
                    opts.w_percept_loss * pcp_loss + \
                    opts.w_content_loss * ctt_loss

                loss_G.backward()
                optimizer_G.step()
                # endregion

                # region -------- Log Progress --------
                logger.add_scalar("train/W dist", -loss_D_wo_GP.item(), iteration)
                logger.add_scalar("train/GP term", gradient_penalty.item(), iteration)

                logger.add_scalar("train/G loss", loss_G.item(), iteration)
                logger.add_scalar("train/G perceptual loss", pcp_loss.item(), iteration)
                logger.add_scalar("train/G content loss", ctt_loss.item(), iteration)
                logger.add_scalar("train/G adversarial loss", adv_loss.item(), iteration)
                # endregion

            scheduler_G.step()
            scheduler_D.step()

            # region -------- Save Training Results --------
            if opts.sample_interval >= 1 and iteration % opts.sample_interval == 0:
                lr_images = nn_f.interpolate(lr_images, scale_factor=opts.scale_factor, mode="nearest")
                path_to_save = SR_TRAIN_RESULTS_PATH + f"/epoch{epoch:06d}_idx{i:04d}_iter{iteration:08d}/"
                os.makedirs(path_to_save, exist_ok=True)
                save_image(hr_images, path_to_save + f"HR.png")
                save_image(sr_images, path_to_save + f"SR.png")
                save_image(lr_images, path_to_save + f"LR.png")
            # endregion

        # endregion

        # region -------- SR Validation --------
        if opts.sr_valid_interval >= 1 and epoch % opts.sr_valid_interval == 0:
            assert sr_valid_loader.batch_size == 1, (
                f"The batch size should be 1 "
                f"rather than {sr_valid_loader.batch_size} during SR validation."
            )
            with torch.no_grad():
                G.eval(), D.eval()
                valid_res = {"W_dist": 0, "Y-PSNR": 0, "Y-SSIM": 0}
                for i, images in enumerate(sr_valid_loader):
                    hr_images = images["HR"].to(device)
                    lr_images = images["LR"].to(device)

                    sr_images = G(lr_images)

                    # Calculate metric values.
                    valid_res["W_dist"] += (torch.mean(D(hr_images)) - torch.mean(D(sr_images))).item()
                    valid_res["Y-PSNR"] += psnr_ssim.tensor_psnr(
                        sr_images.squeeze(0),
                        hr_images.squeeze(0),
                        test_y_channel=True
                    )
                    valid_res["Y-SSIM"] += psnr_ssim.tensor_ssim(
                        sr_images.squeeze(0),
                        hr_images.squeeze(0),
                        test_y_channel=True
                    )

                    # Save LR, SR and HR images.
                    if (opts.sr_valid_save_images is None) or (i in opts.sr_valid_save_images):
                        lr_images = nn_f.interpolate(lr_images, scale_factor=opts.scale_factor, mode="nearest")
                        path_to_save = SR_VALID_RESULTS_PATH + f"/epoch{epoch:06d}/"
                        os.makedirs(path_to_save, exist_ok=True)
                        save_image(hr_images, path_to_save + f"idx{i:04d}_HR.png")
                        save_image(sr_images, path_to_save + f"idx{i:04d}_SR.png")
                        save_image(lr_images, path_to_save + f"idx{i:04d}_LR.png")

                # Log average metric values in valid set.
                logger.add_scalar("SR valid/avg W dist", valid_res["W_dist"] / len(sr_valid_loader), epoch)
                logger.add_scalar("SR valid/avg Y-PSNR", valid_res["Y-PSNR"] / len(sr_valid_loader), epoch)
                logger.add_scalar("SR valid/avg Y-SSIM", valid_res["Y-SSIM"] / len(sr_valid_loader), epoch)
        # endregion

        # region -------- IQA Validation --------
        if opts.iqa_valid_interval >= 1 and epoch % opts.iqa_valid_interval == 0:
            assert iqa_valid_loader.batch_size == 1, (
                f"The batch size should be 1 "
                f"rather than {iqa_valid_loader.batch_size} during IQA validation."
            )
            iqa_valid_log_file = IQA_VALID_RESULTS_PATH + f"/epoch{epoch:06d}.txt"
            with open(iqa_valid_log_file, 'w') as iqa_valid_logger, torch.no_grad():
                D.eval()
                valid_res = {"image_name": [], "MOS": [], "pred_score": []}
                print("image_name,mos,pred", file=iqa_valid_logger)
                for i, samples in enumerate(iqa_valid_loader):
                    images = samples["image"].to(device)
                    mos = samples["MOS"]
                    names = samples["image_name"]
                    pred_score = D(images)

                    valid_res["image_name"].extend(names)
                    valid_res["MOS"].append(mos.item())
                    valid_res["pred_score"].append(pred_score.item())
                    print(f"{names[0]},{mos.item()},{pred_score.item()}", file=iqa_valid_logger)

                # Calculate PLCC and SRCC.
                plcc = correlation.plcc(valid_res["MOS"], valid_res["pred_score"])
                srcc = correlation.srcc(valid_res["MOS"], valid_res["pred_score"])
                krcc = correlation.krcc(valid_res["MOS"], valid_res["pred_score"])
                print(f"PLCC,{plcc},SRCC,{srcc},KRCC,{krcc}", file=iqa_valid_logger)

                # Log average metric values in valid set.
                logger.add_scalar("IQA valid/PLCC", plcc, epoch)
                logger.add_scalar("IQA valid/SRCC", srcc, epoch)
                logger.add_scalar("IQA valid/KRCC", krcc, epoch)
        # endregion

        # region -------- Checkpoint --------
        if opts.checkpoint_interval >= 1 and epoch % opts.checkpoint_interval == 0:
            path_to_save = SAVE_MODELS_PATH + f"/epoch{epoch:06d}/"
            os.makedirs(path_to_save, exist_ok=True)
            access.save_model(G, epoch, path_to_save + "G.pth")
            access.save_model(D, epoch, path_to_save + "D.pth")
            access.save_model(optimizer_G, epoch, path_to_save + "optimG.pth")
            access.save_model(optimizer_D, epoch, path_to_save + "optimD.pth")
            access.save_model(scheduler_G, epoch, path_to_save + "schedG.pth")
            access.save_model(scheduler_D, epoch, path_to_save + "schedD.pth")
        # endregion

        print(f"The epoch {epoch} has been finished in formal training.")

    print("Formal training complete.")
    logger.close()
    # endregion


if __name__ == '__main__':
    from options import train_srgan_opts

    main(train_srgan_opts.get_options())

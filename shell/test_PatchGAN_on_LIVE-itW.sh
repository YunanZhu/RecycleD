#!/bin/bash
cd ../  # Change to the main directory that has "test_iqa.py".
D:/Program/Anaconda/envs/py36_torch12/python.exe test_iqa.py  `# Modify this to use your own python.exe` \
\
--channels 3 \
\
--D "PatchVGG"  `# Support: "VGG", "PatchVGG", "DeeperVGG", "ResNet18".` \
--D_model "./pretrained_models/netD/RecycleD_SR-WGAN_Discriminator_PatchGAN_epoch2000_ForAuthDistImg.pth" \
--epoch 2000 \
\
--fusion_mode "SOD BASNet + SR residual" \
--coef_sod 5.0 \
--coef_res 380.0 \
--coef_avg 1.0 \
--norm_type "set_sum_1" \
\
--n_cpu 3 \
--batch_size 8 \
\
--test_set "E:/Resources/datasets/CLIVE/"  `# Your own LIVE-itW dataset path.` \
--test_set_guider "./datasets/CLIVE_all_pair_resize2img.txt" \
\
--save_pred 0  `# Set to 1 if you want to save the test results.` \
--res_file ""  `# Set to your own txt file if you want to save the test results.` \
\

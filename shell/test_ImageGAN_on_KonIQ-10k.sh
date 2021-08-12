#!/bin/bash
cd ../  # Change to the main directory that has "test_iqa.py".
D:/Program/Anaconda/envs/py36_torch12/python.exe test_iqa.py  `# Modify this to use your own "python.exe".` \
\
--channels 3 \
\
--D "VGG"  `# Support: "VGG", "PatchVGG", "DeeperVGG", "ResNet18".` \
--D_model "./pretrained_models/netD/RecycleD_SR-WGAN_Discriminator_ImageGAN_epoch2000_ForAuthDistImg.pth" \
--epoch 2000 \
\
--fusion_mode "Naive" \
--coef_sod 0.0  `# Useless here.` \
--coef_res 0.0  `# Useless here.` \
--coef_avg 0.0  `# Useless here.` \
\
--n_cpu 4 \
--batch_size 12 \
\
--test_set "E:/Resources/datasets/KonIQ-10K/1024x768/"  `# Your own KonIQ-10k dataset path.` \
--test_set_guider "./datasets/KonIQ-10k_all_pair.txt"  `# Test on the whole set.` \
\
--save_pred 0  `# Set to 1 if you want to save the test results.` \
--res_file ""  `# Set to your own txt file if you want to save the test results.` \
\

#!/bin/bash
export TZ=Asia/Shanghai  # You can simply delete this line or modify it.
cd ../  # Change to the main directory that has "test_iqa.py".
D:/Program/Anaconda/envs/py36_torch12/python.exe -u train_srgan.py  `# Use your own "python.exe" here.` \
\
--channels 3 \
--scale_factor 4 \
\
--st_epoch 1 \
--ed_epoch 2000  `# Though it can be trained for more epochs, we find the IQA performance has been relatively stable.` \
--n_critic 5 \
\
--sample_interval 1000 \
--checkpoint_interval 50 \
\
--output_path "./output/" \
\
--G "RRDBNet"  `# You can also choose "MSRResNet" or "EnhanceNet".` \
--D "PatchVGG"  `# You can set to "VGG" for ImageGAN discriminator training.` \
\
--train_set "E:/Resources/datasets/DIV2K/"  `# Use your own DIV2K dataset path.` \
--train_set_form "Pair"  `# Use "Pair" when LR images have been prepared. Use "Single" when you only have HR images.` \
--train_set_guider "./datasets/DIV2K_train_X4_pair.txt" \
--lr_size 48  `# Use 192x192 HR patches when perform x4 SISR.` \
--augment_data 1 \
--batch_size 16 \
\
--sr_valid_set "E:/Resources/datasets/SISR_benchmark_datasets/Set14/"  `# Use your own SR validation set path.` \
--sr_valid_set_form "SR_Single"  `# Use "SR_Pair" when LR images have been prepared. Use "SR_Single" when you only have HR images.` \
--sr_valid_set_guider "./datasets/Set14_single.txt" \
--sr_valid_interval 50  `# Recommend to use the same interval with "--checkpoint_interval".` \
--sr_valid_save_images 0 1 2 3  `# I choose to save first 4 images for valid visualization.` \
\
--iqa_valid_interval -1  `# I did not perform IQA validation here, you can modify the related options for your own IQA validation.` \
\
--n_cpu 4 \
\
--lr 1e-4 \
--b1 0.9 \
--b2 0.999 \
--adjust_lr 1 \
\
--pretrain_epochs 30 \
--pretrain_lr 2e-4 \
--pretrain_adjust_lr 1 \
--pretrain_chkpt_intrvl 5 \
\
--w_percept_loss 1.0 \
--w_adv_loss 5e-3 \
--w_content_loss 1e-2 \
--content_criterion "L1" \
--percept_criterion "L1" \
--percept_vgg_model "E:/Resources/models/vgg/vgg19-dcbb9e9d.pth"  `# This model file is provided by PyTorch, I download it and load it manually.` \
--percept_last_layer 35 \
\

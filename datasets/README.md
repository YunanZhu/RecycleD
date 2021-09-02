# Filelists

These text files are all filelists.

## SR dataset
"DIV2K_train_X4_pair.txt", "DIV2K_valid_X4_pair.txt", "Flickr2K_all_X4_pair.txt" and "DF2K_train_X4_pair.txt" are filelists of DIV2K & Flickr2K.
Their formats are all similar to:
```
DIV2K_train_HR/0001.png,DIV2K_train_LR_bicubic/X4/0001x4.png
DIV2K_train_HR/0002.png,DIV2K_train_LR_bicubic/X4/0002x4.png
DIV2K_train_HR/0003.png,DIV2K_train_LR_bicubic/X4/0003x4.png
DIV2K_train_HR/0004.png,DIV2K_train_LR_bicubic/X4/0004x4.png
DIV2K_train_HR/0005.png,DIV2K_train_LR_bicubic/X4/0005x4.png
...
```
In a single line, there is a pair of one HR image and one LR image. HR image is the first one, and LR image is the 2nd one. It uses commas to separate.

"Set5_single.txt" and "Set14_single.txt" are used to validate for SR WGAN training. Their single line only has one HR image without LR image.

## IQA dataset
All of these are filelists of IQA datasets:
```
"CLIVE_all_pair_original.txt", "CLIVE_all_pair_resize2img.txt",
"KonIQ-10k_all_pair.txt", "KonIQ-10k_test_pair.txt", "KonIQ-10k_train_pair.txt", "KonIQ-10k_valid_pair.txt",
"MaScoreDataset_all_pair.txt",
"PIPAL_train_AllSR_pair.txt", "PIPAL_train_DenoSR_pair.txt", "PIPAL_train_GANbSR_pair.txt", "PIPAL_train_KnMmSR_pair.txt", "PIPAL_train_PsnrSR_pair.txt", "PIPAL_train_TradSR_pair.txt", "PIPAL_valid_pair_woGT.txt",
```
"BappsSuperres_all_pair.txt" is the filelist of [the BAPPS dataset (proposed in LPIPS)](https://github.com/richzhang/PerceptualSimilarity#2-berkeley-adobe-perceptual-patch-similarity-bapps-dataset). It is also used to test.

# Train & Test
You can use these shell files to train SR WGAN, or to test the discriminator on IQA datasets.  
Please modify the necessary options according to the annotations, to ensure that they can run successfully in your environment.

## Train
Run [train_SR-WGAN.sh](shell/train_SR-WGAN.sh).

## Test
Test on super-resolved images:
- [test_ImageGAN_on_PIPAL.sh](shell/test_ImageGAN_on_PIPAL.sh)
- [test_PatchGAN_on_MaDataset.sh](shell/test_PatchGAN_on_MaDataset.sh)
- [test_PatchGAN_on_BAPPS-Superres.sh](shell/test_PatchGAN_on_BAPPS-Superres.sh)
- [test_ImageGAN_on_KonIQ-10k.sh](shell/test_ImageGAN_on_KonIQ-10k.sh)

Test on authentically distorted images:
- [test_PatchGAN_on_LIVE-itW.sh](shell/test_PatchGAN_on_LIVE-itW.sh)
- [test_PatchGAN_on_KonIQ-10k.sh](shell/test_PatchGAN_on_KonIQ-10k.sh)

# Data and model attribution

- Dataset: Pacheco et al., **PAD-UFES-20**, version 1, DOI [10.17632/zr7vgbcyr2.1](https://doi.org/10.17632/zr7vgbcyr2.1). The publisher distributes it under **CC BY 4.0**. Images and clinical records are downloaded separately from the publisher; this repository includes code and aggregate experiment results.
- Dataset paper: Pacheco et al. (2020), *PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones*, Data in Brief 32, 106221. DOI [10.1016/j.dib.2020.106221](https://doi.org/10.1016/j.dib.2020.106221).
- U-Net: Ronneberger, Fischer and Brox (2015), [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). This project uses a custom contracting-path encoder; it does not implement a complete segmentation U-Net.
- ResNet: He, Zhang, Ren and Sun (2015), [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). ResNet50 and ImageNet1K V2 weights are obtained through [torchvision](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html).

Third-party data, libraries and pretrained weights retain their respective terms. The dataset's CC BY 4.0 license is not a license grant for this project's source code. No separate source-code license is declared in this repository.

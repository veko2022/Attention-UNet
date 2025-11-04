# Attention-UNet
Attention U-Net for accurate image segmentation using deep learning with attention mechanisms.


Attention U-Net for Image Segmentation
This repository contains the implementation of the Attention U-Net, a deep learning model for image segmentation that incorporates attention gates into the U-Net architecture.


Features
*Attention gates to focus on relevant spatial features
*Encoder-decoder structure with skip connections
*Achieves high segmentation accuracy with Dice score ~0.94 and IoU ~0.85


Install dependencies:
pip install -r requirements.txt


Dataset
This project uses the Carvana Image Masking Challenge Dataset for training and evaluation. It contains high-resolution car images and their corresponding masks for segmentation tasks.
The dataset consists of car images (*.jpg) and binary masks (*_mask.gif).
The code expects the dataset to be organized in two directories: one for input images and one for masks.
You can download the dataset from the Carvana dataset page.

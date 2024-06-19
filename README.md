# Gradient-Optimized Dice Loss

This repository provides a PyTorch based implementation of the Gradient-Optimized Dice Loss function as described in the paper:

**Towards Accurate Medical Image Segmentation With Gradient-Optimized Dice Loss** by Q. Ming and X. Xiao (2024)


## Description

The Gradient-Optimized Dice Loss is a novel loss function designed to improve the accuracy of medical image segmentation tasks. It corrects the abnormal gradient changes in the segmentation loss, which accelerates the model convergence and can achieve better segmentation performance.


## Paper

You can access the original paper [here](https://ieeexplore.ieee.org/document/10304274). Please cite the paper if you use this code in your research or work.

```
@ARTICLE{10304274,
  author={Ming, Qi and Xiao, Xiaowu},
  journal={IEEE Signal Processing Letters}, 
  title={Towards Accurate Medical Image Segmentation With Gradient-Optimized Dice Loss}, 
  year={2024},
  volume={31},
  number={},
  pages={191-195},
  keywords={Image segmentation;Feature extraction;Training;Medical diagnostic imaging;Convergence;Semantics;Task analysis;Medical image segmentation;dice loss;convolutional neural network;gradient descent algorithm},
  doi={10.1109/LSP.2023.3329437}}
```
  
## Contributing

If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request. Contributions are welcome!

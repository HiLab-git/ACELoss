# Active Contour Euler Elastica Loss Functions
Official implementations of paper: [Learning Euler's Elastica Model for Medical Image Segmentation](https://arxiv.org/submit/3446612/view).
* Implemented a novel active contour-based loss function, a combination of region term, length term, and elastica term (mean curvature).
* Reimplemented some popular active contour-based loss functions in different ways, such as 3D Active-Contour-Loss based on Sobel filter and max-and min-pool.

## Introduction
![](https://github.com/Luoxd1996/Active_Contour_Euler_Elastica_Loss/blob/main/ACELoss_pipeline.png) 

* If you want to use these methods just as constrains (combining with dice loss or ce loss), you can use **torch.mean()** to replace **torch.sum()**.

## Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.
* Python == 3.6.

Follow official guidance to install. [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

## Citation
If you find Active Contour Based Loss Functions are useful in your research, please consider to cite:

	@inproceedings{chen2020aceloss,
	  title={Learning Euler's Elastica Model for Medical Image Segmentation},
	  author={Chen, Xu and Luo, Xiangde and Zhao, Yitian and Zhang, Shaoting and Wang, Guotai and Zheng, Yalin},
	  journal={arXiv preprint arXiv:arxiv.org/submit/3446612/view},
	  year={2020}
	}

	@inproceedings{chen2019learning,
	  title={Learning Active Contour Models for Medical Image Segmentation},
	  author={Chen, Xu and Williams, Bryan M and Vallabhaneni, Srinivasa R and Czanner, Gabriela and Williams, Rachel and Zheng, Yalin},
	  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
	  pages={11632--11640},
	  year={2019}
	}

## Other Active Contour Based Loss Functions
* Active Contour Loss. ([ACLoss](https://github.com/xuuuuuuchen/Active-Contour-Loss))


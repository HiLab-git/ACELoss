# Active Contour Euler Elastica Loss Functions
Official implementations of paper: [Learning Euler's Elastica Model for Medical Image Segmentation](https://arxiv.org/pdf/2011.00526.pdf), and a short version was accepted by ISBI 2021 .
* Implemented a novel active contour-based loss function, a combination of region term, length term, and elastica term (mean curvature).
* Reimplemented some popular active contour-based loss functions in different ways, such as 3D Active-Contour-Loss based on Sobel filter and max-and min-pool.

## Introduction and Some Results
* ### **Pipeline of ACE loss**.
![](https://github.com/Luoxd1996/Active_Contour_Euler_Elastica_Loss/blob/main/ACELoss_pipeline.png) 
* ### **2D results and visualization**.
![](https://github.com/Luoxd1996/Active_Contour_Euler_Elastica_Loss/blob/main/table1.png) 
![](https://github.com/Luoxd1996/Active_Contour_Euler_Elastica_Loss/blob/main/figure1.png) 
* ### **3D results and visualization**.
![](https://github.com/Luoxd1996/Active_Contour_Euler_Elastica_Loss/blob/main/table2.png) 
![](https://github.com/Luoxd1996/Active_Contour_Euler_Elastica_Loss/blob/main/figure2.png) 

* If you want to use these methods just as constrains (combining with dice loss or ce loss), you can use **torch.mean()** to replace **torch.sum()**.

## Requirements
Some important required packages include:
* [Pytorch][torch_link] version >= 0.4.1.
* Python  >= 3.6.

Follow official guidance to install. [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

## Citation
If you find Active Contour Based Loss Functions are useful in your research, please consider to cite:

	@inproceedings{chen2020aceloss,
	  title={Learning Euler's Elastica Model for Medical Image Segmentation},
	  author={Chen, Xu and Luo, Xiangde and Zhao, Yitian and Zhang, Shaoting and Wang, Guotai and Zheng, Yalin},
	  journal={arXiv preprint arXiv:2011.00526},
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
* Active Contour Loss ([ACLoss](https://github.com/xuuuuuuchen/Active-Contour-Loss)).
* Geodesic Active Contour Loss ([GAC](https://ieeexplore.ieee.org/document/9187860)).
* Elastic-Interaction-based Loss ([EILoss](https://github.com/charrywhite/elastic_interaction_based_loss))

## Acknowledgement
* We thank [Dr. Jun Ma](https://github.com/JunMa11) for instructive discussion of curvature implementation and also thank [Mr. Yechong Huang](https://github.com/huohuayuzhong) for instructive help during the implementation processing of 3D curvature, Sobel, and  Laplace operators.

# Active_Contour_Euler_Elastica_Loss
Official implementations of Learning Euler's Elastica Model for Medical Image Segmentation

## Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.
* Python == 3.6 
Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

# Notes
More details will be released latter.

If you want to use these methods just as constrains (combine with dice loss or ce loss), you can use **torch.mean()** to replace **torch.sum()**.

## Citation
If you find Active-Contour-Loss is useful in your research, please consider to cite:

	@inproceedings{chen2019learning,
	  title={Learning Active Contour Models for Medical Image Segmentation},
	  author={Chen, Xu and Williams, Bryan M and Vallabhaneni, Srinivasa R and Czanner, Gabriela and Williams, Rachel and Zheng, Yalin},
	  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
	  pages={11632--11640},
	  year={2019}
	}

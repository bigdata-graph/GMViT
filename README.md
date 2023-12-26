## Pytorch code for GMViT.
**Group Multi-View Transformer for 3D Shape Analysis with Spatial Encoding**.

## Trainning
###  Requiement
This code is tested on Python 3.6 and Pytorch 1.0 +.
###  Dataset
First download the ModelNet datasets and unzip inside the `data/` directories as follows:

- [[Dodecahedron-20]](https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet40v2png_ori4.tar
) is provided by [[RotationNet]](https://github.com/kanezaki/pytorch-rotationnet) (rendered from viewpoints uniformly sampled on a bounding sphere encompassing the 3D object, corresponding to virtual camera positions at the twenty vertices of a dodecahedron).
- [[Circle-12]](https://supermoe.cs.umass.edu/shape_recog/depth_images.tar.gz) is provided by [[MVCNN]](https://github.com/jongchyisu/mvcnn_pytorch) (rendered from 12 virtual camera viewpoints evenly spaced around the object circumference at an elevation angle of 30 degrees).

### Command for training
- Pretrain CNN model in the `train_cnn/` directory: 

      python train_cnn.py

- Train GMViT: 

      python train.py -name GMViT -num_views 20 -group_num 12 -cnn_name resnet18 
      
- Distillation student model: 

      python KD_GMViT_simple.py -name GMViT_simple -num_views 20 -group_num 12
      python KD_GMViT_mini.py -name GMViT_mini -num_views 20 -group_num 12

## Acknoledgements

This paper and repo borrows codes and ideas from several great github repos: [MVCNN](https://github.com/RBirkeland/MVCNN-PyTorch), [PointNet](https://github.com/charlesq34/pointnet), [GVCNN](https://github.com/waxnkw/gvcnn-pytorch).

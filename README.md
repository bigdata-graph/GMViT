## Pytorch code for GMViT.
**Group Multi-View Transformer for 3D Shape Analysis with Spatial Encoding**.

## Trainning
###  Requiement
This code is tested on Python 3.6 and Pytorch 1.0 +.
###  Dataset
First download the ModelNet datasets and unzip inside the `data/` directories as follows:

- Dodecahedron-20 [[this link]](https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet40v2png_ori4.tar
)  (rendered from viewpoints uniformly sampled on a bounding sphere encompassing the 3D object, corresponding to virtual camera positions at the twenty vertices of a dodecahedron).
- Circle-12 [[this link]](https://supermoe.cs.umass.edu/shape_recog/depth_images.tar.gz) (rendered from 12 virtual camera viewpoints evenly spaced around the object circumference at an elevation angle of 30 degrees).

### Command for training
- Pretrain CNN model in the `train_cnn/` directory: 

      python train_cnn.py

- Train GMViT: 

      python train.py -name GMViT -num_views 20 -group_num 12 -cnn_name resnet18 
      
- Distillation student model: 

      python KD_GMViT_simple.py -name GMViT_simple -num_views 20 -group_num 12
      python KD_GMViT_mini.py -name GMViT_mini -num_views 20 -group_num 12

### Pretrained Model
We have provided a pre-trained model in [[here]](https://pan.baidu.com/s/1uSt-RxG3zhUZeSVjNS4fng?pwd=bs66) (code: bs66), achieving an Overall Accuracy (OA) of 97.77% and a mean Accuracy (mA) of 97.07% under the Dodecahedron-20 setting on the ModelNet40 dataset. Please download it and please it in `models/GMViT/models/model.t7` directory.

## References

L. Xu, Q. Cui, W. Xu, E. Chen, H. Tong, Y. Tang, Walk in views: Multi-view path aggregation graph network for 3d shape analysis, Information Fusion 103 (2024) 102131.

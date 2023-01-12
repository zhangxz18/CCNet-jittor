# CCNet-jittor
基于Jittor框架完成的《CCNet: Criss-Cross Attention for Semantic Segmentation》复现。支持VAN或ResNet作为Backbone，Cityscapes或ADE20K作为数据集。  
## 运行
从 https://www.cityscapes-dataset.com/ 获取Cityscapes，或从 https://groups.csail.mit.edu/vision/datasets/ADE20K/ 获取ADE20K。
train的方法见train.sh(单gpu）或train_mpi.sh（多gpu），eval的方法见eval.sh。将data-dir参数修改为数据集所在路径。
## Reference
https://github.com/speedinghzl/CCNet
https://github.com/speedinghzl/CCNet/tree/pure-python


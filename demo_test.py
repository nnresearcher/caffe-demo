# -*- coding: utf-8 -*-

# source ~/.bashrc

import caffe
from CaffeModel import lenet

# 将这个模型的细节写入到文件夹中
with open('test2.prototxt', 'w') as f:
    f.write(str(lenet()))
# 设置为GPU模式
caffe.set_mode_gpu()

caffeModel = lenet("~/caffe/examples/mnist/mnist_train_lmdb",64)

solver = caffe.SGDSolver('lenet_auto_solver.prototxt')
a = [(k, v.data.shape) for k, v in solver.net.blobs.items()]
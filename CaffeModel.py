# -*- coding: utf-8 -*-
# source ~/.bashrc
import caffe
from caffe import layers as L, params as P

def lenet(param):
    n = caffe.NetSpec()
    n.data, n.label = L.Python(module='wwt_data_layer', layer='wwtdatalayer',ntop=2,param_str=str(param))
    n.bn1 = L.BatchNorm(n.data,batch_norm_param=dict(moving_average_fraction=0.90,use_global_stats=False,eps=1e-5),in_place=True)

    n.conv1 = L.Convolution(n.bn1, kernel_size=5, num_output=64, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.conv1, in_place=True)
    n.pool1 = L.Pooling(n.relu1, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.relu2 = L.ReLU(n.pool1, in_place=True)

    n.conv2 = L.Convolution(n.relu2, kernel_size=5, num_output=64, weight_filler=dict(type='xavier'))
    n.relu3 = L.ReLU(n.conv2, in_place=True)
    n.pool3 = L.Pooling(n.relu3, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.relu4 = L.ReLU(n.pool3, in_place=True)

    n.fc1 =   L.InnerProduct(n.relu4, num_output=384, weight_filler=dict(type='xavier'))
    n.relu5 = L.ReLU(n.fc1, in_place=True)

    n.fc2 = L.InnerProduct(n.relu5, num_output=192, weight_filler=dict(type='xavier'))
    n.relu6 = L.ReLU(n.fc2, in_place=True)

    n.score = L.InnerProduct(n.relu6, num_output=10, weight_filler=dict(type='xavier'))
    n.loss =  L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()


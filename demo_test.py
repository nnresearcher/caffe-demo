# -*- coding: utf-8 -*-

# source ~/.bashrc

import caffe
import numpy as np
from CaffeModel import lenet
from solver import solver
# 将这个模型的细节写入到文件夹中
solver_path = 'solver.prototxt'
train_net_path = 'train_test.prototxt'
test_net_path = 'train_test.prototxt'


param=dict(batch_size=64, height=32,width=32,train_url='train.txt')
# 建立模型文件，写成prototxt格式
with open('train_test.prototxt', 'w') as f:
    f.write(str(lenet(param)))

# 建立求解文件，再写成prototxt格式
with open(solver_path, 'w') as f:
    f.write(str(solver(train_net_path, test_net_path)))

# 设置模型为GPU模式
caffe.set_mode_gpu()

# 实例化模型求解方法
solver = caffe.get_solver(solver_path)
train_iter = 100
test_iter = 100
for _ in range(test_iter):
    for _ in range(train_iter):
        solver.step(10)
    solver.test_nets[0].forward()
    # solver.test_nets[0].forward(start='conv1')
    accuracy = 0
    for i in range(len(solver.test_nets[0].blobs['score'].data)):
        test_data = solver.test_nets[0].blobs['score'].data[i]
        pre_max_index = int(np.where(test_data==max(test_data))[0][0])
        true_index = int(solver.test_nets[0].blobs['label'].data[i][0])
        if true_index == pre_max_index:
            accuracy = accuracy + 1
    print 'accuracy:',accuracy
    stop = None



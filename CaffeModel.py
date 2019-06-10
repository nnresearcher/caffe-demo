# -*- coding: utf-8 -*-
# source ~/.bashrc
import caffe
import random
import cv2
import sys
import numpy as np
from caffe import layers as L, params as P

class CaffeDataLayer(caffe.Layer):

    def __init__(self,batch_size,crop_size,train_url,shuffle):
        self.batch_size = batch_size
        self.train_url = train_url
        self.top_names = ['data', 'label']
        self.height = crop_size[0]
        self.width = crop_size[1]
        self.shuffle = shuffle

    def BatchLoader(self):
        f = open(self.train_url)
        x_data = []
        y_label = []
        while True:
            line = f.readline()
            if not line:
                break
            x, y = [str(i) for i in line.split()]
            x_data.append(x)
            y_label.append(y)
        len_data = len(x_data)

        while True:
            input = []
            output = []
            for _ in range(self.batch_size):
                index = random.randint(0, len_data)
                input.append(x_data[index])
                output.append(y_label[index])
            yield [input, output]

    def setup(self, bottom, top):
        self.batch_loader = self.BatchLoader()
        top[0].reshape(self.batch_size, 3, self.height, self.width)
        top[1].reshape(self.batch_size, 1)

    def forward(self, bottom, top):
        batch = self.batch_loader.next()
        labels = batch[1]
        imgs = []
        if self.shuffle:
            random.shuffle(batch)
        for it in batch[0]:
            img = cv2.imread(it)
            if img is None:
                print("cv2.read %s is None, exit.".format(it))
                sys.exit(1)
            img = np.reshape(img, [self.height, self.width, 3])
            img = np.transpose(img, (2, 0, 1))  # convert (height, width, 3) to (3, height, width)
            imgs.append(img)

        top[0].data = np.array(imgs)
        top[1].data = np.array(labels)

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


def lenet():
    # our version of LeNet: a series of linear and simple nonlinear transformations
    train_url = '~/caffe/examples/wwt/data/train/train.txt'

    n = caffe.NetSpec()
    n.data, n.label = CaffeDataLayer(batch_size=64,crop_size=[64,64],train_url = train_url,shuffle=True)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss =  L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()




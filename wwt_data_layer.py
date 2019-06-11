# imports
import caffe
import numpy as np
from PIL import Image
import random


class wwtdatalayer(caffe.Layer):

    """
    This is a simple synchronous datalayer for training a multilabel model on
    PASCAL.
    """

    def setup(self, bottom, top):

        # self.param = eval(self.param_str)
        # self.batch_size = self.param['batch_size']
        # self.height = self.param['crop_size'][0]
        # self.width = self.param['crop_size'][1]
        # self.train_url = self.param['train_url']

        self.batch_size = 64
        self.height = 64
        self.width = 64
        self.train_url = 'train.txt'
        self.batch_loader = BatchLoader(self.train_url,self.batch_size)
        top[0].reshape(self.batch_size, 3, self.width, self.height)
        top[1].reshape(self.batch_size, 1)

    def forward(self, bottom, top):
        """
        Load data.
        """

        for itt in range(self.batch_size):
            img, label = self.batch_loader.load_next_image()
            img = np.resize(img,[self.height, self.width, 3])
            img = np.transpose(img, (2, 0, 1))
            top[0].data[itt, ...] = img
            top[1].data[itt, ...] = label
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


class BatchLoader(object):

    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, train_url, batch_size):
        self.train_url = train_url
        self.batch_size = batch_size

        f = open(self.train_url)
        self.x_data = []
        self.y_label = []
        while True:
            line = f.readline()
            if not line:
                break
            x, y = [str(i) for i in line.split()]
            self.x_data.append(x)
            self.y_label.append(y)
        self.len_data = len(self.x_data)

    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        # if self._cur+self.batch_size > self.len_data:
        #    self._cur = 0


        index = random.randint(0, self.len_data-1)
        input = np.asarray(Image.open(self.x_data[index]))
        output= np.float(self.y_label[index])


        return input, output

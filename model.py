import chainer.functions as F
import chainer.links as L
from chainer import Chain, optimizers
from chainer import Variable as V
import cupy as xp

class GCN(Chain):
    def __init__(self, in_channels, out_channels, ksize=4, stride=2, pad=1):
        super(GCN, self).__init__()
        with self.init_scope():
            self.conv0_0 = L.Convolution2D(in_channels, out_channels, ksize=(1, ksize), stride=(1, stride), pad=(0, pad))
            self.conv0_1 = L.Convolution2D(in_channels, out_channels, ksize=(ksize, 1), stride=(stride, 1), pad=(pad, 0))
            self.conv1_0 = L.Convolution2D(in_channels, out_channels, ksize=(ksize, 1), stride=(stride, 1), pad=(pad, 0))
            self.conv1_1 = L.Convolution2D(in_channels, out_channels, ksize=(1, ksize), stride=(1, stride), pad=(0, pad))

    def __call__(self, x):
        return self.conv0_1(self.conv0_0(x)) + self.conv1_1(self.conv1_0(x))

class reduction(Chain):
    def __init__(self, in_channels, out_channels, pooling=F.average_pooling_2d, activation=F.leaky_relu):
        super(reduction, self).__init__()
        with self.init_scope():
            convOutChannels = (out_channels - in_channels) / 2
            # self.conv0 = L.Convolution2D(in_channels, out_channels , ksize=3, stride=2)
            self.conv0 = GCN(in_channels, convOutChannels)
            self.conv1 = L.Convolution2D(in_channels, in_channels, ksize=1)
            self.conv2 = L.Convolution2D(in_channels, convOutChannels / 2, ksize=3, pad=1)
            self.conv3 = GCN(convOutChannels, convOutChannels, ksize=3, stride=2)
        self.pooling = pooling
        self.activation = activation
    
    def __call__(self, x):
        h0 = self.pooling(x, ksize=2, stride=2)
        h1 = self.activation(self.conv0(x))
        h2 = self.activation(self.conv1(x))
        h2 = self.activation(self.conv2(h2))
        h2 = self.activation(self.conv3(h2))
        h = F.concat([h0, h1, h2])
        return h

class Inception(Chain):
    def __init__(self, in_channels, out_channels, activation=F.leaky_relu):
        super(Inception, self).__init__()
        with self.init_scope():
            self.conv0 = GCN(in_channels, in_channels//8, 1)
            self.conv1 = L.Convolution2D(in_channels, in_channels//8, 1)
            self.conv2 = GCN(in_channels//8, in_channels//8, 3, pad=1)
            self.conv3 = L.Convolution2D(in_channels, in_channels//8, 1)
            self.conv4 = GCN(in_channels//8, in_channels*3//16, 7, pad=3)
            self.conv5 = GCN(in_channels*3//16, in_channels//4, 7, pad=3)
            self.conv6 = L.Convolution2D(128, out_channels, 1)
        self.activation = activation

    def __call__(self, x):
        h0 = self.conv0(x)
        h1 = self.conv1(x)
        h1 = self.activation(h1)
        h1 = self.conv2(h1)
        h2 = self.conv3(x)
        h2 = self.activation(h2)
        h2 = self.conv4(h2)
        h2 = self.activation(h2)
        h2 = self.conv5(h2)
        h = F.concat([h0, h1, h2])
        h = self.activation(h)
        h = self.conv6(h)
        h = h + x
        h = self.activation(h)
        return h

class Inception1(Chain):
    def __init__(self, activation=F.leaky_relu):
        super(Inception1, self).__init__()
        with self.init_scope():
            self.conv0 = L.Convolution2D(256, 32, 1)
            self.conv1 = L.Convolution2D(256, 32, 1)
            self.conv2 = L.Convolution2D(32, 32, 3, pad=1)
            self.conv3 = L.Convolution2D(256, 32, 1)
            self.conv4 = L.Convolution2D(32, 48, 3, pad=1)
            self.conv5 = L.Convolution2D(48, 64, 3, pad=1)
            self.conv6 = L.Convolution2D(128, 256, 1)
        self.activation = activation

    def __call__(self, x):
        h0 = self.conv0(x)
        h1 = self.conv1(x)
        h1 = self.activation(h1)
        h1 = self.conv2(h1)
        h2 = self.conv3(x)
        h2 = self.activation(h2)
        h2 = self.conv4(h2)
        h2 = self.activation(h2)
        h2 = self.conv5(h2)
        h = F.concat([h0, h1, h2])
        h = self.activation(h)
        h = self.conv6(h)
        h = h + x
        h = self.activation(h)
        return h


class model(Chain):
    def __init__(self):
        super(model, self).__init__()
        with self.init_scope():
            # self.conv0 = L.Convolution2D(3, 64, ksize=3, stride=2)
            self.conv0 = GCN(3, 64)
            for i in range(1, 6):
                self.add_link(f"conv{i}", reduction(2**i*32, 2**i*64))
            self.inception = Inception(2048, 2048)

            self.conv_ = L.Convolution2D(64, 3, ksize=1)
            
    
    def __call__(self, x):
        shape = [x.shape]
        h = self.conv0(x)
        shape.append(h.shape)
        h = F.leaky_relu(h)
        h_ = [h]

        for i in range(1, 6):
            h = self[f"conv{i}"](h)
            h_.append(h)
            shape.append(h.shape)

        for i in range(5):
            h = self.inception(h)
        
        for i in range(5,-1,-1):
            h = self[f"dc{i}"](h, h_[i])[:,:,1:shape[i][2],1:shape[i][3]]
        
        h = self.conv_(h)

        return h


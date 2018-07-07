import numpy as np
import chainer.functions as F
from chainer.optimizers import Adam
import cupy as cp
from model import Model
from data import Data, Eshi
import pickle
from concurrent.futures import ThreadPoolExecutor
from chainer.serializers import save_npz

class Train:
    def __init__(self):
        with open("data.pickle", "rb") as f:
            self.data = pickle.load(f)
        self.model = Model()
        self.model.to_gpu()
        self.optimizer = Adam()
        self.optimizer.setup(self.model)

        self.executor = ThreadPoolExecutor(8)

        # self.hoge = self.data.next(2, 2)
        self.hoge = self.executor.submit(self.data.next, 2, 2)
        self.one = cp.ones(1, dtype=int)
        self.zero = cp.zeros(1, dtype=int)
    
    def load(self):
        d = self.hoge.result()
        self.hoge = self.executor.submit(self.data.next, 2, 2)
        return d

    def training(self):
        for i in range(1000000000000000):
            a = self.batch()
            if i%100==0:
                print(f"{i} loss:{a}")
                save_npz(f"param/model{i}.npz", self.model)

        
    def batch(self):
        a, b = self.load()
        self.model.cleargrads()
        # self.model(a[0])
        # input()
        y = tuple(self.executor.map(self.model, a + b))
        loss = F.contrastive(y[0], y[1], self.one) +\
               F.contrastive(y[2], y[3], self.one) +\
               F.contrastive(y[0], y[2], self.zero) +\
               F.contrastive(y[1], y[3], self.zero) 
        loss.backward()
        self.optimizer.update()
        return loss.data.get()


def main():
    train = Train()
    train.training()

if __name__ == '__main__':
    main()
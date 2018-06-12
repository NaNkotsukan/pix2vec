import numpy as np
import chainer.functions as F
from chainer.optimizers import Adam
import cupy as cp
from model import Model
from data import Data, Eshi
import pickle
from concurrent.futures import ThreadPoolExecutor

class Train:
    def __init__(self):
        with open("data.pickle", "rb") as f:
            self.data = pickle.load(f)
        self.model = Model()
        self.model.to_gpu()
        self.optimizer = Adam()
        self.optimizer.setup(self.model)

        self.executor = ThreadPoolExecutor(8)

        self.hoge = self.data.next(2, 2)
    
    def load(self):
        d = self.hoge
        self.hoge = self.executor.submit(self.data.next, 2, 2)
        return d

    def training(self):
        for i in range(1000000000000000):
            a = self.batch()
            if i%100==0:
                print(f"{i} loss:{a}")

        
    def batch(self):
        a, b = self.load()
        self.model.cleargrads()
        y = tuple(self.executor.map(self.model, a + b))
        loss = F.contrastive(y[0], y[1], [1]) +\
               F.contrastive(y[2], y[3], [1]) +\
               F.contrastive(y[0], y[2], [0]) +\
               F.contrastive(y[1], y[3], [0]) 
        loss.backward()
        self.optimizer.update()
        return loss.data.get()


def main():
    train = Train()
    train.training()

if __name__ == '__main__':
    main()
# import cupy as cp
import numpy as np
from PIL import Image
import os
import pickle

class Eshi:
    def __init__(self, path, id):
        self.id = id
        self.imageList = os.listdir(f"{path}/{id}")
        self.getID = self.genID()
        self.path = f"{path}/{self.id}/"
    
    def genID(self):
        while True:
            for i in np.random.permutation(len(self.imageList)):
                yield self.imageList[i]
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['getID']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.getID = self.genID()

    def next(self, n):
        # Image.open(f"/illust/{self.id}/{next(self.getID)}")
        # np.transpose(np.array(Image.open(f"/illust/{self.id}/{next(self.getID)}"))[:,:,:3],(2,0,1))
        return [np.transpose(np.asarray(Image.open(f"{self.path}{next(self.getID)}").convert("RGB")),(2,0,1)) for _ in range(n)]


class Data:
    def __init__(self, path):
        self.load = np.loadtxt("list.txt", dtype=np.int, delimiter=",")
        self.load = self.load[self.load[:,1]>20, 0]
        print(self.load)
        print(self.load.shape)
        self.eshiList = [Eshi(path, x) for x in self.load]
        self.getEshi = self.genEshi()
    
    def genEshi(self):
        while True:
            for i in np.random.permutation(len(self.eshiList)):
                yield self.eshiList[i]
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['getEshi']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.getEshi = self.genEshi()
    
    def next(self, m, n):
        return [next(self.getEshi).next(n) for x in range(m)]


def main():
    path = "D:/data/dataset/illust/pixiv_images"
    hoge = Data(path)
    with open('data.pickle', 'wb') as f:
        pickle.dump(hoge, f)


if __name__ == '__main__':
    main()
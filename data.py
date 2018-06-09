# import cupy as cp
import numpy as np
from PIL import Image
import os


class eshi:
    def __init__(self, id):
        self.id = id
        self.imageList = os.listdir(f"/illust/{id}")
        self.getID = self.genID()
    
    def genID(self):
        while True:
            for i in np.random.permutation(len(self.imageList)):
                yield self.imageList[i]
    
    def next(self, n):
        # Image.open(f"/illust/{self.id}/{next(self.getID)}")
        # np.transpose(np.array(Image.open(f"/illust/{self.id}/{next(self.getID)}"))[:,:,:3],(2,0,1))
        return [np.transpose(np.array(Image.open(f"/illust/{self.id}/{next(self.getID)}"))[:,:,:3],(2,0,1)) for i in range(n)]


class data:
    def __init__(self):
        self.load = np.loadtxt("list.txt", dtype=np.int, delimiter=",")
        self.load = self.load[self.load[:,1]>20]
        self.eshiList = [eshi(x) for x in self.load]
        self.getEshi = self.genEshi()
    
    def genEshi(self):
        while True:
            for i in np.random.permutation(len(self.eshiList)):
                yield self.eshiList[i]

    def next(self, m, n):
        return [next(self.getEshi).next(n) for x in range(m)]

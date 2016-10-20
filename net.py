#!/usr/bin/env python

# Author : Thibault Barbie

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import link
from chainer import reporter


class Regressor(link.Chain):
    def __init__(self, predictor,lossfun=F.mean_squared_error):
        super(Regressor, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        
    def __call__(self, *args):
        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]

        self.y = self.predictor(*x)
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)

        return self.loss

class Autoencoder(chainer.Chain):
    def __init__(self,n_units):
        super(Autoencoder, self).__init__(
            l1=L.Linear(784,n_units),
            l2=L.Linear(n_units,784),
        )

    def __call__(self, x):
        h = self.l1(x)
        h = F.sigmoid(self.l2(h))
        return h
    

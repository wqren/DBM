from __future__ import division
"""
Stochastic Gradient Descent and related functionality such as
learning rate adaptation, momentum, and Polyak averaging.

Modified from pylearn2.training_algorithms.sgd by Guillaume Desjardins,
to match:

"Learning Feature Hierarchies with Centered Deep Boltzmann Machines",
Gregoire Montavon, Klaus-Robert Muller.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow, David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow, David Warde-Farley"
__email__ = "goodfeli@iro"
from theano import function
from pylearn2.utils import sharedX
from pylearn2.training_callbacks.training_callback import TrainingCallback
from pylearn2.utils import serial

class PolyakAveraging(TrainingCallback):
    """
    See "A Tutorial on Stochastic Approximation Algorithms
    for Training Restricted Boltzmann Machines and
        Deep Belief Nets" by Kevin Swersky et al

    Notes: this is usually used with a fixed, rather than
        annealed learning rate.
        It may be used in conjunction with momentum.

    This functionality is still a work in progress. Currently,
    your model needs to implement "add_polyak_channels" to
    use it.

    The problem is that Polyak averaging shouldn't modify
    the model parameters. It should keep a second copy
    that it averages in the background. This second copy
    doesn't get to come back in and affect the learning process
    though.

    (IG tried having the second copy get pushed back into
    the model once per epoch, but this turned out to be
    harmful, at least in limited tests)

    So we need a cleaner interface for monitoring the
    averaged copy of the parameters, and we need to make
    sure the saved model at the end uses the averaged
    parameters, not the parameters used for computing
    the gradients during training.
    """

    def __init__(self, model, save_path = None, kc=10, save_freq = 1):
        self.__dict__.update(locals())

        updates = {}
        k = sharedX(0.)
        self.param_to_mean = {}
        for param in model.get_params():
            mean = sharedX(param.get_value())
            assert type(mean) == type(param)
            self.param_to_mean[param] = mean
            updates[mean] = k / (k + kc) * mean + kc / (k + kc) * param
            updates[k] = k + 1.
        self.avg = function([], updates = updates)
        self._count = 0
        self.kc = kc
        self.k = k

    def __call__(self, model, dataset, algorithm):
        if self._count > 0 and self._count % self.save_freq == 0:
            self.avg()
            saved_params = {}
            for param in model.get_params():
                saved_params[param] = param.get_value()
                param.set_value(self.param_to_mean[param].get_value())
            serial.save(self.save_path, model)
            for param in model.get_params():
                param.set_value(saved_params[param])
        self._count += 1


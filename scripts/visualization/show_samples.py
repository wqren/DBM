import sys
import numpy
import pylab as pl
import pickle
from optparse import OptionParser

from theano import function
import theano.tensor as T
import theano

from pylearn2.gui.patch_viewer import make_viewer
from pylearn2.gui.patch_viewer import PatchViewer
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial


parser = OptionParser()
parser.add_option('-m', '--model', action='store', type='string', dest='path')
parser.add_option('--width',  action='store', type='int', dest='width')
parser.add_option('--height', action='store', type='int', dest='height')
parser.add_option('--channels',  action='store', type='int', dest='chans')
(opts, args) = parser.parse_args()

def get_dims(nf):
    num_rows = numpy.floor(numpy.sqrt(nf))
    return (num_rows, numpy.ceil(nf / num_rows))

# load model and retrieve parameters
model = serial.load(opts.path)

samples = model.neg_ev.get_value()
viewer = make_viewer(samples, (5,10), (28,28))
pl.imshow(viewer.get_img())
pl.show()

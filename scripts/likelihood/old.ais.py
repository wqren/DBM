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
floatX = theano.config.floatX

def estimate_from_weights(log_ais_w):

    # estimate the log-mean of the AIS weights
    dlogz = log_mean(log_ais_w)

    # estimate log-variance of the AIS weights
    # VAR(log(X)) \approx VAR(X) / E(X)^2 = E(X^2)/E(X)^2 - 1
    m = numpy.max(log_ais_w)
    var_dlogz = (log_ais_w.shape[0] *
                 numpy.sum(numpy.exp(2 * (log_ais_w - m))) /
                 numpy.sum(numpy.exp(log_ais_w - m)) ** 2 - 1.)

    return dlogz, var_dlogz


parser = OptionParser()
parser.add_option('-m', '--model', action='store', type='string', dest='path')
parser.add_option('--dataset', action='store', type='string', dest='dataset')
parser.add_option('--large', action='store_true', dest='large')
(opts, args) = parser.parse_args()

# load model and retrieve parameters
model = serial.load(opts.path)


##########################
## BUILD THEANO FUNCTIONS
##########################
beta = T.scalar()

# build function to retrieve energy
E = model.energy(model.nsamples, beta)
energy_fn = theano.function([beta], E)

# build inference function
new_psamples = model.e_step(model.psamples, n_steps=model.pos_mf_steps)
inference_fn = function([model.input], new_psamples)

# build function to sample
updates = {}
new_nsamples = model.neg_sampling(model.nsamples, n_steps=model.neg_sample_steps, beta=beta)
for (nsample, new_nsample) in zip(model.nsamples, new_nsamples):
    updates[nsample] = new_nsample
sample_fn = function([beta], [], updates=updates, name='sample_func')

###########
## RUN AIS
###########

# generate exact sample for the base model
for i, nsample_i in enumerate(model.nsamples):
    if i == 0: 
        v0 = numpy.tile(1. / (1. + numpy.exp(-model.bias[0].get_value())), (model.batch_size, 1))
        v0 = numpy.array(v0 > numpy.random.random_sample(v0.shape), dtype=floatX)
        nsample_i.set_value(v0)
    else:
        temp = numpy.random.randint(0,2,size=(model.batch_size,model.n_u[i]))
        nsample_i.set_value(numpy.cast[floatX](temp))

# utility function for safely computing log-mean of the ais weights
ais_w = T.vector()
dlogz = T.log(T.mean(T.exp(ais_w - T.max(ais_w)))) + T.max(ais_w)
log_mean = theano.function([ais_w], dlogz, allow_input_downcast=False)

# initialize log-ais weights
log_ais_w = numpy.zeros(model.batch_size, dtype=floatX)

# default configuration for interpolating distributions
if not opts.large:
    betas = numpy.cast[floatX](
        numpy.hstack((numpy.linspace(0, 0.5, 1e3),
                     numpy.linspace(0.5, 0.9, 1e4),
                     numpy.linspace(0.9, 1.0, 1e4))))
else:
    betas = numpy.cast[floatX](
        numpy.hstack((numpy.linspace(0, 0.5, 1e4),
                     numpy.linspace(0.5, 0.9, 5e4),
                     numpy.linspace(0.9, 1.0, 5e4))))

for i in range(len(betas) - 1):

    bp, bp1 = betas[i], betas[i+1]

    # log-ratio of (free) energies for two nearby temperatures
    log_ais_w += energy_fn(bp) - energy_fn(bp1)

    sample_fn(bp1)

    if i % 1e3 == 0:
        print 'Temperature %f ' % bp1

dlogz, var_dlogz = estimate_from_weights(log_ais_w)
print 'dlogz = ', dlogz
print 'var_dlogz = ', var_dlogz

# default log-partition
log_za = numpy.sum(numpy.log(1 + numpy.exp(model.bias[0].get_value())))
for n_ui in model.n_u[1:]:
    log_za += n_ui * numpy.log(2)
log_z = log_za + dlogz

print 'log_za = ', log_za
print 'log_z = ', log_z
print 'var_dlogz = ',  var_dlogz

##############################
# COMPUTE TEST SET LIKELIHOOD
##############################
from pylearn2.datasets import mnist
assert opts.dataset in ['train','test']
data = mnist.MNIST(opts.dataset)

i = 0.
nll = 0

for i in xrange(0, len(data.X), model.batch_size):

    # recast data as floatX and apply preprocessing if required
    x = numpy.array(data.X[i:i + model.batch_size, :], dtype=floatX)

    # perform inference
    psamples = inference_fn(x)

    # entropy of h(q) adds contribution to variational lower-bound
    hq = 0
    for psample in psamples[1:]:
        temp = - psample * numpy.log(1e-5 + psample) - (1.-psample) * numpy.log(1. - psample + 1e-5)
        hq += numpy.sum(temp, axis=1)

    # copy into negative phase buffers to measure energy
    for ii, psample in enumerate(psamples):
        model.nsamples[ii].set_value(psample)

    # compute sum of likelihood for current buffer
    x_nll = numpy.sum(-energy_fn(1.0) + hq - log_z)

    # perform moving average of negative likelihood
    # divide by len(x) and not bufsize, since last buffer might be smaller
    nll = (i * nll + x_nll) / (i + len(x))

print 'Test set likelihood: ', nll

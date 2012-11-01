import os
import numpy
import pickle
from scipy import stats

import theano
import theano.tensor as T
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import function, shared

from pylearn2.training_algorithms import default
from pylearn2.utils import serial
from pylearn2.base import Block
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace

from DBM import tools
from DBM import cost as utils_cost
from DBM import minres
from DBM import natural
from DBM import sharedX, floatX, npy_floatX

class DBM(Model, Block):
    """Bilinear Restricted Boltzmann Machine (RBM)  """


    def __init__(self, input = None, n_u=[100,100], enable={},
            iscales=None, clip_min={}, clip_max={},
            pos_mf_steps=1, neg_sample_steps=1, 
            lr = 1e-3, lr_anneal_coeff=0, lr_timestamp=None, lr_mults = {},
            l1 = {}, l2 = {}, l1_inf={}, flags={},
            batch_size = 13,
            computational_bs = 0,
            compile=True,
            seed=1241234,
            my_save_path=None, save_at=None, save_every=None):
        """
        :param n_u: list, containing number of units per layer. n_u[0] contains number
         of visible units, while n_u[i] (with i > 0) contains number of hid. units at layer i.
        :param enable: dictionary of flags with on/off behavior
        :param iscales: optional dictionary containing initialization scale for each parameter.
               Key of dictionary should match the name of the associated shared variable.
        :param pos_mf_steps: number of mean-field iterations to perform in positive phase
        :param neg_sample_steps: number of sampling updates to perform in negative phase.
        :param lr: base learning rate
        :param lr_anneal_coeff: float. 0 for constant learning rate. Other values will anneal
        :      learning rate with profile: lr / (1 + lr_anneal_coeff * batch_index)
        :param lr_timestamp: list containing update indices at which to change the lr multiplier
        :param lr_mults: dictionary, optionally containing a list of learning rate multipliers
               for parameters of the model. Length of this list should match length of
               lr_timestamp (the lr mult will transition whenever we reach the associated
               timestamp). Keys should match the name of the shared variable, whose learning
               rate is to be adjusted.
        :param l1: dictionary, whose keys are model parameter names, and values are
               hyper-parameters controlling degree of L1-regularization.
        :param l2: same as l1, but for L2 regularization.
        :param l1_inf: same as l1, but the L1 penalty is centered as -\infty instead of 0.
        :param batch_size: size of positive and negative phase minibatch
        :param computational_bs: batch size used internaly by natural
               gradient to reduce memory consumption
        :param seed: seed used to initialize numpy and theano RNGs.
        :param my_save_path: if None, do not save model. Otherwise, contains stem of filename
               to which we will save the model (everything but the extension).
        :param save_at: list containing iteration counts at which to save model
        :param save_every: scalar value. Save model every `save_every` iterations.
        """
        Model.__init__(self)
        Block.__init__(self)

        self.depth = len(n_u)

        for (k,v) in clip_min.iteritems(): clip_min[k] = npy_floatX(v)
        for (k,v) in clip_max.iteritems(): clip_max[k] = npy_floatX(v)
        for i in xrange(self.depth):
            iscales.setdefault('bias%i'%i, 0.)
            iscales.setdefault('W%i'%i, 0.1)
        assert len(n_u) > 1

        # dump initialization parameters to object
        for (k,v) in locals().iteritems():
            if k!='self': setattr(self,k,v)

        # allocate random number generators
        self.rng = numpy.random.RandomState(seed)
        self.theano_rng = RandomStreams(self.rng.randint(2**30))

        ############### ALLOCATE PARAMETERS #################
        self.n_v = n_u[0]

        # allocate bilinear-weight matrices
        self.W = []
        self.bias = []
        self.nsamples = []
        self.psamples = []
        self.neg_ev = sharedX(self.rng.rand(batch_size, n_u[0]), name='neg_ev')
        self.input = T.matrix('input')
        self.computational_bs = computational_bs

        # configure input-space (?new pylearn2 feature?)
        self.input_space = VectorSpace(n_u[0])
        self.output_space = VectorSpace(n_u[-1])

        for i, nui in enumerate(n_u):

            self.bias += [sharedX(iscales['bias%i' %i] * numpy.ones(nui), name='bias%i'%i)]
            self.nsamples += [sharedX(self.rng.rand(batch_size, nui), name='nsamples%i'%i)]

            # parameter to keep track of biases

            if i > 0: 
                # declarate weight matrix and storage for positive phase statistics
                self.psamples += [sharedX(self.rng.rand(batch_size, nui), name='psamples%i'%i)]
                # weight matrix for each layer
                wv_val  = self.rng.randn(n_u[i-1], n_u[i]) * iscales.get('W%i'%i,1.0)
                self.W += [sharedX(wv_val, name='W%i' % i)]
            else:
                # define symbolic input, to serve as theano function input
                self.psamples += [self.input]
                self.input = self.psamples[0]
                self.W += [None]

        # learning rate - implemented as shared parameter for GPU
        self.lr_shrd = sharedX(lr, name='lr_shrd')
        self.lr_mults_it = {}
        self.lr_mults_shrd = {}
        for (k,v) in lr_mults.iteritems():
            # make sure all learning rate multipliers are float64
            self.lr_mults_it[k] = tools.HyperParamIterator(lr_timestamp, lr_mults[k])
            self.lr_mults_shrd[k] = sharedX(self.lr_mults_it[k].value, name='lr_mults_shrd'+k)

        # counters used by pylearn2 trainers
        self.batches_seen = 0                    # incremented on every batch
        self.examples_seen = 0                   # incremented on every training example
        self.force_batch_size = batch_size       # force minibatch size

        self.error_record = []

        ## ESTABLISH LIST OF LEARNT MODEL PARAMETERS ##
        self.params = []
        for i in xrange(self.depth):   
            self.params += [self.bias[i]]
            if i > 0: 
                self.params += [self.W[i]]

        if compile: self.do_theano()


    def do_theano(self):
        """ Compiles all theano functions needed to use the model"""

        init_names = dir(self)

        ###### All fields you don't want to get pickled (e.g., theano functions) should be created below this line

        ###
        # SAMPLING: NEGATIVE PHASE
        ###
        new_nsamples = self.neg_sampling(self.nsamples, n_steps=self.neg_sample_steps)
        new_ev = self.hi_given(new_nsamples, 0)
        neg_updates = {self.neg_ev: new_ev}
        for (nsample, new_nsample) in zip(self.nsamples, new_nsamples):
            neg_updates[nsample] = new_nsample
        self.sample_neg_func = function([], [],
                updates=neg_updates,
                name='sample_neg_func',
                profile=0)

        ###
        # POSITIVE PHASE ESTEP + LEARNING
        ###
        pos_updates = {}

        new_psamples = self.e_step(self.psamples, n_steps=self.pos_mf_steps)
        for (psample, new_psample) in zip(self.psamples[1:], new_psamples[1:]):
            pos_updates[psample] = new_psample

        ml_cost = self.ml_cost(new_psamples, self.nsamples)
        ml_cost.compute_gradients()
        reg_cost = self.get_reg_cost()
        if self.flags.get('enable_natural', False):
            self.get_natural_direction(ml_cost, self.nsamples)
        learning_grads = utils_cost.compute_gradients(ml_cost, reg_cost)

        ##
        # COMPUTE GRADIENTS WRT. TO ALL COSTS
        ##
        learning_updates = utils_cost.get_updates(             
                learning_grads,
                self.lr_shrd,
                multipliers = self.lr_mults_shrd) 
        learning_updates.update(pos_updates)
      
        # build theano function to train on a single minibatch
        self.batch_train_func = function([self.input], [],
                updates=learning_updates,
                name='train_rbm_func',
                profile=0)

        ##
        # CONSTRAINTS
        ##
        constraint_updates = {}

        ## clip parameters to maximum values (if applicable)
        for (k,v) in self.clip_max.iteritems():
            assert k in [param.name for param in self.params]
            param = getattr(self, k)
            constraint_updates[param] = T.clip(param, param, v)

        ## clip parameters to minimum values (if applicable)
        for (k,v) in self.clip_min.iteritems():
            assert k in [param.name for param in self.params]
            for p in self.params:
                if p.name == k:
                    break
            constraint_updates[p] = T.clip(constraint_updates.get(p, p), v, p)

        self.enforce_constraints = theano.function([],[], updates=constraint_updates)

        ###### All fields you don't want to get pickled should be created above this line
        final_names = dir(self)
        self.register_names_to_del( [ name for name in (final_names) if name not in init_names ])

        # Before we start learning, make sure constraints are enforced
        self.enforce_constraints()

    def set_batch_size(self, batch_size):
        """
        Change the batch size of a model which has already been initialized.
        :param batch_size: int. new batch size.
        """
        self.neg_ev = sharedX(self.rng.rand(batch_size, self.n_u[0]), name='neg_ev')
        self.psamples = []
        self.nsamples = []

        for i, nui in enumerate(self.n_u):
            self.nsamples += [sharedX(self.rng.rand(batch_size, nui), name='nsamples%i'%i)]
            if i > 0:
                self.psamples += [sharedX(self.rng.rand(batch_size, nui), name='psamples%i'%i)]
            else:
                self.psamples += [self.input]

        self.force_batch_size = batch_size       # force minibatch size
        self.do_theano()

    def train_batch(self, dataset, batch_size):
        """
        Performs one-step of gradient descent, using the given dataset.
        :param dataset: Pylearn2 dataset to train the model with.
        :param batch_size: int. Batch size to use.
               HACK: this has to match self.batch_size.
        """

        # First-layer biases of RBM-type models should always be initialized to the log-odds
        # ratio. This ensures that the weights don't attempt to learn the mean.
        if self.batches_seen == 0:
            mean_x = numpy.mean(dataset.X, axis=0)
            clip_x = numpy.clip(mean_x, 1e-5, 1-1e-5)
            self.bias[0].set_value(numpy.log(clip_x / (1. - clip_x)))

        x = dataset.get_batch_design(batch_size, include_labels=False)
        self.learn_mini_batch(x)

        # accounting...
        self.examples_seen += self.batch_size
        self.batches_seen += 1

        # modify learning rate multipliers
        for (k, iter) in self.lr_mults_it.iteritems():
            if iter.next():
                print 'self.batches_seen = ', self.batches_seen
                self.lr_mults_shrd[k].set_value(iter.value)
                print 'lr_mults_shrd[%s] = %f' % (k,iter.value)

        self.enforce_constraints()

        # save to different path each epoch
        if self.my_save_path and \
           (self.batches_seen in self.save_at or
            self.batches_seen % self.save_every == 0):

            fname = self.my_save_path + '_e%i.pkl' % (self.batches_seen)
            print 'Saving to %s ...' %fname,
            serial.save(fname, self)
            print 'done'

        return True


    def learn_mini_batch(self, x):
        """
        Performs the substeps involed in one iteration of PCD/SML. We first adapt the learning
        rate, generate new negative samples from our persistent chain and then perform a step
        of gradient descent.
        :param x: numpy.ndarray. mini-batch of training examples, of shape (batch_size, self.n_u[0])
        """

        # anneal learning rate
        self.lr_shrd.set_value(self.lr / (1. + self.lr_anneal_coeff * self.batches_seen))

        # perform negative phase sampling
        self.sample_neg_func()

        # update parameters
        self.batch_train_func(x)


    def energy(self, samples, beta=1.0):
        """
        Computes energy for a given configuration of visible and hidden units.
        :param samples: list of T.matrix of shape (batch_size, n_u[i])
        samples[0] represents visible samples.
        """
        energy = - T.dot(samples[0], self.bias[0]) * beta
        for i in xrange(1, self.depth):
            energy -= T.sum(T.dot(samples[i-1], self.W[i] * beta) * samples[i], axis=1)
            energy -= T.dot(samples[i], self.bias[i] * beta)

        return energy

    ######################################
    # MATH FOR CONDITIONAL DISTRIBUTIONS #
    ######################################

    def hi_given(self, samples, i, beta=1.0, apply_sigmoid=True):
        """
        Compute the state of hidden layer i given all other layers.
        :param samples: list of tensor-like objects. For the positive phase, samples[0] is
        points to self.input, while samples[i] contains the current state of the i-th layer. In
        the negative phase, samples[i] contains the persistent chain associated with the i-th
        layer.
        :param i: int. Compute activation of layer i of our DBM.
        :param beta: used when performing AIS.
        :param apply_sigmoid: when False, hi_given will not apply the sigmoid. Useful for AIS
        estimate.
        """
        hi_mean = 0.

        if i < self.depth-1:
            # top-down input
            wip1 = self.W[i+1]
            hi_mean += T.dot(samples[i+1], wip1.T) * beta

        if i > 0:
            # bottom-up input
            wi = self.W[i]
            hi_mean += T.dot(samples[i-1], wi) * beta

        hi_mean += self.bias[i] * beta

        if apply_sigmoid:
            return T.nnet.sigmoid(hi_mean)
        else:
            return hi_mean

    def sample_hi_given(self, samples, i, beta=1.0):
        """
        Given current state of our DBM (`samples`), sample the values taken by the i-th layer.
        See self.hi_given for detailed description of parameters.
        """

        hi_mean = self.hi_given(samples, i, beta)

        hi_sample = self.theano_rng.binomial(
                size = (self.batch_size, self.n_u[i]),
                n=1, p=hi_mean, 
                dtype=floatX)

        return hi_sample

 
    ##################
    # SAMPLING STUFF #
    ##################

    def e_step(self, psamples, n_steps=5):
        """
        Performs `n_steps` of mean-field inference (used to compute positive phase statistics).
        :param psamples: list of tensor-like objects, representing the state of each layer of
        the DBM (during the inference process). psamples[0] points to self.input.
        :param n_steps: number of iterations of mean-field to perform.
        """
        v = psamples[0]

        # initialize hidden layers
        new_psamples = [psamples[0]] + [None for i in xrange(1,self.depth)]
        for i in xrange(1,self.depth):
            new_psamples[i] = T.ones((v.shape[0],self.n_u[i]))

        # now alternate mean-field inference for even/odd layers
        for si in xrange(n_steps):
            for i in xrange(1,self.depth,2):
                new_psamples[i] = self.hi_given(new_psamples, i)
            for i in xrange(2,self.depth,2):
                new_psamples[i] = self.hi_given(new_psamples, i)

        return new_psamples

    def neg_sampling(self, nsamples, n_steps, beta=1.0):
        """
        Perform `n_steps` of block-Gibbs sampling (used to compute negative phase statistics).
        This method alternates between sampling of odd given even layers, and vice-versa.
        :param nsamples: list (of length len(self.n_u)) of tensor-like objects, representing
        the state of the persistent chain associated with layer i.
        """

        new_nsamples = [nsamples[i] for i in xrange(self.depth)]

        for si in xrange(n_steps):
            for i in xrange(1,self.depth,2):
                new_nsamples[i] = self.sample_hi_given(new_nsamples, i, beta)
            for i in xrange(0,self.depth,2):
                new_nsamples[i] = self.sample_hi_given(new_nsamples, i, beta)

        return new_nsamples

    def ml_cost(self, psamples, nsamples):
        """
        Variational approximation to the maximum likelihood positive phase.
        :param v: T.matrix of shape (batch_size, n_v), training examples
        :return: tuple (cost, gradient)
        """
        pos_cost = T.sum(self.energy(psamples))
        neg_cost = T.sum(self.energy(nsamples))
        batch_cost = pos_cost - neg_cost
        cost = batch_cost / self.batch_size

        cte = psamples + nsamples
        return utils_cost.Cost(cost, self.params, cte)


    def monitor_stats(self, b, axis=(0,1), name=None, track_min=True, track_max=True):
        if name is None: assert hasattr(b, 'name')
        name = name if name else b.name

        channels = {name + '.mean': T.mean(b, axis=axis)}
        if track_min: channels[name + '.min'] = T.min(b, axis=axis)
        if track_max: channels[name + '.max'] = T.max(b, axis=axis)
        
        return channels

    def get_monitoring_channels(self, x, y=None):
        chans = {}

        for i in xrange(self.depth):
            chans.update(self.monitor_stats(self.bias[i], axis=(0,)))
            chans.update(self.monitor_stats(self.nsamples[i]))

        for i in xrange(1, self.depth):
            chans.update(self.monitor_stats(self.W[i]))
            chans.update(self.monitor_stats(self.psamples[i]))
            norm_wi = T.sqrt(T.sum(self.W[i]**2, axis=0))
            chans.update(self.monitor_stats(norm_wi, axis=(0,), name='norm_w%i'%i))

        return chans

    ##############################
    # GENERIC OPTIMIZATION STUFF #
    ##############################
    def get_reg_cost(self):
        """
        Builds the symbolic expression corresponding to first-order gradient descent
        of the cost function ``cost'', with some amount of regularization defined by the other
        parameters.
        :param l2: dict containing amount of L2 regularization for Wg, Wh and Wv
        :param l1: dict containing amount of L1 regularization for Wg, Wh and Wv
        :param l1_inf: dict containing amount of L1 (centered at -inf) reg for Wg, Wh and Wv
        """
        cost = 0.
        params = []

        for p in self.params:

            if self.l1.has_key(p.name):
                cost += self.l1[p.name] * T.sum(abs(p))
                params += [p]

            if self.l1_inf.has_key(p.name):
                cost += self.l1_inf[p.name] * T.sum(p)
                params += [p]

            if self.l2.has_key(p.name):
                cost += self.l2[p.name] * T.sum(p**2)
                params += [p]

        return utils_cost.Cost(cost, params)

    def get_natural_direction(self, ml_cost, nsamples):
        assert self.depth == 3
        inputs = [ml_cost.grads[self.W[1]],
                  ml_cost.grads[self.W[2]],
                  ml_cost.grads[self.bias[0]],
                  ml_cost.grads[self.bias[1]],
                  ml_cost.grads[self.bias[2]]]
        if self.computational_bs > 0:
            def Lx_func(xw1, xw2, xbias0, xbias1, xbias2):
                return natural.compute_Lx_batches(
                        self.nsamples[0],
                        self.nsamples[1],
                        self.nsamples[2],
                        xw1, xw2, xbias0, xbias1, xbias2,
                        self.force_batch_size, self.computational_bs)
        else:
            def Lx_func(xw1, xw2, xbias0, xbias1, xbias2):
                return natural.compute_Lx(
                        self.nsamples[0],
                        self.nsamples[1],
                        self.nsamples[2],
                        xw1, xw2, xbias0, xbias1, xbias2)

        rvals = minres.minres(
                Lx_func,
                inputs,
                rtol=1e-4,
                damp = 0.01,
                maxit = 80,
                profile=0)
        newgrads = rvals[0]
        self.flag = rvals[1]
        self.niters = rvals[2]
        self.rel_residual = rvals[3]
        self.rel_Aresidual = rvals[4]
        self.Anorm = rvals[5]
        self.Acond = rvals[6]
        self.xnorm = rvals[7]
        self.Axnorm = rvals[8]


        ml_cost.grads[self.W[1]] = newgrads[0]
        ml_cost.grads[self.W[2]] = newgrads[1]
        ml_cost.grads[self.bias[0]] = newgrads[2]
        ml_cost.grads[self.bias[1]] = newgrads[3]
        ml_cost.grads[self.bias[2]] = newgrads[4]

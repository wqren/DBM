import numpy
import pickle

from pylearn2.training_callbacks.training_callback import TrainingCallback

from DBM.scripts.likelihood import ais

class pylearn2_ais_callback(TrainingCallback):

    def __init__(self, trainset, testset,
                 switch_threshold=None,
                 switch_at=None,
                 ais_interval=10):

        self.trainset = trainset
        self.testset = testset
        self.switch_threshold = switch_threshold
        self.switch_at = switch_at
        self.has_switched = False
        self.ais_interval = ais_interval

        self.pkl_results = {
                'epoch': [],
                'batches_seen': [],
                'cpu_time': [],
                'train_ll': [],
                'test_ll': [],
                'logz': [],
                }

        self.jobman_results = {
                'best_epoch': 0,
                'best_batches_seen': 0,
                'best_cpu_time': 0,
                'best_train_ll': -numpy.Inf,
                'best_test_ll': -numpy.Inf,
                'best_logz': 0,
                'switch_epoch': 0,
                }
        fp = open('ais_callback.log','w')
        fp.write('Epoch\tBatches\tCPU\tTrain\tTest\tlogz\n')
        fp.close()

    def switch_to_full_natural(self, model):
        model.switch_to_full_natural()
        self.jobman_results['switch_epoch'] = model.epochs
        self.has_switched = True
        self.ais_interval = 1

    def __call__(self, model, train, algorithm):

        if self.switch_at and (not self.has_switched) and model.epochs >= self.switch_at:
            self.switch_to_full_natural(model)

        # measure AIS periodically
        if (model.epochs % self.ais_interval) == 0:
            (train_ll, test_ll, logz) = ais.estimate_likelihood(model,
                        self.trainset, self.testset, large_ais=False)
            if self.switch_threshold and model.epochs > 0 and (not self.has_switched):
                improv = train_ll - self.jobman_results['train_ll']
                if improv < abs(self.switch_threshold * self.jobman_results['train_ll']):
                    self.switch_to_full_natural(model)
            self.log(model, train_ll, test_ll, logz)

            if model.jobman_channel:
                model.jobman_channel.save()

    def log(self, model, train_ll, test_ll, logz):

        # log to database
        self.jobman_results['epoch'] = model.epochs
        self.jobman_results['batches_seen'] = model.batches_seen
        self.jobman_results['cpu_time'] = model.cpu_time
        self.jobman_results['train_ll'] = train_ll
        self.jobman_results['test_ll'] = test_ll
        self.jobman_results['logz'] = logz
        if train_ll > self.jobman_results['best_train_ll']:
            self.jobman_results['best_epoch'] = self.jobman_results['epoch']
            self.jobman_results['best_batches_seen'] = self.jobman_results['batches_seen']
            self.jobman_results['best_cpu_time'] = self.jobman_results['cpu_time']
            self.jobman_results['best_train_ll'] = self.jobman_results['train_ll']
            self.jobman_results['best_test_ll'] = self.jobman_results['test_ll']
            self.jobman_results['best_logz'] = self.jobman_results['logz']
        model.jobman_state.update(self.jobman_results)

        # save to text file
        fp = open('ais_callback.log','a')
        fp.write('%i\t%i\t%f\t%f\t%f\t%f\n' % (
            self.jobman_results['epoch'],
            self.jobman_results['batches_seen'],
            self.jobman_results['cpu_time'],
            self.jobman_results['train_ll'],
            self.jobman_results['test_ll'],
            self.jobman_results['logz']))
        fp.close()

        # save to pickle file
        self.pkl_results['epoch'] += [model.epochs]
        self.pkl_results['batches_seen'] += [model.batches_seen]
        self.pkl_results['cpu_time'] += [model.cpu_time]
        self.pkl_results['train_ll'] += [train_ll]
        self.pkl_results['test_ll'] += [test_ll]
        self.pkl_results['logz'] += [logz]
        fp = open('ais_callback.pkl','w')
        pickle.dump(self.pkl_results, fp)
        fp.close()



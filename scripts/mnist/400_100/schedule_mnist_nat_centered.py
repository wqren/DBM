from jobman.tools import DD, flatten
from jobman import api0, sql
import numpy

from pylearn2.scripts.jobman import experiment

rng = numpy.random.RandomState(4312987)

if __name__ == '__main__':
    db = api0.open_db('postgres://gershwin.iro.umontreal.ca/desjagui_db/aistats12_mnist_nat')

    state = DD()

    state.yaml_template = """
            !obj:pylearn2.scripts.train.Train {
                "max_epochs": 250,
                "save_freq": 10,
                "save_path": "dbm",
                # dataset below is now (temporarily) obsolete
                "dataset": &data !obj:pylearn2.datasets.mnist.MNIST {
                    "which_set": 'train',
                    "shuffle": True,
                    "one_hot": True,
                    "binarize": %(binarize)i,
                },
                "model": &model !obj:DBM.dbm.DBM {
                    "seed" : 123141,
                    "n_u": [784, %(nu1)i, %(nu2)i],
                    "lr": %(lr)f,
                    "flags": {
                        'enable_natural': %(enable_natural)i,
                        'enable_centering': %(enable_centering)i,
                        'enable_natural_diag': %(enable_natural_diag)i,
                        'enable_warm_start': %(enable_warm_start)i,
                        'precondition': None,
                        'mlbiases': %(mlbiases)i,
                    },
                    "lr_timestamp": [0.],
                    "lr_mults": {},
                    "pos_mf_steps": %(pos_mf_steps)i,
                    "pos_sample_steps": %(pos_sample_steps)i,
                    "neg_sample_steps": %(neg_sample_steps)i,
                    "iscales": {
                        'W1': %(iscale_w1)f,
                        'W2': %(iscale_w2)f,
                        'bias0': %(iscale_bias0)f,
                        'bias1': %(iscale_bias1)f,
                        'bias2': %(iscale_bias2)f,
                    },
                    "clip_min": {},
                    "l1": {
                        'W1': %(l1_W1)f,
                        'W2': %(l1_W2)f,
                    },
                    "batch_size": %(batch_size)i,
                    "computational_bs": 0,
                    "minres_params": {
                        'rtol': %(rtol)f,
                        'damp': %(damp)f,
                        'maxit': %(maxit)i,
                    },
                    "my_save_path": 'dbm',
                    "save_at": [],
                    "save_every": 50000,
                },
                "algorithm": !obj:pylearn2.training_algorithms.default.DefaultTrainingAlgorithm {
                           "batches_per_iter" : 1000,
                           "monitoring_batches": 1,
                           "monitoring_dataset": *data,
                },
                "callbacks": [
                    !obj:DBM.polyak.PolyakAveraging {
                        "model": *model,
                        "save_path": 'polyak_dbm.pkl',
                        "save_freq": 1,
                        "kc": 10,
                    },
                    !obj:DBM.scripts.likelihood.ais.pylearn2_ais_callback {
                        "trainset": *data,
                        "testset": !obj:pylearn2.datasets.mnist.MNIST {
                            "which_set": 'test',
                            "one_hot": True,
                            "binarize": True,
                        },
                        "switch_threshold": %(switch_threshold)f,
                        "switch_at": %(switch_at)i,
                        "ais_interval": %(ais_interval)i,
                    }
                ]
            }
    """

    njobs = 100
    for lr in [5e-3, 1e-3, 1e-4]:
        for batch_size in [25, 128, 256]:
            for (pos_mf_steps, pos_sample_steps) in [(5,0),(0,5)]:
                for iscale_bias in [0]:
                    for damp in [0.1]:
                        state.hyper_parameters = {
                            'binarize': 1,
                            'nu1': 400,
                            'nu2': 100,
                            'lr': lr,
                            'enable_natural': 1,
                            'enable_natural_diag': 0,
                            'enable_centering': 1,
                            'enable_warm_start': 0,
                            'mlbiases': 1,
                            'pos_mf_steps': pos_mf_steps,
                            'pos_sample_steps': pos_sample_steps,
                            'neg_sample_steps': 5,
                            'iscale_w1': 0.,
                            'iscale_w2': 0.,
                            'iscale_bias0': iscale_bias,
                            'iscale_bias1': iscale_bias,
                            'iscale_bias2': iscale_bias,
                            'l1_W1': 0.,
                            'l1_W2': 0.,
                            'batch_size': batch_size,
                            'rtol': 0.00001,
                            'damp': damp,
                            'maxit': 80,
                            'switch_threshold': 0,
                            'switch_at': 0,
                            'ais_interval': 1,
                        }
                        
                        sql.insert_job(
                                experiment.train_experiment,
                                flatten(state),
                                db,
                                force_dup=True)

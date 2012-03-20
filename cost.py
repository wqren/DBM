import theano.tensor as T

class Cost():

    def __init__(self, cost, params, constants=None):
        self.cost = cost
        self.grads = {}

        self.params = {}
        for p in params:
            self.params[p] = True

        self.constants = {}
        constants = [] if constants is None else constants
        for c in constants:
            self.constants[c] = True

    def combine(self, new_cost):
        self.cost += new_cost.cost
        self.params.update(new_cost.params)
        self.constants.update(new_cost.constants)

    def compute_gradients(self):
        grads =  T.grad(self.cost, self.params.keys(), 
                        consider_constant=self.constants.keys())
        for param, gparam in zip(self.params.keys(), grads):
            self.grads[param] = gparam

    def get_updates(self, base_lr, multipliers=None):
        """
        Returns an updates dictionary corresponding to a single step of SGD. The learning rate
        for each parameter is computed as base_lr * multipliers[param]
        :param base_lr: base learning rate (common to all parameters)
        :param multipliers: dictionary of learning rate multipliers, each being a shared var
                            e.g. {'hbias': sharedX(0.1), 'Wf': sharedX(0.01)}
        """

        updates = {}
        multipliers = {} if multipliers is None else multipliers

        for (param, gparam) in self.grads.iteritems():

            # each parameter can have its own multiplier on the learning rate
            multiplier = multipliers.get(param.name, 1.0)
            lr_param   = base_lr * multiplier
            # perform SGD
            updates[param] = param - lr_param * gparam

        return updates



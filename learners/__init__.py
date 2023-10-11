from math import log2
from itertools import count, chain
from typing import Hashable, Sequence, Mapping, Any

from coba.random import CobaRandom
from coba.learners import VowpalMediator

MemVal = Any

class VWC:

    def __init__(self, args:str, weight:bool) -> None:
        self._vw_args = args
        self._vw = VowpalMediator()
        self._rng = CobaRandom(1)
        self._weight = weight

    @property
    def params(self) -> Mapping[str,Any]:
        return {'args': self._vw_args, 'w':self._weight}

    def __reduce__(self):
        return (VWC, (self._vw_args,self._weight))

    def set_params(self,actions):
        if not self._vw.is_initialized:
            self._act = list(actions)
            self._map = dict(chain(zip(count(1),actions), zip(actions,count(1))))
            self._vw.init_learner(str.format(self._vw_args,len(actions)), label_type=2)

    def predict(self, features: Mapping) -> MemVal:
        pr = self._vw.predict(self._vw.make_example(features, None))
        return self._map[pr] if pr != 0 else self._rng.choice(self._act)

    def learn(self, features: Mapping, value: Any, weight: float):
        self._vw.learn(self._vw.make_example(features, f"{self._map[value]} {weight if self._weight else 1}"))

class EMT(VWC):

    def __init__(self, split:int = 100, scorer:str="self_consistent_rank", router:str="eigen", bound:int=0, interactions: Sequence[str]=[], weight:bool = False, rng : int = 1337) -> None:

        self._args = (split, scorer, router, bound, interactions, weight, rng)

        vw_args = [
            "--emt",
            f"--emt_tree {bound}",
            f"--emt_leaf {split}",
            f"--emt_scorer {scorer}",
            f"--emt_router {router}",
            "--min_prediction 0",
            "--max_prediction 3",
            "--coin",
            "--noconstant",
            f"-b {26}",
            "--initial_weight 0",
            *[ f"--interactions {i}" for i in interactions ],
        ]

        super().__init__(f"{' '.join(vw_args)} --quiet --random_seed {rng}", weight)

    def __reduce__(self):
        return (EMT, self._args)

    @property
    def params(self) -> Mapping[str,Any]:
        keys = ['split', 'scorer', 'router', 'bound', 'X', 'w']
        return { 'type':'EMT', **dict(zip(keys,self._args))}

class CMT(VWC):

    def __init__(self, mems_per_leaf:int=100, dream_repeats:int=5, alpha:float=0.5, lr:int=.001, coin:bool = True, interactions: Sequence[str]=[], max_nodes:int=100, learn_at_leaf:bool=True, weight:bool = False, rng : int = 1337) -> None:

        self._init_args = (mems_per_leaf, dream_repeats, alpha, lr, coin, interactions, max_nodes, learn_at_leaf, weight, rng)

        #leaf splits when 
            # n_leaf_examples >= leaf_example_multiplier*log2(tree->max_nodes)
        #and
            # current number of node < max_nodes-2

        vw_args = [
            f"--memory_tree {max_nodes}",
            "--learn_at_leaf" if learn_at_leaf else "",
            "--online",
            f"-l {lr}",
            "--max_number_of_labels {}",
            f"--leaf_example_multiplier {int(mems_per_leaf/log2(max_nodes))}",
            f"--dream_repeats {dream_repeats}",
            f"--alpha {alpha}",
            f"--power_t {0}",
            f"-b {26}",
            *[ f"--interactions {i}" for i in interactions ]
        ]

        if coin: vw_args.append("--coin")
        super().__init__(f"{' '.join(vw_args)} --quiet --random_seed {rng}", weight)

    def __reduce__(self):
        return (CMT, self._init_args)

    @property
    def params(self) -> Mapping[str,Any]:
        keys = ['split','dreams','alpha','lr','coin','X','nodes','lrn_at_leaf']
        return { 'type':'CMT', **dict(zip(keys,self._init_args)) }

class EpisodicLearner:

    def __init__(self, epsilon: float, mem: VWC) -> None:

        assert 0 <= epsilon and epsilon <= 1

        self._epsilon = epsilon
        self._i       = 0
        self._mem     = mem

    @property
    def params(self) -> Mapping[str,Any]:
        return { 'family': 'EpisodicLearner','e':self._epsilon, **self._mem.params }

    def predict(self, context: Hashable, actions: Sequence[Hashable]) -> Sequence[float]:
        """Choose which action index to take."""

        if self._i == 0:
            self._mem.set_params(['0','1'])

        self._i += 1

        rewards = [int(self._mem.predict({'x':context,'a':a})) for a in actions]

        greedy_r = -float('inf')
        greedy_A = []

        for action, mem_value in zip(actions, rewards):

            mem_value = mem_value or 0

            if mem_value == greedy_r:
                greedy_A.append(action)

            if mem_value > greedy_r:
                greedy_r = mem_value
                greedy_A = [action]

        min_p = self._epsilon / len(actions)
        grd_p = (1-self._epsilon)/len(greedy_A)

        return [ grd_p+min_p if a in greedy_A else min_p for a in actions ], {'actions': actions}

    def learn(self, context: Hashable, action: Hashable, reward: float, probability: float, actions: Sequence[Hashable]) -> None:
        """Learn about the result of an action that was taken in a context."""
        self._mem.learn({'x':context,'a':action}, str(reward), weight=1./(len(actions)*probability))

class StackedLearner:

    def __init__(self, epsilon: float, mem: EMT, X:str, coin:bool, constant:bool) -> None:

        assert 0 <= epsilon and epsilon <= 1

        self._epsilon = epsilon
        self._i       = 0
        self._mem     = mem
        self._args    = (X, coin, constant)

        if X == 'xa':
            args = f"--quiet --cb_explore_adf --epsilon {epsilon} --ignore_linear x --interactions xa --random_seed {1}"

        if X == 'xxa':
            args = f"--quiet --cb_explore_adf --epsilon {epsilon} --ignore_linear x --interactions xa --interactions xxa --random_seed {1}"

        if coin: 
            args += ' --coin'

        if not constant:
            args += " --noconstant"

        self._vw = VowpalMediator().init_learner(args,4)

    @property
    def params(self) -> Mapping[str,Any]:
        return { 'family': 'ComboLearner', 'e': self._epsilon, **self._mem.params, "other": self._args }

    def predict(self, context: Hashable, actions: Sequence[Hashable]) -> Sequence[float]:
        """Choose which action index to take."""

        if self._i == 0:
            self._mem.set_params(['0','1'])

        self._i += 1

        memories = [ int(self._mem.predict({'x':context,'a':a})) for a in actions ]
        adfs     = [ {'a':a, 'm':m }  for a,m in zip(actions,memories) ]
        probs    = self._vw.predict(self._vw.make_examples({'x':context}, adfs, None))

        return probs, {'adfs':adfs, 'actions':actions}

    def learn(self, context: Hashable, action: Hashable, reward: float, probability: float, adfs: Any, actions: Sequence[Hashable]) -> None:
        """Learn about the result of an action that was taken in a context."""

        self._mem.learn({'x':context,'a':action}, str(reward), weight=1./(len(actions)*probability))
        labels = self._labels(actions, action, reward, probability)
        self._vw.learn(self._vw.make_examples({'x':context}, adfs, labels))

    def _labels(self,actions,action,reward:float,prob:float):
        return [ f"{i+1}:{round(-reward,5)}:{round(prob,5)}" if a == action else None for i,a in enumerate(actions)]

    def __reduce__(self):
        return (type(self), (self._epsilon, self._mem, *self._args))

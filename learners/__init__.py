import time
from typing import Hashable, Sequence, Mapping, Any

from coba.learners import VowpalMediator

logn = 500

MemVal = Any

class MemoryKey:
    __slots__=('x','a','_hash')
    def __init__(self, context, action) -> None:

        self.x = context
        self.a = action

        self._hash = hash((context,action))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, MemoryKey) and self.x == __o.x and self.a == __o.a

class EMT:

    def __init__(self, split:int = 100, scorer:int=3, router:int=2, bound:int=0, min:float=0, interactions: Sequence[str]=[], rng : int = 1337) -> None:

        self._args = (split, scorer, router, bound, min, interactions, rng)

        vw_args = [
            "--eigen_memory_tree",
            f"--emt_tree {bound}",
            f"--emt_leaf {split}",
            f"--emt_scorer {scorer}",
            f"--emt_router {router}",
            f"--min_prediction {min}",
            "--max_prediction 3",
            "--coin",
            "--noconstant",
            f"--power_t {0}",
            "--loss_function squared",
            f"-b {26}",
            "--initial_weight 0",
            *[ f"--interactions {i}" for i in interactions ],
            "--quiet"
        ]

        init_args = f"{' '.join(vw_args)} --quiet --random_seed {rng}"
        label_type = 2

        self._vw = VowpalMediator().init_learner(init_args, label_type)

    def __reduce__(self):
        return (EMT, self._args)
    
    @property
    def params(self) -> Mapping[str,Any]:
        keys = ['split', 'scorer', 'router', 'bound', 'min', 'X']
        return { 'type':'EMT', **dict(zip(keys,self._args))}

    def predict(self, key: MemoryKey) -> MemVal:
        ex = self._vw.make_example({'x': key.x, 'a': key.a}, None)
        pr = self._vw.predict(ex)
        return pr

    def learn(self, key: MemoryKey, value: MemVal, weight: float):
        self._vw.learn(self._vw.make_example({'x': key.x, 'a': key.a}, f"{value} {weight}"))

class CMT:

    def __init__(self, n_nodes:int=100, leaf_multiplier:int=15, dream_repeats:int=5, alpha:float=0.5, coin:bool = True, interactions: Sequence[str]=[], rng : int = 1337) -> None:

        self._args = (n_nodes, leaf_multiplier, dream_repeats, alpha, coin, interactions, rng)

        vw_args = [
            f"--memory_tree {n_nodes}",
            "--learn_at_leaf",
            "--online",
            f"--leaf_example_multiplier {leaf_multiplier}",
            f"--dream_repeats {dream_repeats}",
            f"--alpha {alpha}",
            f"--power_t {0}",
            f"-b {25}",
            "--quiet",
            *[ f"--interactions {i}" for i in interactions ]
        ]

        if coin: vw_args.append("--coin")

        init_args = f"{' '.join(vw_args)} --quiet --random_seed {rng}"
        label_type = 2

        self._vw = VowpalMediator().init_learner(init_args, label_type)

    def __reduce__(self):
        return (CMT, self._args)

    @property
    def params(self) -> Mapping[str,Any]:
        keys = ['nodes','multiplier','dreams','alpha','coin','X']
        return { 'type':'CMT', **dict(zip(keys,self._args)) }

    def predict(self, key: MemoryKey) -> MemVal:
        ex = self._vw.make_example({'x': key.x, 'a': key.a}, None)
        pr = self._vw.predict(ex)
        return pr

    def learn(self, key: MemoryKey, value: MemVal, weight: float):
        self._vw.learn(self._vw.make_example({'x': key.x, 'a': key.a}, f"{value} {weight}"))

class EpisodicLearner:

    def __init__(self, epsilon: float, cmt: EMT) -> None:

        assert 0 <= epsilon and epsilon <= 1

        self._epsilon = epsilon
        self._i       = 0
        self._cmt     = cmt
        self._times   = [0, 0]

    @property
    def params(self) -> Mapping[str,Any]:
        return { 'family': 'EpisodicLearner','e':self._epsilon, **self._cmt.params }

    def predict(self, context: Hashable, actions: Sequence[Hashable]) -> Sequence[float]:
        """Choose which action index to take."""

        self._i += 1

        if logn and self._i % logn == 0:
            print(f"MEM {self._i}. avg prediction time {round(self._times[0]/self._i,2)}")
            print(f"MEM {self._i}. avg learn      time {round(self._times[1]/self._i,2)}")

        predict_start = time.time()

        rewards = [ self._cmt.predict(MemoryKey(context, a)) for a in actions]

        greedy_r = -float('inf')
        greedy_A = []

        for action, mem_value in zip(actions, rewards):

            mem_value = mem_value or 0

            if mem_value == greedy_r:
                greedy_A.append(action)

            if mem_value > greedy_r:
                greedy_r = mem_value
                greedy_A = [action]

        self._times[0] += time.time()-predict_start

        min_p = self._epsilon / len(actions)
        grd_p = (1-self._epsilon)/len(greedy_A)

        return [ grd_p+min_p if a in greedy_A else min_p for a in actions ]

    def learn(self, context: Hashable, actions: Sequence[Hashable], action: Hashable, reward: float, probability: float) -> None:
        """Learn about the result of an action that was taken in a context."""

        n_actions  = len(actions)
        action     = actions[action]
        memory_key = MemoryKey(context, action)

        learn_start = time.time()
        self._cmt.learn(key=memory_key, value=reward, weight=1/(n_actions*probability))
        self._times[1] += time.time()-learn_start

class StackedLearner:

    def __init__(self, epsilon: float, emt: EMT, X:str, coin:bool, constant:bool) -> None:

        assert 0 <= epsilon and epsilon <= 1

        self._epsilon = epsilon
        self._i       = 0
        self._emt     = emt
        self._times   = [0, 0]
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
        return { 'family': 'ComboLearner', 'e': self._epsilon, **self._emt.params, "other": self._args }

    def predict(self, context: Hashable, actions: Sequence[Hashable]) -> Sequence[float]:
        """Choose which action index to take."""

        self._i += 1

        if logn and self._i % logn == 0:
           print(f"MEM {self._i}. avg prediction time {round(self._times[0]/self._i,2)}")
           print(f"MEM {self._i}. avg learn      time {round(self._times[1]/self._i,2)}")

        memories = [ self._emt.predict(MemoryKey(context, a)) for a in actions ]
        adfs     = [ {'a':a, 'm':m }  for a,m in zip(actions,memories) ]
        probs    = self._vw.predict(self._vw.make_examples({'x':context}, adfs, None))

        return probs, {'adfs':adfs}

    def learn(self, context: Hashable, actions: Sequence[Hashable], action: Hashable, reward: float, probability: float, adfs: Any) -> None:
        """Learn about the result of an action that was taken in a context."""

        n_actions = len(actions)
        action    = actions[action]

        self._emt.learn(key=MemoryKey(context, action), value=reward, weight=1/(n_actions*probability))
        labels = self._labels(actions, action, reward, probability)
        self._vw.learn(self._vw.make_examples({'x':context}, adfs, labels))

    def _labels(self,actions,action,reward:float,prob:float):
        return [ f"{i+1}:{round(-reward,5)}:{round(prob,5)}" if a == action else None for i,a in enumerate(actions)]

    def __reduce__(self):
        return (type(self), (self._epsilon, self._emt, *self._args))

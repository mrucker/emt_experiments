from learners import EMT, CMT, EpisodicLearner, StackedLearner

import coba as cb

n_shuffle = 20 #To reproduce the EMT paper set this to 20
processes = 14
epsilon   = 0.1

if __name__ == '__main__':

   #all of the hyperparameters were tuned for optimal performance
   learners = [
      #Parametric
      cb.VowpalEpsilonLearner(epsilon, features=["a","xa","xxa"]),

      #EMT-CB (self-consistent)
      EpisodicLearner(epsilon, EMT(split=300, scorer="self_consistent_rank"    , router="eigen"   , interactions=["xa"], weight=False)),

      #EMT-CB (not self-consistent)
      EpisodicLearner(epsilon, EMT(split=50 , scorer="not_self_consistent_rank", router="eigen"   , interactions=[    ], weight=False)),

      #CMT-CB
      EpisodicLearner(epsilon, CMT(max_nodes=2000, mems_per_leaf=100, dream_repeats=10, alpha=0.50, interactions=['xa'], weight=False)),

      #PEMT-CB
      StackedLearner (epsilon, EMT(split=300, scorer="self_consistent_rank"    , router="eigen"   , interactions=['xa'], weight=False), "xxa", False, True),

      #PCMT-CB
      StackedLearner (epsilon, CMT(max_nodes=2000, mems_per_leaf=100, dream_repeats=10, alpha=0.50, interactions=['xa']              ), "xxa", False, True),
   ]

   description = "Experiments with unbounded memory on EMT."
   log         = "./outcomes/unbounded.log.gz"

   environments = cb.Environments.from_template("./experiments/new.json", n_shuffle=n_shuffle)

   result = cb.Experiment(environments, learners, description=description).run(log,processes=processes)
   result.filter_fin(4_000).plot_learners()

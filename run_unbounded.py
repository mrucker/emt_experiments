from learners import EMT, CMT, EpisodicLearner, StackedLearner

import coba as cb

n_shuffle = 1 #To reproduce the EMT paper set this to 50
config    = {"processes": 7 }
epsilon   = 0.1

if __name__ == '__main__':

   #all of the hyperparameters were tuned for optimal performance
   learners = [
      #Parametric
      cb.VowpalEpsilonLearner(epsilon, features=["a","xa","xxa"]),

      # #EMT-CB (self-consistent)
      EpisodicLearner        (epsilon, EMT(split=100, scorer="self_consistent_rank"    , router="eigen"  , interactions=["xa"])),

      #EMT-CB (not self-consistent)
      EpisodicLearner        (epsilon, EMT(split=50 , scorer="not_self_consistent_rank", router="eigen"  , interactions=[])),

      #CMT-CB
      EpisodicLearner        (epsilon, CMT(max_nodes=2000, mems_per_leaf=100, dream_repeats=10, alpha=0.50 , interactions=['xa'])),

      #PEMT-CB
      StackedLearner         (epsilon, EMT(split=100, scorer="self_consistent_rank"    , router="eigen"  , interactions=['xa']), "xxa", False, True),

      #PCMT-CB
      StackedLearner         (epsilon, CMT(max_nodes=2000, mems_per_leaf=100, dream_repeats=10, alpha=0.50 ,  interactions=['xa']), "xxa", False, True),
   ]

   description = "Experiments with unbounded memory on EMT."
   log         = "./outcomes/unbounded.log.gz"

   environments = cb.Environments.cache_dir('.coba_cache').from_template("./experiments/unbounded.json", n_shuffle=n_shuffle)
   result = cb.Experiment(environments, learners, description=description).config(**config).run(log)
   result.filter_fin().plot_learners()

from learners import EMT, CMT, EpisodicLearner, StackedLearner

import coba as cb
import coba.experiments as cbe

n_shuffle = 50
config    = {"processes": 2}
epsilon   = 0.1

if __name__ == '__main__':

   learners = [
      # #Parametric
      cb.VowpalEpsilonLearner(epsilon, features=["a","xa","xxa"]),

      # #EMT-CB (self-consistent)
      EpisodicLearner        (epsilon, EMT(split=100, scorer=3, router=2, bound=0,                         interactions=["xa"])),

      #EMT-CB (not self-consistent)
      EpisodicLearner        (epsilon, EMT(split=50, scorer=4, router=2, bound=-1,                         interactions=[])),

      #CMT-CB
      EpisodicLearner        (epsilon, CMT(n_nodes=2000, leaf_multiplier=9 , dream_repeats=10, alpha=0.50, interactions=['xa'])),

      #PEMT-CB
      StackedLearner         (epsilon, EMT(split=100, scorer=3, router=2, bound=-1,                        interactions=['xa']), "xxa", False, True),

      #PCMT-CB
      StackedLearner         (epsilon, CMT(n_nodes=2000, leaf_multiplier=9, dream_repeats=10, alpha=0.50,  interactions=['xa']), "xxa", False, True),
   ]

   description = "Experiments with bounded memory on EMT."
   log         = "./outcomes/unbounded.log.gz"

   environments = cb.Environments.cache_dir('.coba_cache').from_template("./experiments/unbounded.json", n_shuffle=n_shuffle)

   # We sort so that the results written to bounded.log.gz are in a more desirable 
   # order for plotting. This has no effect on the actual results of the experiments.
   environments = sorted(environments, key=lambda e: (e.params['shuffle'],e.params['openml_task']))

   result = cb.Experiment(environments, learners, description, environment_task=cbe.ClassEnvironmentInfo()).config(**config).evaluate(log)
   result.filter_fin().plot_learners()

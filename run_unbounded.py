from learners import EMT, CMT, EpisodicLearner, StackedLearner
import coba as cb

n_shuffle = 20 #To reproduce the EMT paper set this to 20
processes = 14
epsilon   = 0.1

if __name__ == '__main__':

   #all of the hyperparameters were tuned for optimal performance
   learners = [
      #Parametric
      cb.VowpalEpsilonLearner(epsilon, features=["a","xa"]),

      #EMT-CB (self-consistent)
      EpisodicLearner(epsilon, EMT(split=300, scorer="self_consistent_rank"    , router="eigen"   , interactions=["xa"], weight=False)),

      #EMT-CB (not self-consistent)
      EpisodicLearner(epsilon, EMT(split=50 , scorer="not_self_consistent_rank", router="eigen"   , interactions=["xa"], weight=False)),

      #CMT-CB
      EpisodicLearner(epsilon, CMT(max_nodes=2000, mems_per_leaf=100, dream_repeats=10, alpha=0.50, interactions=['xa'], weight=False)),

      #PEMT-CB
      StackedLearner (epsilon, EMT(split=300, scorer="self_consistent_rank"    , router="eigen"   , interactions=['xa'], weight=False), "xa", False, True),

      #PCMT-CB
      StackedLearner (epsilon, CMT(max_nodes=2000, mems_per_leaf=100, dream_repeats=10, alpha=0.50, interactions=['xa']              ), "xa", False, True),
   ]

   description = "Experiments with unbounded memory on EMT."
   log         = "./results/unbounded.log.gz"
   env         = cb.Environments.from_template("./environments/feurer.json", n_shuffle=n_shuffle)

   cb.Experiment(env, learners, description=description).run(log,processes=processes)

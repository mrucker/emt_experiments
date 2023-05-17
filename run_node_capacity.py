from learners import EMT, EpisodicLearner

import coba as cb

n_shuffle = 1 #To reproduce the EMT paper set this to 2
config    = {"processes": 7 }
epsilon   = 0.1

if __name__ == '__main__':

   learners = [
      EpisodicLearner (epsilon, EMT(split=10 , scorer="self_consistent_rank", router="eigen", interactions=["xa"])),
      EpisodicLearner (epsilon, EMT(split=25 , scorer="self_consistent_rank", router="eigen", interactions=["xa"])),
      EpisodicLearner (epsilon, EMT(split=50 , scorer="self_consistent_rank", router="eigen", interactions=["xa"])),
      EpisodicLearner (epsilon, EMT(split=100, scorer="self_consistent_rank", router="eigen", interactions=["xa"])),
      EpisodicLearner (epsilon, EMT(split=150, scorer="self_consistent_rank", router="eigen", interactions=["xa"])),
   ]

   description = "Experiments with varying levels of c."
   log         = "./outcomes/node_capacity.log.gz"

   environments = cb.Environments.cache_dir('.coba_cache').from_template("./experiments/unbounded.json", n_shuffle=n_shuffle).where(n_interactions=4000)

   result = cb.Experiment(environments, learners, description=description, evaluation_task=cb.OnPolicyEvaluation(['reward','time'])).config(**config).run(log)
   result.filter_fin(4000).plot_learners()

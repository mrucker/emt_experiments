from learners import EMT, EpisodicLearner

import coba as cb

n_shuffle = 1 #To reproduce the EMT paper set this to 2
processes = 14
epsilon   = 0.1

cb.OnPolicyEvaluator

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

   environments = cb.Environments.from_template("./experiments/class190.json", n_shuffle=n_shuffle, strict=True)

   result = cb.Experiment(environments, learners, cb.OnPolicyEvaluator(['reward','time'], description=description,)).run(log,processes=processes)
   result.plot_learners()

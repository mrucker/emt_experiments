from learners import EMT, EpisodicLearner

import coba as cb

n_shuffle = 20
processes = 14
epsilon   = 0.1

if __name__ == '__main__':

   learners = [
      EpisodicLearner (epsilon, EMT(split=25 , scorer="self_consistent_rank", router="eigen", interactions=["xa"], weight=False)),
      EpisodicLearner (epsilon, EMT(split=50 , scorer="self_consistent_rank", router="eigen", interactions=["xa"], weight=False)),
      EpisodicLearner (epsilon, EMT(split=100, scorer="self_consistent_rank", router="eigen", interactions=["xa"], weight=False)),
      EpisodicLearner (epsilon, EMT(split=200, scorer="self_consistent_rank", router="eigen", interactions=["xa"], weight=False)),
      EpisodicLearner (epsilon, EMT(split=300, scorer="self_consistent_rank", router="eigen", interactions=["xa"], weight=False)),
   ]

   description = "Experiments with varying levels of c."
   log         = "./results/capacity.log.gz"
   env         = cb.Environments.from_template("./environments/feurer.json", n_shuffle=n_shuffle)

   cb.Experiment(env, learners, cb.OnPolicyEvaluator(['reward','time']), description=description,).run(log,processes=processes)

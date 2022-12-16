from learners import EMT, StackedLearner
import coba as cb

n_shuffle = 1 #To reproduce the EMT paper results set this to 50
config    = {"processes": 2}
epsilon   = 0.1

if __name__ == '__main__':
    
    #the learners we wish to test
    learners = [
        cb.VowpalEpsilonLearner(epsilon, features=["a","xa","xxa"]),
        StackedLearner         (epsilon, EMT(bound=1000 , scorer="self_consistent_rank", router="eigen", split=100,  interactions=['xa']), "xxa", False, True),
        StackedLearner         (epsilon, EMT(bound=2000 , scorer="self_consistent_rank", router="eigen", split=100,  interactions=['xa']), "xxa", False, True),
        StackedLearner         (epsilon, EMT(bound=16000, scorer="self_consistent_rank", router="eigen", split=100,  interactions=['xa']), "xxa", False, True),
        StackedLearner         (epsilon, EMT(bound=32000, scorer="self_consistent_rank", router="eigen", split=100,  interactions=['xa']), "xxa", False, True),
    ]

    description = "Experiments with bounded memory on EMT."
    log         = "./outcomes/bounded.log.gz"

    environments = cb.Environments.cache_dir(".coba_cache").from_template("./experiments/bounded.json", n_shuffle=n_shuffle, n_take=32000)
    
    #we sort so that the results written to bounded.log.gz are in a more desirable 
    #order for processing. This has no effect on the actual results of the experiments.
    environments = sorted(environments, key=lambda e: (e.params['shuffle'],e.params['openml_task']))

    result = cb.Experiment(environments, learners, description=description).config(**config).evaluate(log)
    result.filter_fin(32000).plot_learners(y='reward')

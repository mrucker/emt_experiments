from learners import EMT, StackedLearner
import coba as cb

n_shuffle = 20 #To reproduce the EMT paper results set this to 20
processes = 14
epsilon   = 0.1

if __name__ == '__main__':
    
    #the learners we wish to test
    learners = [
        cb.VowpalEpsilonLearner(epsilon, features=["a","xa"]),
        StackedLearner         (epsilon, EMT(bound=1000 , scorer="self_consistent_rank", router="eigen", split=300,  interactions=['xa'], weight=False), "xa", False, True),
        StackedLearner         (epsilon, EMT(bound=2000 , scorer="self_consistent_rank", router="eigen", split=300,  interactions=['xa'], weight=False), "xa", False, True),
        StackedLearner         (epsilon, EMT(bound=16000, scorer="self_consistent_rank", router="eigen", split=300,  interactions=['xa'], weight=False), "xa", False, True),
        StackedLearner         (epsilon, EMT(bound=32000, scorer="self_consistent_rank", router="eigen", split=300,  interactions=['xa'], weight=False), "xa", False, True),
    ]

    description = "Experiments with bounded memory on EMT."
    log         = "./outcomes/bounded4.log.gz"
    env         = cb.Environments.from_template("./experiments/new.json", n_shuffle=n_shuffle, n_take=32_000, strict=True).scale(0,'minmax')
    
    cb.Experiment(env, learners, description=description).run(log,processes=processes)

import time
import bisect
from subprocess import Popen, PIPE, STDOUT
from pathlib import Path

import coba as cb

from learners import EMT, CMT, VWC

def evaluator(learner: VWC, interactions):
    
    first, interactions = cb.peek_first(interactions)
    learner.set_params(first['actions'])

    for i,interaction in enumerate(interactions):
        key = {'x':interaction['context']}
        true_action = interaction['rewards'].argmax()

        start = time.time()
        pred_action = learner.predict(key)
        pred_time = time.time()-start

        start = time.time()
        learner.learn(key, true_action , 1)
        learn_time = time.time()-start
        
        yield {'reward': int(str(pred_action) == true_action), 'pred_time':pred_time, 'learn_time':learn_time, 'total_time': pred_time+learn_time }

class NShot:
    def __init__(self, n, seed) -> None:
        self._n = n
        self._seed = seed

    def filter(self, interactions):

        rng = cb.CobaRandom(self._seed)
        get_label = lambda i: i['rewards'].argmax()

        interactions  = sorted(interactions, key=get_label)        
        class_indexes = []

        lo = 0
        hi = len(interactions)

        while lo != hi:
            new_lo = bisect.bisect_right(interactions, get_label(interactions[lo]), lo, hi, key=get_label)
            class_indexes.append(rng.shuffle(list(range(lo,new_lo)),inplace=True))
            lo = new_lo

        for _ in range(self._n):
            for indexes in rng.shuffle(class_indexes,inplace=True):
                yield interactions[indexes.pop()]            

if __name__ == "__main__":

    n_processes = 8
    n_shuffle   = 10
    log         = 'few_shot.log.gz' 

    #datasets for these experiments can be found at http://kalman.ml.cmu.edu/wen_datasets/
    #the experiment assumes all data sets are stored in a ./data/ directory.

    #you should also make a ./caches/ and ./models/ directory for outputs from this experiment

    env = cb.Environments.from_supervised(cb.LibSvmSource("./data/aloi")).filter([NShot(100,i) for i in range(n_shuffle)])

    learners = [
        VWC("--oaa {} --quiet"),
        VWC("--log_multi {} --quiet"),
        VWC("--recall_tree {} --quiet"),
        EMT(100),
        CMT(100,50,.4 ,max_nodes=9000,learn_at_leaf=False),
    ]

    cb.Experiment(env,learners,evaluation_task=evaluator).run(log,processes=n_processes)

    datasets = {
        "aloi":{"train":"aloi_train.vw"                    , "test": "aloi_test.vw"                    , "classes":1_000 },
        "par1":{"train":"paradata10000_one_shot.vw.train"  , "test": "paradata10000_one_shot.vw.test"  , "classes":10_000},
        "par2":{"train":"paradata10000_two_shot.vw.train"  , "test": "paradata10000_two_shot.vw.test"  , "classes":10_000},
        "par3":{"train":"paradata10000_three_shot.vw.train", "test": "paradata10000_three_shot.vw.test", "classes":10_000},
        "img1":{"train":"imagenet_1_shots_training.txt"    , "test": "imagenet_1_shots_testing.txt"    , "classes":21_807},
        "img2":{"train":"imagenet_2_shots_training.txt"    , "test": "imagenet_2_shots_testing.txt"    , "classes":21_807},
        "img3":{"train":"imagenet_3_shots_training.txt"    , "test": "imagenet_3_shots_testing.txt"    , "classes":21_807},
        "img5":{"train":"imagenet_5_shots_training.txt"    , "test": "imagenet_5_shots_testing.txt"    , "classes":21_807},
    }

    #items = [(model, dataset, passes, train args, test args)...]
    items = [
        ['aloi_emt', datasets['aloi'], 3, '--emt --emt_leaf 200 --noconstant --coin', '--testonly'],
        ['aloi_cmt', datasets['aloi'], 3, '--memory_tree 9030 -l 0.001 --max_number_of_labels 1000 --leaf_example_multiplier 4 --dream_repeats 3 --alpha 0.1', ''],
        ['aloi_oaa', datasets['aloi'], 3, '--oaa 1000', '--testonly'],
        ['aloi_lom', datasets['aloi'], 3, '--log_multi 1000', '--testonly'],
        ['aloi_rt' , datasets['aloi'], 3, '--recall_tree 1000', '--testonly'],

        ['par1_emt', datasets['par1'], 2, '--emt --emt_leaf 200 --noconstant', '--testonly'],
        ['par1_cmt', datasets['par1'], 2, '--memory_tree 752 -l 0.1 --max_number_of_labels 10000 --leaf_example_multiplier 4 --dream_repeats 5 --dream_at_update 1 --alpha 0.1', ''],
        ['par1_oaa', datasets['par1'], 2, '--oaa 10000', '--testonly'],
        ['par1_lom', datasets['par1'], 2, '--log_multi 10000', '--testonly'],
        ['par1_rt' , datasets['par1'], 2, '--recall_tree 10000', '--testonly'],

        ['par2_emt', datasets['par2'], 2, '--emt --emt_leaf 200 --noconstant', '--testonly'],
        ['par2_cmt', datasets['par2'], 2, '--memory_tree 1399 -l 0.1 --max_number_of_labels 10000 --leaf_example_multiplier 4 --dream_repeats 5 --dream_at_update 1 --alpha 0.1', ''],
        ['par2_oaa', datasets['par2'], 2, '--oaa 10000', '--testonly'],
        ['par2_lom', datasets['par2'], 2, '--log_multi 10000', '--testonly'],
        ['par2_rt' , datasets['par2'], 2, '--recall_tree 10000', '--testonly'],

        ['par3_emt', datasets['par3'], 2, '--emt --emt_leaf 200 --noconstant', '--testonly'],
        ['par3_cmt', datasets['par3'], 2, '--memory_tree 2017 -l 0.1 --max_number_of_labels 10000 --leaf_example_multiplier 4 --dream_repeats 5 --dream_at_update 1 --alpha 0.1', ''],
        ['par3_oaa', datasets['par3'], 2, '--oaa 10000', '--testonly'],
        ['par3_lom', datasets['par3'], 2, '--log_multi 10000', '--testonly'],
        ['par3_rt' , datasets['par3'], 2, '--recall_tree 10000', '--testonly'],
 
        ['img1_emt', datasets['img1'], 2, '--emt --emt_leaf 200 --noconstant', '--testonly'],
        ['img1_cmt', datasets['img1'], 2, '--memory_tree 1513 -l 0.01 --max_number_of_labels 21850 --leaf_example_multiplier 4 --dream_repeats 3 --dream_at_update 1 --alpha 0.1', ''],
        ['img1_oaa', datasets['img1'], 2, '--oaa 21850', '--testonly'],
        ['img1_lom', datasets['img1'], 2, '--log_multi 21850', '--testonly'],
        ['img1_rt' , datasets['img1'], 2, '--recall_tree 21850', '--testonly'],

        ['img2_emt', datasets['img2'], 2, '--emt --emt_leaf 200 --noconstant', '--testonly'],
        ['img2_cmt', datasets['img2'], 2, '--memory_tree 2829 -l 0.01 --max_number_of_labels 21850 --leaf_example_multiplier 4 --dream_repeats 3 --dream_at_update 1 --alpha 0.1', ''],
        ['img2_oaa', datasets['img2'], 2, '--oaa 21850', '--testonly'],
        ['img2_lom', datasets['img2'], 2, '--log_multi 21850', '--testonly'],
        ['img2_rt' , datasets['img2'], 2, '--recall_tree 21850', '--testonly'],

        ['img3_emt', datasets['img3'], 2, '--emt --emt_leaf 200 --noconstant', '--testonly'],
        ['img3_cmt', datasets['img3'], 2, '--memory_tree 4089 -l 0.01 --max_number_of_labels 21850 --leaf_example_multiplier 4 --dream_repeats 3 --dream_at_update 1 --alpha 0.1', ''],
        ['img3_oaa', datasets['img3'], 2, '--oaa 21850', '--testonly'],
        ['img3_lom', datasets['img3'], 2, '--log_multi 21850', '--testonly'],
        ['img3_rt' , datasets['img3'], 2, '--recall_tree 21850', '--testonly'],

        ['img5_emt', datasets['img5'], 2, '--emt --emt_leaf 200 --noconstant', '--testonly'],
        ['img5_cmt', datasets['img5'], 2, '--memory_tree 6515 -l 0.01 --max_number_of_labels 21850 --leaf_example_multiplier 4 --dream_repeats 3 --dream_at_update 1 --alpha 0.1', ''],
        ['img5_oaa', datasets['img5'], 2, '--oaa 21850', '--testonly'],
        ['img5_lom', datasets['img5'], 2, '--log_multi 21850', '--testonly'],
        ['img5_rt' , datasets['img5'], 2, '--recall_tree 21850', '--testonly'],
    ]

    def wait_one(workers):
        while True:
            time.sleep(5)
            for i,worker in enumerate(workers):
                if worker.poll() is not None:
                    return i

    is_training = lambda item: len(item) == 5
    is_testing  = lambda item: len(item) == 3
    is_finished = lambda item: len(item) == 2

    already_trained  = lambda item: Path(f"./models/{item[0]}").exists()

    def map_to_testing_item(item): del item[2:4]

    def print_worker(worker,workitem):
        if is_testing(workitem):
            print(f"Finished Training {workitem[0]}")

        if is_finished(workitem):
            output = worker.communicate()[0].splitlines()
            try:                    
                usr_time = float(output[-2][5:])
                sys_time = float(output[-1][4:])
                tot_time = usr_time+sys_time
            except:
                tot_time = 0
            
            try:
                n_examples = float(output[-8][21:])
            except:
                n_examples = 1

            try:
                error = float(output[-5][14:])
            except:
                error = 0
            print(f"Finished Testing {workitem[0]} {round(tot_time,3)} {round(1000*tot_time/n_examples,5)} {round(error,4)}")

    workers = []
    working = []

    while not all(map(is_finished,items)):

        for item in items:

            if is_finished(item):
                continue

            if already_trained(item) and is_training(item):
                map_to_testing_item(item)

            if item not in working:
                if is_training(item):
                    model  = item[0]
                    train  = item[1]['train']
                    passes = item.pop(2)
                    args   = item.pop(2)
                    print(f"Started Training {model}")
                    cmd    = f'vw ./data/{train} -f ./models/{model} {args} -b 29 --random_seed 1337 --holdout_off --passes {passes} --cache --cache_file ./caches/{model}.cache --quiet'                

                elif is_testing(item):
                    model = item[0]
                    test  = item[1]['test']
                    args  = item.pop(2)
                    print(f"Started Testing {model}")
                    cmd = f'time -p vw ./data/{test} -i ./models/{model} {args}'

                workers.append(Popen(cmd,shell=True,stdout=PIPE,stderr=STDOUT,encoding='utf8'))
                working.append(item)

            #Wait for workers to finish if we're currently using all of our 
            #processes or if we're already working on all remaining items
            if len(workers) == n_processes or all(i in working for i in items if not is_finished(i)):
                i = wait_one(workers)
                print_worker(workers.pop(i),working.pop(i))

    while workers:
        i = wait_one(workers)
        print_worker(workers.pop(i),working.pop(i))

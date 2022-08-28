#!/bin/python3

from pickle import INT
from scipy.sparse.construct import rand
import cv_search as cv
import argparse
from profiles import profile_manager as pm
import multiprocessing as mp
import pandas as pd

from uuid import uuid4

############ PARAMS #############
#   ITERATIONS                  #
#   SETS                        #
#   CONFIG-PATH                 #
#   FILE-PATH                   #
#   FILE-NAME                   #
#################################

parser = argparse.ArgumentParser()

# necessary
parser.add_argument('iterations', help='sv search iterations', type=int)
parser.add_argument('sets', help='sv search sets', type=int)

# optional
parser.add_argument('-c', '--conf-path', help='file that contains a cv search profile', type=str, default='./configs/cv_search.json')
parser.add_argument('-p', '--file-path', help='file path for the best trained model', type=str, default='./models')
parser.add_argument('-f', '--file-name', help='file path for the best trained model', type=str, default=str(uuid4())+'.joblib')

arguments = parser.parse_args()

PROFILE     = pm.get_profile_by_file(arguments.conf_path)
PROVIDER    = cv.get_profile_provider(PROFILE)
CORES       = mp.cpu_count()
N_ITER      = arguments.iterations
N_SETS      = arguments.sets
SAVE_PATH   = f'{arguments.file_path}/{arguments.file_name}'

# distribute iterations to cores
params = [0 for _ in range(CORES if N_ITER >= CORES else N_ITER)]
for i in range(N_ITER):
    params[i % CORES] += 1

print('PARAMS:', params)

DATA = PROVIDER.get_data()

def move (y, x):
    return "\033[%d;%dH" % (y, x)

# Print iterations progress
def printProgressBar (iteration, total, y, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'{move(y, 1)}{prefix} |{bar}| {percent}% {suffix}')
    # Print New Line on Complete

    if iteration == total: 
        print()


def run_search(iterations: int, index: int):
    model, grid = cv.get_model_and_grid(cv.Classifier.GRADIENT_BOOSTED_CLASSIFIER)
    wrapped     = cv.Wrapper(model, DATA, PROFILE)

    search = cv.CV_Search(
        model     = wrapped,
        grid      = grid,
        n_iter    = iterations,
        n_cv_sets = N_SETS,
        verbose   = 0
    )

    N_SETS_TOTAL = iterations * N_SETS

    progress = { 'n_iterations_total': 0 }

    @search.event_emitter.on('iteration_run')
    def on_iteration(current, max):
        progress['n_iterations_total'] = current

    @search.event_emitter.on('set_run')
    def on_set(current, max):
        n_passed_sets = progress['n_iterations_total'] * N_SETS + current + 1
        printProgressBar(n_passed_sets, N_SETS_TOTAL, index + 5, f'CORE: {index}')

    return search.run()
    
pool = mp.Pool(CORES)
results = pool.starmap(run_search, [(params[i], i + 1) for i in range(len(params))])
pool.close()

result = pd.concat(results, axis=0).sort_values(['val_score'], ascending=False).iloc[0]

print('BEST MODEL:', result[["train%", "train_score", "val%", "val_score", "min_val_score"]])

model, grid = cv.get_model_and_grid(cv.Classifier.GRADIENT_BOOSTED_CLASSIFIER)
wrapped     = cv.Wrapper(model, DATA, PROFILE)
wrapped.set_params(result['params'])
wrapped.fit(wrapped.df, wrapped.labels)
wrapped.save(path=SAVE_PATH)

print(f'model saved at {SAVE_PATH}')
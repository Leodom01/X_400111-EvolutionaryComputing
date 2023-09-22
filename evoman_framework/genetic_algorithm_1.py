import sys, os
import time
import numpy as np
import matplotlib.pyplot as plt

from evoman.environment import Environment
from demo_controller import player_controller
import matplotlib.pyplot as plt

import pandas as pd


def dump_it(run_numb,matrix_of_run,name):
    
    if not os.path.exists("data"):
        os.makedirs("data")
    
    if not os.path.exists("data/"+name):
        os.makedirs("data/"+name)
    
    header = ["step", "bsf", "mean_fitness", "std_fitness", "min_fitness", "max_fitness", "sq_distance_diversity"]
    
    df = pd.DataFrame(matrix_of_run,columns=header)
    df.to_csv(os.path.join("data/"+name, "run-"+str(run_numb)+".csv"), index=False)

# Genetic Algorithm Implementation #

def simulation(env, individual):
    """
    runs single simulation, returns fitness score
    """
    fitness, p, e, t = env.play(pcont=individual)
    return fitness


def evaluate(env, population):
    """
    obtains fitness scores for population_size individuals
    returns a numpy array [population_size]
    """
    return np.array(list(map(lambda y: simulation(env,y), population)))

#from sample provided by them
def save_on_file(name,fit_pop,best,mean,std,i):
    
    if(i==0):
        file_aux  = open(name+'/results.txt','w')
        file_aux.write('\n\ngen best mean std')
    else:
        file_aux  = open(name+'/results.txt','a')
    print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()


def tournament_selection(parent_size, tournament_size, p, p_f):
    """
    selects parent_size individuals using tournaments with tournament_size comparisons
    where p is the current population and p_f the fitness scores of p
    returns single parent
    """
    def single_tournament():
        best = np.random.randint(0, len(p))
        members = [best] + [np.random.randint(0, len(p)) for _ in range(tournament_size)]
        winner = max(members, key=lambda idx: p_f[idx])
        return p[winner]

    for parent in range(parent_size):
        yield single_tournament()


def komma_selection(env, amount, offspring, fit_scores):
    """
    survivor selection using the komma strategy
    best amount of offspring is selected
    returns new population and fitness scores
    """
    best_indices = fit_scores.argsort()[-amount:]
    fit_scores = np.array([fit_scores[child] for child in best_indices])
    population = np.array([offspring[child] for child in best_indices])
    return population, fit_scores


def gaussian_mutation(child, sigma, wl, wh):
    """
    adds gaussian noise (mutation) to every allele of child
    such that wl <= allele' <= wh, using a std of sigma
    returns child'
    """
    def checkrange(allele):
        if allele < wl:
            return wl
        if allele > wh:
            return wh
        return allele

    child = [(allele + np.random.normal(0, sigma)) for allele in child]
    return [checkrange(allele) for allele in child]

def give_two_random_parents(parents):
    p1, p2 = np.random.randint(0, len(parents)), np.random.randint(0, len(parents))
    while p2 != p1:
        p2 = np.random.randint(0, len(parents))
    p1, p2 = parents[p1], parents[p2]
    return p1,p2


def two_point_crossover(parents, child_size):
    """
    two point crossover implementation
    """

    def return_children():

        p1,p2 = give_two_random_parents(parents)

        size = len(p1)
        point_1_chosen = np.random.randint(0, size)
        point_2_chosen = np.random.randint(0, size)

        #in case of clash
        while point_1_chosen == point_2_chosen:
            point_2_chosen = np.random.randint(0, size)
        
        #in case of order not respected
        if point_1_chosen>point_2_chosen:
                point_1_chosen, point_2_chosen = point_2_chosen, point_1_chosen
        
        #cause apparently for numpy this was a too smart way to do it
        #child1 = p1[:point_1_chosen] + p2[point_1_chosen:point_2_chosen]+ p1[point_2_chosen:]
        master_array_temp=np.array([])
        master_array_temp=np.append(master_array_temp,p1[:point_1_chosen], axis=0)
        master_array_temp=np.append(master_array_temp,p2[point_1_chosen:point_2_chosen], axis=0)
        master_array_temp=np.append(master_array_temp,p1[point_2_chosen:], axis=0)
        child_1=master_array_temp

        #child2 = p2[:point_1_chosen] + p1[point_1_chosen:point_2_chosen] + p2[point_2_chosen:]
        master_array_temp=np.array([])
        master_array_temp=np.append(master_array_temp,p2[:point_1_chosen], axis=0)
        master_array_temp=np.append(master_array_temp,p1[point_1_chosen:point_2_chosen], axis=0)
        master_array_temp=np.append(master_array_temp,p2[point_2_chosen:], axis=0)
        child_2=master_array_temp

        return child_1, child_2

    child_counter = 0
    while child_counter < child_size:
        child_1, child_2 = return_children()
        yield child_1
        child_counter += 1
        if child_counter < child_size:
            yield child_2
            child_counter += 1



def whole_arithmetic_crossover(parents, child_size, alpha):
    """
    whole arithmetic crossover implementation
    """
    def return_children():
        p1,p2 = give_two_random_parents(parents)
        child_1 = [alpha * p1[allele] + (1-alpha) * p2[allele] for allele in range(len(p1))]
        child_2 = [alpha * p2[allele] + (1-alpha) * p1[allele] for allele in range(len(p1))]
        return child_1, child_2

    child_counter = 0
    while child_counter < child_size:
        child_1, child_2 = return_children()
        yield child_1
        child_counter += 1
        if child_counter < child_size:
            yield child_2
            child_counter += 1



def run_experiments(n, enemy, use_2pt_crossover=True, use_plus_selection=True):
    """
    TODO: (OLD)
    - function should run n amount of experiments, and writes to file
    - implement crossover functions 
    - implement mutation function (gaussian)
    - finish genetic algorithm loop
    - correct format for writing to file, such that plot.py from alessio works
    - finetuning parameter?
    - implement Elitsim? (keeping n individuals of population for next generation)
       - because we are currently replacing all parents with children (komma selection)

    """
    

    # hyperparameters
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    n_vars = 265
    population_size = 100
    n_generations = 20
    parent_size = 20
    children_size = 1.5 * population_size
    tournament_size = 20
    alpha = 0.5
    mutation_rate = 0.05
    w_low = -1
    w_high = 1

    # crossover configuration
    crossover = whole_arithmetic_crossover
    if use_2pt_crossover:
        crossover = two_point_crossover
    name = crossover.__name__ + 'use_plus_selection=' + str(use_plus_selection) + "_EA"
    if not os.path.exists(name):
        os.makedirs(name)

    for run in range(n):
        env = Environment(experiment_name=name, enemies=[enemy], playermode="ai",
                          player_controller=player_controller(10), enemymode="static", logs="on",
                          savelogs="no", level=2, speed="fastest", visuals=False)
        population = np.random.uniform(w_low, w_high, (population_size, n_vars))
        fit_scores = evaluate(env, population)
        print('run number: ' + str(run)+"--------------------")

        temp_matrix_for_dumping = np.empty((0,7))

        for gen in range(n_generations):
            print('gen: ' + str(gen))
            print('mean fitness: ' + str(np.mean(fit_scores)))
            print('highest fitness: ' + str(np.max(fit_scores)))
            print('std: ' + str(np.std(fit_scores)))
            print()
            #save_on_file(name,fit_scores,np.argmax(fit_scores),np.mean(fit_scores),np.std(fit_scores),gen)
            #two times best fitness passed?
            #square distance diversity I imagined on fitness temporary fixed to 1
            temp_matrix_for_dumping = np.append(temp_matrix_for_dumping, [[gen,np.max(fit_scores),np.mean(fit_scores),np.std(fit_scores),np.min(fit_scores),np.max(fit_scores),1]], axis=0) 
            
            #why 10 parents?
            parents = np.array([parent for parent in tournament_selection(parent_size, tournament_size, population, fit_scores)])
            if use_2pt_crossover:
                offspring = np.array([child for child in crossover(parents, children_size)])
            else:
                offspring = np.array([child for child in crossover(parents, children_size, alpha)])
            offspring = np.array([gaussian_mutation(child, mutation_rate, w_low, w_high) for child in offspring])
            offspring_fit = evaluate(env, offspring)
            if use_plus_selection:
                offspring = np.concatenate((population, offspring), axis=0)
                offspring_fit = np.concatenate((fit_scores, offspring_fit))
            population, fit_scores = komma_selection(env, population_size, offspring, offspring_fit)

        #np.savetxt(name+'/best.txt',population[np.argmax(fit_scores)])
        dump_it(run,temp_matrix_for_dumping,name)
        np.savetxt("data/"+name+"/individual"+str(run)+".txt",population[np.argmax(fit_scores)])
        
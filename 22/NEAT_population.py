"""Implements the core evolution algorithm."""
from __future__ import division, print_function

from neat.reporting import ReporterSet, BaseReporter
from neat.population import Population, CompleteExtinctionException
from neat.six_util import iteritems, itervalues

import csv
import os

from neat.math_util import mean, stdev
from neat.six_util import itervalues, iterkeys

RUNS_DIR = 'runs'


class EvomanPopulation(Population):
    def __init__(self, config):
        super().__init__(config)

    def run(self, fitness_function, n=None):

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation)

            # Evaluate all genomes using the user-provided function.
            fitness_function(list(iteritems(self.population)), self.config)

            # Gather and report statistics.
            best = None
            for g in itervalues(self.population):
                # if best is None or g.fitness > best.fitness:
                if best is None or g.individual_gain > best.individual_gain or (
                        (g.individual_gain == best.individual_gain) and (
                        g.fitness > best.fitness)):  # first criteria is individual gain, then fitness
                    best = g

            self.reporters.post_evaluate(self.config, self.generation, self.population, best)

            # Track the best genome ever seen.x
            if self.best_genome is None or best.individual_gain > self.best_genome.individual_gain or (
                    (best.individual_gain == self.best_genome.individual_gain) and (
                    best.fitness > self.best_genome.fitness)):
                self.best_genome = best

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)

            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        # self.reporters.reporters[0].final_plot_report()
        return self.best_genome, best

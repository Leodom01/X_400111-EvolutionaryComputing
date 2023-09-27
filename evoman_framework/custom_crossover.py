from toolz import curry
from leap_ec import ops

@curry
@ops.iteriter_op
def whole_arithmetic_recombination(next_individual, alpha = 0.5):
    def _whole_arithmetic_recombination(parent1, parent2, alpha):
        tmp = alpha * parent1.genome + (1 - alpha) * parent2.genome
        parent2.genome = (1 - alpha) * parent1.genome + alpha * parent2.genome
        parent1.genome = tmp

        return parent1, parent2

    while True:
        parent1 = next(next_individual)
        parent2 = next(next_individual)

        child1, child2 = _whole_arithmetic_recombination(
            parent1, 
            parent2, 
            alpha
        )

        child1.fitness = child2.fitness = None

        yield child1
        yield child2

@curry
@ops.iteriter_op
def random_arithmetic_recombination(next_individual):
    def _random_arithmetic_recombination(parent1, parent2):
        alpha = np.random.uniform()
        tmp = alpha * parent1.genome + (1 - alpha) * parent2.genome
        parent2.genome = (1 - alpha) * parent1.genome + alpha * parent2.genome
        parent1.genome = tmp

        return parent1, parent2

    while True:
        parent1 = next(next_individual)
        parent2 = next(next_individual)

        child1, child2 = _random_arithmetic_recombination(
            parent1, 
            parent2, 
        )

        child1.fitness = child2.fitness = None

        yield child1
        yield child2

@curry
@ops.iteriter_op
def no_crossover(next_individual):
    def _no_crossover(parent1, parent2):
        return parent1, parent2

    while True:
        parent1 = next(next_individual)
        parent2 = next(next_individual)

        child1, child2 = _no_crossover(
            parent1, 
            parent2, 
        )

        yield child1
        yield child2

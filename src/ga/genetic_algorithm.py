import copy
import random
import os
import json
from typing import Callable, Tuple
from src.ga.representation import Strategy, random_strategy
from src.ga.fitness import fitness
from src.ga.selection import elitism, tournament_selection


class GeneticAlgorithm:
    """
    A genetic algorithm to evolve strategies for playing the Iterated Prisoner's Dilemma.
    """
    def __init__(
        self,
        population_size: int,
        crossover_rate: float,
        crossover_func: Callable[[Strategy, Strategy], Tuple[Strategy, Strategy]],
        mutation_rate: float,
        mutation_func: Callable[[Strategy, float], Strategy],
        generations: int,
        early_stop_threshold: int,
        elitism_rate: float,
        tournament_size: int,
        ipd_rounds: int
    ):
        """
        Initializes the genetic algorithm.

        Args:
            population_size: The number of individuals in the population.
            crossover_rate: The probability of performing crossover.
            crossover_func: The function that performs crossover on two parents.
            mutation_rate: The probability of performing mutation.
            mutation_func: The function that performs mutation on an individual.
            generations: The number of generations to run the algorithm for.
            early_stop_threshold: The number of generations without improvement before stopping.
            elitism_rate: The proportion of individuals to retain through elitism.
            tournament_size: The size of the tournament for selection.
            ipd_rounds: The number of IPD rounds.
        """
        self.population_size = population_size
        self.population = [random_strategy() for _ in range(population_size)]
        self.crossover_rate = crossover_rate
        self.crossover_func = crossover_func
        self.mutation_rate = mutation_rate
        self.mutation_func = mutation_func

        self.generations = generations
        self.early_stop_threshold = early_stop_threshold
        self.elitism_rate = elitism_rate
        self.elitism_count = int(elitism_rate * population_size)
        self.tournament_size = tournament_size
        self.ipd_rounds = ipd_rounds

        self.avg_fitness_per_gen = []
        self.best_fitness_per_gen = []
        self.best_fitness = float("-inf")
        self.best_solution = None
        self.no_improvement_count = 0

    def evolve(self) -> None:
        """
        Runs the genetic algorithm to evolve strategies.
        """
        for _ in range(self.generations):
            fitness_scores = [
                fitness(individual, self.ipd_rounds) for individual in self.population
            ]

            self.avg_fitness_per_gen.append(sum(fitness_scores) / len(fitness_scores))
            gen_best_fitness = max(fitness_scores)
            self.best_fitness_per_gen.append(gen_best_fitness)

            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_solution = copy.deepcopy(
                    self.population[fitness_scores.index(gen_best_fitness)]
                )
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            # Check for early stopping
            if self.no_improvement_count >= self.early_stop_threshold:
                break

            # Elitism
            elite_individuals = elitism(self.population, fitness_scores, self.elitism_count)

            # Selection
            parents = tournament_selection(
                self.population,
                fitness_scores,
                self.tournament_size,
                len(self.population) - self.elitism_count
            )

            # Crossover
            next_population = []
            for i in range(0, len(parents) - 1, 2):
                parent1, parent2 = parents[i], parents[i+1]

                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover_func(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

                next_population.extend([child1, child2])

            # Handle odd-lengths
            if len(parents) % 2 == 1:
                next_population.append(parents[-1])

            # Mutation
            for i in range(len(next_population)):
                next_population[i] = self.mutation_func(next_population[i], self.mutation_rate)

            # Replacement
            self.population = elite_individuals + next_population

    def save_results(self, path: str) -> None:
        """
        Saves the results and configuration of the genetic algorithm to a JSON file.

        Args:
            path: The file path where the results will be saved.
        """
        results = {
            "results": {
                "best_fitness": self.best_fitness,
                "best_solution": self.best_solution.bitstring,
                "avg_fitness_per_gen": [round(fitness, 4) for fitness in self.avg_fitness_per_gen],
                "best_fitness_per_gen": [round(fitness, 4) for fitness in self.best_fitness_per_gen]
            },
            "config": {
                "population_size": self.population_size,
                "crossover_rate": self.crossover_rate,
                "crossover_func": self.crossover_func.__name__,
                "mutation_rate": self.mutation_rate,
                "mutation_func": self.mutation_func.__name__,
                "elitism_rate": self.elitism_rate,
                "tournament_size": self.tournament_size,
                "ipd_rounds": self.ipd_rounds
            }
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            json.dump(results, file, indent=4)

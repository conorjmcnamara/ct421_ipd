import os
import json
import matplotlib.pyplot as plt


def analyse_results(results_dir: str, plot_all: bool = False) -> None:
    """
    Analyses the results of the genetic algorithm and plots data from the results files.

    Args:
        results_dir: The directory containing the results JSON files.
        plot_all: Boolean flag to determine if all results or only the best should be plotted.
    """
    results_paths = [
        os.path.join(results_dir, filename)
        for filename in os.listdir(results_dir)
        if filename.endswith(".json")
    ]

    if plot_all:
        plot_fitness(results_paths)
    else:
        best_fitness = -float("inf")
        best_path = None

        for path in results_paths:
            with open(path, 'r') as file:
                results = json.load(file)
                current_best_fitness = max(results["results"]["best_fitness_per_gen"])

                if current_best_fitness > best_fitness:
                    best_fitness = current_best_fitness
                    best_path = path

        plot_fitness([best_path])


def plot_fitness(results_paths: list) -> None:
    """
    Plots the fitness scores (average and best) per generation from one or more result files.

    Args:
        results_paths: A list of paths to the results JSON files.
    """
    # Plot average fitness
    plt.figure(figsize=(10, 6))
    for results_path in results_paths:
        with open(results_path, 'r') as file:
            results = json.load(file)

        avg_fitness = results["results"]["avg_fitness_per_gen"]
        generations = range(len(avg_fitness))

        # Create label
        memory_size = results["config"]["memory_size"]
        noise_rate = results["config"]["noise_rate"]

        plt.plot(
            generations,
            avg_fitness,
            label=f"Mem={memory_size}, Noise={noise_rate}",
            linewidth=2,
            alpha=0.7
        )

    plt.xlabel("Generations")
    plt.ylabel("Average Fitness")
    plt.title("Average Fitness vs Generations")
    plt.legend()
    plt.grid(True)

    avg_plot_path = os.path.join(
        os.path.dirname(results_paths[0]), "plots", "avg_fitness_comparison.png"
    )
    os.makedirs(os.path.dirname(avg_plot_path), exist_ok=True)
    plt.savefig(avg_plot_path, bbox_inches="tight")
    plt.show()

    # Plot best fitness
    plt.figure(figsize=(10, 6))
    for results_path in results_paths:
        with open(results_path, 'r') as file:
            results = json.load(file)

        best_fitness = results["results"]["best_fitness_per_gen"]
        generations = range(len(best_fitness))

        # Create label
        memory_size = results["config"]["memory_size"]
        noise_rate = results["config"]["noise_rate"]

        plt.plot(
            generations,
            best_fitness,
            label=f"Mem={memory_size}, Noise={noise_rate}",
            linewidth=2,
            alpha=0.7
        )

    plt.xlabel("Generations")
    plt.ylabel("Best Fitness")
    plt.title("Best Fitness vs Generations")
    plt.legend()
    plt.grid(True)

    best_plot_path = os.path.join(
        os.path.dirname(results_paths[0]), "plots", "best_fitness_comparison.png"
    )
    os.makedirs(os.path.dirname(best_plot_path), exist_ok=True)
    plt.savefig(best_plot_path, bbox_inches="tight")
    plt.show()

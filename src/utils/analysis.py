import os
import json
import matplotlib.pyplot as plt


def analyse_results(results_dir: str) -> None:
    """
    Analyses the results of the genetic algorithm.

    Args:
        results_dir: The directory containing the results JSON files.
    """
    results_paths = [
        os.path.join(results_dir, filename)
        for filename in os.listdir(results_dir)
        if filename.endswith(".json")
    ]

    all_results = []
    for path in results_paths:
        with open(path, 'r') as file:
            data = json.load(file)
            all_results.append(data)

    best_idx = max(range(len(all_results)), key=lambda i: all_results[i]["results"]["best_fitness"])
    best_config = all_results[best_idx]
    best_path = results_paths[best_idx]

    print(f"Best configuration: {best_config}")
    plot_fitness(best_path)


def plot_fitness(results_path: str) -> None:
    """
    Plots the fitness scores (average and best) per generation from a results file.

    Args:
        results_path: The path to the results JSON file.
    """
    with open(results_path, 'r') as file:
        results = json.load(file)

    avg_fitness = results["results"]["avg_fitness_per_gen"]
    best_fitness = results["results"]["best_fitness_per_gen"]
    generations = range(len(best_fitness))

    # Plot average fitness
    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg_fitness, label="Average Fitness", color="blue", linewidth=2)
    plt.xlabel("Generations")
    plt.ylabel("Average Fitness")
    plt.title("Average Fitness vs Generations")
    plt.legend()
    plt.grid(True)

    avg_plot_path = results_path.replace("results", "plots").replace(".json", "_avg.png")
    os.makedirs(os.path.dirname(avg_plot_path), exist_ok=True)
    plt.savefig(avg_plot_path, bbox_inches="tight")
    plt.show()

    # Plot best fitness
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness, label="Best Fitness", color="red", linewidth=2)
    plt.xlabel("Generations")
    plt.ylabel("Best Fitness")
    plt.title("Best Fitness vs Generations")
    plt.legend()
    plt.grid(True)

    best_plot_path = results_path.replace("results", "plots").replace(".json", "_best.png")
    plt.savefig(best_plot_path, bbox_inches="tight")
    plt.show()

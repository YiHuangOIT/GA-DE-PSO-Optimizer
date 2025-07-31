# Prepare the complete Python code with GA, DE, and PSO implementations, including detailed English comments

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# -------------------------
# Fixed Constraints Setting
# -------------------------

Max_P = 30        # Population Size for GA/DE/PSO
Max_G = 100        # Number of Generation for GA/DE/PSO
Max_V= 2            # Number of Variant for GA/DE/PSO
Bound_L = -5.12     # Lower limit of Variant for GA/DE/PSO
Bound_U = 5.12      # Upper limit of Variant for GA/DE/PSO
pass_line = 1e-10   # Evaluate the result quality


# -------------------------
# Configurable parameter Setting
# -------------------------

# # Sets 1 - Conservative parameter sets
# -----------------------------------------------------------
# GA_MR = 0.05    # Mutation Rate of GA, (0.05~0.3)
# DE_MF = 0.3     # Mutation Factor of DE, (0.3~0.9)
# DE_CR = 0.5     # Crossover Rate of DE, (0.5~1.0)
# PSO_IW = 0.7        #Inertia weight of PSO, (0.3~0.7)
# PSO_PC = 1.0        # Personal coefficient of PSO, (1.0~2.5)
# PSO_GC = 1.0        # Global coefficient of PSO, (1.0~2.5)


# # Sets 2 - Standard parameter sets
# -----------------------------------------------------------
# GA_MR = 0.2   # Mutation Rate of GA, (0.05~0.3)
# DE_MF = 0.5     # Mutation Factor of DE, (0.3~0.9)
# DE_CR = 0.7     # Crossover Rate of DE, (0.5~1.0)
# PSO_IW = 0.5      #Inertia weight of PSO, (0.3~0.7)
# PSO_PC = 2.0      # Personal coefficient of PSO, (1.0~2.5)
# PSO_GC = 2.0      # Global coefficient of PSO, (1.0~2.5)

# Sets 3 - Aggressive parameter sets
# -----------------------------------------------------------
GA_MR = 0.3   # Mutation Rate of GA, (0.05~0.3)
DE_MF = 0.9     # Mutation Factor of DE, (0.3~0.9)
DE_CR = 1.0     # Crossover Rate of DE, (0.5~1.0)
PSO_IW = 0.3      #Inertia weight of PSO, (0.3~0.7)
PSO_PC = 2.5      # Personal coefficient of PSO, (1.0~2.5)
PSO_GC = 2.5      # Global coefficient of PSO, (1.0~2.5)

# String convert for export to csv file.
GA_PARAMS = f'MR={GA_MR}'
DE_PARAMS = f'MF={DE_MF}, CR={DE_CR}'
PSO_PARAMS = f'IW={PSO_IW}, PC={PSO_PC}'


# -------------------------
# Rastrigin Function
# -------------------------
def rastrigin(X):
    """Calculate the Rastrigin function value for a 2D input."""
    x, y = X
    return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))


# -------------------------
# Re-scale Best solution
# -------------------------
def normalize_by_min(a, b, c):
    numbers = [a, b, c]
    min_val = min(numbers)  # Find minumn value

    # Re-scale the list by setting the minimum value to 1.
    result = [1 if x == min_val else round(x / min_val, 4) for x in numbers]

    return result

# -------------------------
# Grade Best solution
# -------------------------
def grade(a, b, c):
    numbers = [a, b, c]

    result = [100+np.log(pass_line/x) if x > pass_line else 100 for x in numbers]
    return result

    # Re-scale the list by setting the minimum value to 1.
    result = [1 if x == min_val else round(x / min_val, 4) for x in numbers]

    return result
# -------------------------
# Genetic Algorithm (GA)
# -------------------------
def genetic_algorithm(pop_size=Max_P, generations=Max_G-1, mutation_rate=GA_MR):
    """Genetic Algorithm to optimize the Rastrigin function."""

    bounds = [Bound_L, Bound_U]  # Lower and upper bounds of the search space
    dim = Max_V  # Dimensionality of the problem (e.g., 2 for 2D Rastrigin)

    # Step 1: Initialize population randomly within bounds
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))

    # Step 2: Evaluate initial fitness of the population
    fitness = np.array([rastrigin(ind) for ind in population])
    best_costs = [np.min(fitness)]  # Track best fitness value in each generation
    gen = 0  # Generation counter

    # Step 3: Start evolution loop
    while gen < generations and best_costs[-1] > pass_line:

        # Evaluate fitness for current population
        fitness = np.array([rastrigin(ind) for ind in population])
        best_costs.append(np.min(fitness))  # Store best cost of current generation

        # Step 4: Selection - choose top 50% fittest individuals
        selected = population[np.argsort(fitness)[:pop_size // 2]]

        children = []  # Container for new offspring

        # Step 5: Generate offspring using crossover and mutation
        for _ in range(pop_size // 2):
            # Randomly select two distinct parents
            i1, i2 = np.random.choice(len(selected), 2, replace=False)
            p1 = selected[i1]
            p2 = selected[i2]

            # Perform linear interpolation crossover
            alpha = np.random.rand()
            child = alpha * p1 + (1 - alpha) * p2

            # With some probability, apply Gaussian mutation
            if np.random.rand() < mutation_rate:
                child += np.random.normal(0, 0.1, dim)  # Add small noise to each dimension

            # Ensure child is within bounds
            children.append(np.clip(child, bounds[0], bounds[1]))

        gen += 1  # Proceed to next generation

        # Step 6: Form the new population from selected parents and their offspring
        population = np.vstack((selected, children))

    # Step 7: Return the best solution found and the history of best costs
    best_solution = population[np.argmin([rastrigin(ind) for ind in population])]
    return (best_solution, best_costs)

# -------------------------
# Differential Evolution (DE)
# -------------------------
def differential_evolution(pop_size=Max_P, generations=Max_G-1, F=DE_MF, CR=DE_CR):
    """Differential Evolution to optimize the Rastrigin function."""

    bounds = [Bound_L, Bound_U]  # Lower and upper bounds of the search space
    dim = Max_V  # Dimensionality of the problem (e.g., 2 for 2D Rastrigin)

    # Step 1: Initialize population randomly within the bounds
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))

    # Step 2: Evaluate initial fitness of the population
    fitness = np.array([rastrigin(ind) for ind in population])
    best_costs = [np.min(fitness)]  # Track best fitness value over generations
    gen = 0  # Generation counter

    # Step 3: Begin the evolutionary process
    while gen < generations and best_costs[-1] > pass_line:

        new_population = []  # Store new generation

        # Step 4: For each individual in the population
        for i in range(pop_size):
            # Randomly choose 3 individuals a, b, c different from current index i
            indices = np.random.choice([j for j in range(pop_size) if j != i], 3, replace=False)
            a, b, c = population[indices]

            # Mutation: create mutant vector using differential mutation strategy
            mutant = np.clip(a + F * (b - c), bounds[0], bounds[1])  # Ensure within bounds

            # Crossover: mix mutant and current target individual to form trial vector
            cross_points = np.random.rand(dim) < CR  # Boolean mask of which dimensions to cross
            if not np.any(cross_points):  # Ensure at least one dimension is crossed
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, population[i])

            # Selection: evaluate both trial and original, pick the better one
            fi = rastrigin(population[i])  # Fitness of original
            ft = rastrigin(trial)          # Fitness of trial
            if ft < fi:
                new_population.append(trial)  # Trial is better
            else:
                new_population.append(population[i])  # Keep original

        # Step 5: Update population for next generation
        population = np.array(new_population)

        # Step 6: Record best cost in current generation
        best_costs.append(np.min([rastrigin(ind) for ind in population]))
        gen += 1  # Increment generation counter

    # Step 7: Return best solution found and history of best costs
    best_solution = population[np.argmin([rastrigin(ind) for ind in population])]
    return best_solution, best_costs


# -------------------------
# Particle Swarm Optimization (PSO)
# -------------------------
def particle_swarm_optimization(swarm_size=Max_P, generations=Max_G-1, w=PSO_IW, c1=PSO_PC, c2=PSO_GC):
    """Particle Swarm Optimization to optimize the Rastrigin function."""

    bounds = [Bound_L, Bound_U]  # Search space bounds
    dim = Max_V  # Dimensionality of the problem (e.g., 2 for 2D Rastrigin)

    # Step 1: Initialize particles' positions randomly within bounds
    position = np.random.uniform(bounds[0], bounds[1], (swarm_size, dim))

    # Step 2: Initialize particles' velocities to zero
    velocity = np.zeros((swarm_size, dim))

    # Step 3: Initialize personal best positions (pbest) as current positions
    pbest = position.copy()

    # Step 4: Evaluate initial personal best values
    pbest_val = np.array([rastrigin(p) for p in position])

    # Step 5: Initialize global best (gbest) based on best pbest value
    gbest = position[np.argmin(pbest_val)]
    gbest_val = np.min(pbest_val)

    # Step 6: Record the best fitness value in the current population
    fitness = np.array([rastrigin(ind) for ind in position])
    best_costs = [np.min(fitness)]
    gen = 0  # Generation counter

    # Step 7: Start iteration process
    while gen < generations and best_costs[-1] > pass_line:

        for i in range(swarm_size):
            # Generate random numbers for stochastic components
            r1 = np.random.rand(dim)  # Random influence on personal best
            r2 = np.random.rand(dim)  # Random influence on global best

            # Step 8: Velocity update based on inertia, cognitive, and social components
            velocity[i] = (
                w * velocity[i] +                            # Inertia component
                c1 * r1 * (pbest[i] - position[i]) +         # Cognitive component (self memory)
                c2 * r2 * (gbest - position[i])              # Social component (swarm memory)
            )

            # Step 9: Update particle position and clip to search bounds
            position[i] = np.clip(position[i] + velocity[i], bounds[0], bounds[1])

            # Step 10: Evaluate new fitness
            val = rastrigin(position[i])

            # Step 11: Update personal best if improved
            if val < pbest_val[i]:
                pbest[i] = position[i]
                pbest_val[i] = val

                # Step 12: Update global best if improved
                if val < gbest_val:
                    gbest = position[i]
                    gbest_val = val

        # Step 13: Record global best value of this generation
        best_costs.append(gbest_val)
        gen += 1  # Move to next generation

    # Step 14: Return global best solution and convergence history
    return gbest, best_costs


# -------------------------
# Run and Plot and Export
# -------------------------

ga_start = time.time()
ga_sol, ga_costs = genetic_algorithm()
ga_end = time.time()

de_start = time.time()
de_sol, de_costs = differential_evolution()
de_end = time.time()

pso_start = time.time()
pso_sol, pso_costs = particle_swarm_optimization()
pso_end = time.time()

# Convergence plot
plt.figure(figsize=(10, 5))
plt.plot(ga_costs, label='Genetic Algorithm (GA)')
plt.plot(de_costs, label='Differential Evolution (DE)')
plt.plot(pso_costs, label='Particle Swarm Optimization (PSO)')
plt.xlabel('Generation')
plt.ylabel('Best Cost')
plt.title('Convergence Curves of GA, DE, and PSO on Rastrigin Function')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# -------------------------
# Export to CSV
# -------------------------
# Save Final_result of all Algorithms for easier access.
Result = (ga_costs[-1], de_costs[-1], pso_costs[-1])

# Create Summary data in CSV
summary_data = [
    ['GA', GA_PARAMS, pass_line, f"{Result[0]:.2e}", len(ga_costs), round((ga_end-ga_start)*1000,3)],
    ['DE', DE_PARAMS, pass_line, f"{Result[1]:.2e}", len(de_costs), round((de_end-de_start)*1000,3)],
    ['PSO', PSO_PARAMS, pass_line, f"{Result[2]:.2e}", len(pso_costs), round((pso_end-pso_start)*1000,3)]
]

summary_columns = ['Algorithm', 'Parameters', 'Pass_line', 'Result', 'Generation', 'time(ms)']
df_summary = pd.DataFrame(summary_data, columns=summary_columns)

# Create History data in CSV
max_len = max(len(ga_costs), len(de_costs), len(pso_costs))
history_data = {
    'Generation': list(range(1, max_len + 1)),
    'GA_Cost': ga_costs + [''] * (max_len - len(ga_costs)),
    'DE_Cost': de_costs + [''] * (max_len - len(de_costs)),
    'PSO_Cost': pso_costs + [''] * (max_len - len(pso_costs)),
}
history_columns = ['Generation','GA_Cost', 'DE_Cost', 'PSO_Cost']
df_history = pd.DataFrame(history_data)

# Export Summary and History data to CSV file.
with open('results_summary.csv', "w", newline='') as f:
    f.write("===== Final Summary =====\n")
    df_summary.to_csv(f, index=False)
    f.write("===== Convergence History =====\n")
    df_history.to_csv(f, index=False)

print("✅ Successfully exported results_summary.csv")

# print(f'GA Best Cost: {ga_costs[-1]} at ({ga_sol[0]}, {ga_sol[1]}) \n'
#       f'DE Best Cost: {de_costs[-1]} at ({de_sol[0]}, {de_sol[1]}) \n'
#       f'PSO Best Cost: {pso_sol[-1]} at ({pso_sol[0]}, {pso_sol[1]}) \n')


print(f'GA Best Cost: {Result[0]:.2e} in {len(ga_costs)} interations within {(ga_end-ga_start)*1000:.3f}ms\n'
      f'DE Best Cost: {Result[1]:.2e} in {len(de_costs)} interations within {(de_end-de_start)*1000:.3f}ms\n'
      f'PSO Best Cost: {Result[2]:.2e} in {len(pso_costs)} interations within {(pso_end-pso_start)*1000:.3f}ms')

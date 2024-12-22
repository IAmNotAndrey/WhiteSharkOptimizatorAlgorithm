import numpy as np
from WhiteSharkOptimizer import WhiteSharkOptimizer

# https://youtu.be/q1U6AmIa_uQ?si=jBj87ktNeKu-73A8&t=259

def get_volume(r, h):
    return np.pi * r**2 * h

def get_area(r, h):
    x_length = 2*np.pi*r
    return 2*np.pi*r**2 + x_length * h

def fitness_func(x):
    volume_to_reach = 1.5
    r = x[0]
    h = x[1]
    volume = get_volume(r, h)
    if volume < volume_to_reach:
        return float('inf')
    # Calculate area.
    return get_area(r, h)

# Example usage
if __name__ == "__main__":
    search_space = [
        (0.1, 10), # x
        (0.1, 20), # y/h
    ]

    # Instantiate and run WSO
    wso = WhiteSharkOptimizer(
        fitness_function=fitness_func,
        search_space=search_space,
        population_size=1000,
        max_iterations=500,
        a0=6.25,
        a1=10,
        a2=5e-4,
    )

    # best_position, best_value = wso.optimize()
    print('Computing. Wait...')
    wso.optimize()
    for i, (global_best_position, best_solution) in enumerate(wso.optimize_result_history):
        print(f"Iteration {i+1}/{wso.max_iterations} - Best Solution: {best_solution}")
    
    print(f"Optimal Position: {wso.global_best_position}")
    print(f"Optimal Value: {wso.best_solution}")
    print(f"Volume: {get_volume(wso.global_best_position[0], wso.global_best_position[1])}")

import numpy as np
from WhiteSharkOptimizer import WhiteSharkOptimizer

def f9(x):
    dim = len(x)
    res = pow(np.sin(np.pi*(1+(x[0]-1/4))),2)
    part2 = 0.
    for i in range(1,dim):
        w = 1 + (x[i-1] - 1/4)
        part2 += np.pow(w, 2) * (1 + 10*np.pow(np.pi*w + 1, 2))
    w = (1 + (x[dim - 1] - 1 / 4))
    res += part2 + (np.pow(w-1, 2)*(1 + np.pow(np.sin(2*np.pi*w), 2)))
    return res

# Example usage
if __name__ == "__main__":
    search_space = [
        (-100, 100) for _ in range(2)
    ]

    # Instantiate and run WSO
    wso = WhiteSharkOptimizer(
        fitness_function=f9,
        search_space=search_space,
        population_size=100,
        max_iterations=10000,
        a0=6.25,
        a1=100,
        a2=5e-4,
    )

    # best_position, best_value = wso.optimize()
    print('Computing. Wait...')
    wso.optimize()
    for i, (global_best_position, best_solution) in enumerate(wso.optimize_result_history):
        print(f"Iteration {i+1}/{wso.max_iterations} - Best Solution: {best_solution}")
    
    print(f"Optimal Position: {wso.global_best_position}")
    print(f"Optimal Value: {wso.best_solution}")

    a = np.array([i[1] for i in wso.optimize_result_history])
    print("median:", np.median(a))
    print("ave:", np.average(a))
    print("worst:", np.max(a))
    print("std:", np.std(a))

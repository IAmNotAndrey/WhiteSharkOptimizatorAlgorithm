import numpy as np
from WhiteSharkOptimizer import WhiteSharkOptimizer

def f3(X):
    part1 = 0.
    part2 = 0.
    for x in X:
        part1+=x*x
        part2+=0.5*x
    res = part1+np.pow(part2,2)+np.pow(part2,4)
    return res

# Example usage
if __name__ == "__main__":
    search_space = [
        (-100, 100) for _ in range(3)
    ]

    # Instantiate and run WSO
    wso = WhiteSharkOptimizer(
        fitness_function=f3,
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

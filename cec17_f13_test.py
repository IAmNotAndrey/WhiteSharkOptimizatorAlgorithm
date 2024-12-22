import numpy as np
from WhiteSharkOptimizer import WhiteSharkOptimizer

def f13(X):
    dim = len(X)
    part2 = 0.
    part3 = 0.
    for x in X:
        part2 += x*x
        part3 += np.cos(np.pi*2*x)
    res = -20 * np.exp(-0.2*np.sqrt((1/dim)*part2))-np.exp((1/dim)*part3)+20+np.e
    return res

# Example usage
if __name__ == "__main__":
    search_space = [
        (-100, 100) for _ in range(2)
    ]

    # Instantiate and run WSO
    wso = WhiteSharkOptimizer(
        fitness_function=f13,
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


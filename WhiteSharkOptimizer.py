from typing import Callable
import numpy as np
import math
from WhiteShark import WhiteShark

class WhiteSharkOptimizer:
    def __init__(self, fitness_function: Callable[[np.ndarray], float],
                       search_space: list[tuple[float, float]], 
                       population_size: int,
                       max_iterations: int,
                       a0: float,
                       a1: float, 
                       a2: float,
                       f_min: float = 0.07,
                       f_max: float = 0.75,
                       p_min: float = 0.5,
                       p_max: float = 1.5,
                       t: float     = 4.125
                ):
        """
        White Shark Optimizer initialization.

        :param fitness_function: the fitness function to minimize.
        :param search_space:     bounds for each dimension [(min, max), ...].
        :param population_size:  the number of white sharks (called 'n' in the paper).
        :param max_iterations:   the maximum number of optimization steps. 
        :param a0:               controls exploration.
        :param a1:               controls exploitation.
        :param a2:               a positive constant utilized to control exploration and exploitation behaviors.
        :param f_min:            min frequencies of the undulating motions.
        :param f_max:            max frequencies of the undulating motions.
        :param p_min:            controls the velocity updates and help maintain a stable balance between global and local search.
        :param p_max:            controls the velocity updates and help maintain a stable balance between global and local search.
        :param t:                an acceleration coefficient that influences the algorithm's behavior.
        """
        self.objective_function = fitness_function
        self.search_space =       search_space
        self.population_size =    population_size
        self.max_iterations =     max_iterations
        self.t = t
        self.p_min = p_min
        self.p_max = p_max
        self.f_min = f_min
        self.f_max = f_max
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        
        self.white_sharks : list[WhiteShark] = None
        self.mu = self._get_mu()
        
        self.lower_bounds = np.array([b[0] for b in search_space])
        self.upper_bounds = np.array([b[1] for b in search_space])
        self.f = self._get_f()    # Velocity scaling factor.

        self.best_solution = None
        self.global_best_position = None

    def _initialize_population(self) -> None:
        '''
        Random initialization of positions and creation of a zero-filled matrix of velocities. \n
        Corresponds to formulas (3,4) of the paper.
        '''
        dim = len(self.search_space)
        self.white_sharks = []
        # Generate shark random positions.
        self.positions = np.random.uniform(
            [s[0] for s in self.search_space], # Minimums. For example: [-2, -3, -1, ...]
            [s[1] for s in self.search_space], # Maximums. For example: [-1, 0, 10, ...]
            (self.population_size, dim)        # Matrix size.
        )
        for pos in self.positions:
            shark = WhiteShark(np.zeros(dim), pos, self.search_space)
            self.white_sharks.append(shark)
        
    def _update_velocity(self, i: int, k: int) -> None:
        '''        
        Corresponds to formula (5) of the paper.
        :param i: a shark index. 
        :param k: an iteration number.
        '''
        c1, c2 = np.random.rand(2)
        shark = self.white_sharks[i]
        # Update.
        shark.velocity = self.mu * (
                shark.velocity + 
                self._get_p1(k) * (self.global_best_position - shark.position) * c1 +
                self._get_p2(k) * (self.global_best_position - self.white_sharks[self._get_nu()].position) * c2 
                # self._get_p2(k) * (self.global_best_position - shark.position) * c2 
            )

    def _get_nu(self) -> int:
        ''' 
        Returns random shark index.
        Corresponds to formula (6) of the paper.
        '''
        return math.floor(self.population_size * np.random.rand())

    def _get_p1(self, k: int) -> float:
        ''' 
        Corresponds to formula (7) of the paper.
        :param k: an iteration number. 
        '''
        exp_value = np.exp(
                -(4 * k / self.max_iterations)**2
        )
        second_part = (self.p_max - self.p_min) * exp_value
        return self.p_max + second_part

    def _get_p2(self, k: int) -> float:
        ''' 
        Corresponds to formula (8) of the paper.
        :param k: an iteration number. 
        '''
        exp_value = np.exp(
                -(4 * k / self.max_iterations)**2
        )
        second_part = (self.p_max - self.p_min) * exp_value
        return self.p_min + second_part

    def _get_mu(self) -> float:
        ''' Corresponds to formula (9) of the paper. '''
        return 2 / (np.abs(2 - self.t - np.sqrt(self.t**2 - 4*self.t)))

    def _update_position(self, i: int, k: int) -> None:
        '''
        Corresponds to formula (10) of the paper.
        :param i: a shark index.
        :param k: an iteration number. 
        '''
        # FIXME
        shark = self.white_sharks[i]
        # Exploration phase.
        if np.random.rand() < self._get_mv(k):
            w0 = self._get_w0(i)
            shark.position = shark.position*np.logical_xor(np.logical_not(w0), w0) \
                + self.upper_bounds*self._get_a(i) \
                + self.lower_bounds*self._get_b(i)
        # Exploitation phase.
        else:
            shark.position += shark.velocity / self.f

    def _get_a(self, i: int) -> list[bool]:
        '''
        Corresponds to formula (11) of the paper.
        :param i: a shark index.
        '''
        shark = self.white_sharks[i]
        return (shark.position - self.upper_bounds) > 0

    
    def _get_b(self, i: int) -> list[bool]:
        '''
        Corresponds to formula (12) of the paper.
        :param i: a shark index.
        '''
        shark = self.white_sharks[i]
        return (shark.position - self.lower_bounds) < 0

    def _get_w0(self, i: int) -> list[bool]:
        '''
        Corresponds to formula (13) of the paper.
        :param i: a shark index.
        '''
        return np.logical_xor(self._get_a(i), self._get_b(i))

    def _get_f(self) -> float:
        ''' Corresponds to formula (14) of the paper. '''
        return self.f_min + (self.f_max - self.f_min) / (self.f_max + self.f_min)

    def _get_mv(self, k) -> float:
        '''
        Corresponds to formula (15) of the paper. 
        :param k: an iteration number.
        '''
        return 1 / (self.a0 + np.exp( (self.max_iterations/2 - k) / self.a1 ))

    def _get_w_stroke_kp1(self, i: int, k: int):
        ''' 
        Corresponds to formula (16) of the paper.
        :param i: a shark index.
        :param k: an iteration number.
        '''
        r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
        shark = self.white_sharks[i]
        # Return zero matrix.
        if r3 >= self._get_ss(k):
            return None
        return self.global_best_position + r1*self._get_shark_prey_distance(i)*np.sign(r2-0.5) 
        
    def _get_shark_prey_distance(self, i: int) -> float:
        '''
        Corresponds to formula (17) of the paper.
        :param i: a shark index.
        '''
        return np.abs( 
            np.random.rand() * (self.global_best_position - self.white_sharks[i].position)
        )

    def _get_ss(self, k: int) -> float:
        '''
        Corresponds to formula (18) of the paper.
        :param k: an iteration number.
        '''
        return np.abs(1 - np.exp(-self.a2 * k / self.population_size))

    def _update_position_with_stroke(self, i: int, k: int) -> None:
        '''
        Corresponds to formula (19) of the paper.
        :param i: a shark index.
        :param k: an iteration index.
        '''
        shark = self.white_sharks[i]
        w_stroke_kp1 = self._get_w_stroke_kp1(i, k)
        if w_stroke_kp1 is None:
            return
        if i == 0:
            shark.position = w_stroke_kp1
        else:
            shark.position = (shark.position + w_stroke_kp1) / (2*np.random.rand())

    def _evaluate_population(self):
        ''' Evaluate fitness and find the best solution. Updates 'best_solution' and 'global_best_position'. '''
        positions = np.array([shark.position for shark in self.white_sharks])
        fitness = np.apply_along_axis(self.objective_function, 1, positions)
        best_index = np.argmin(fitness)
        self.best_solution = fitness[best_index]
        self.global_best_position = positions[best_index].copy()

    def optimize(self):
        ''' Returns: (self.global_best_position, self.best_solution) '''
        self._initialize_population()
        self._evaluate_population()

        for k in range(self.max_iterations):
            for i in range(self.population_size):
                self._update_velocity(i, k)
                self._update_position(i, k)
                self._update_position_with_stroke(i, k)
                # Adjust positions of the white sharks that proceed beyond the boundary.
                self.white_sharks[i].adjust_position()
            
            self._evaluate_population()
            print(f"Iteration {k+1}/{self.max_iterations} - Best Solution: {self.best_solution}")

        return self.global_best_position, self.best_solution


# Example usage
if __name__ == "__main__":
    fitness_functions = [
        lambda x: np.sum(x**2)
    ]

    search_space = [
        (-20,20),
        (-10,10),
        (-0,5),
        (0,3),
        (-10,0),
    ]

    # Instantiate and run WSO
    wso = WhiteSharkOptimizer(
        fitness_function=fitness_functions[0],
        search_space=search_space,
        population_size=100,
        max_iterations=1000,
        a0=6.25,
        a1=100,
        a2=5e-4,
    )

    best_position, best_value = wso.optimize()
    print(f"Optimal Position: {best_position}")
    print(f"Optimal Value: {best_value}")

import random
import numpy as np
import math
import time

def random_solution(n):
    solution = list(range(n))
    random.shuffle(solution)
    return solution

def calculate_distance(tsp, solution):
    distance = 0
    for i in range(len(solution)):
        distance += tsp[solution[i-1]][solution[i]]
    return distance

#evaluation function for 2-opt - will swap improve solution (negative value = shorter distance)
def dist_eval(tsp, n1, n2, n3, n4):
    return tsp[n1][n3] + tsp[n2][n4] - tsp[n1][n2] - tsp[n3][n4]

#2-opt local search
def two_opt(n, current_solution, tsp):
    best_solution = current_solution
    improved = True
    while improved:
        improved = False
        for i in range(1, n):
            for j in range(i + 1, n):
                if j - i == 1: continue
                if dist_eval(tsp, best_solution[i - 1], best_solution[i], best_solution[j - 1], best_solution[j]) < 0:
                    best_solution[i:j] = best_solution[j - 1:i - 1:-1]
                    improved = True
    return best_solution

def random_restart(n, tsp, n_iterations):
    best_solution = []
    best_distance = math.inf
    for i in range(n_iterations):
        solution = random_solution(n)
        current_solution = two_opt(n, solution, tsp)
        current_distance = calculate_distance(tsp, current_solution)
        if best_distance > current_distance:
            best_solution = current_solution
            best_distance = current_distance
    return best_solution, best_distance

def main():
    file = input("file name: ")
    n_iterations = int(input("Number of iterations: "))

    n = int(np.loadtxt(file, max_rows=1))
    tsp = np.loadtxt(file, skiprows=1)

    t0 = time.time()
    print(random_restart(n, tsp, n_iterations))
    t1 = time.time()
    total_t = t1-t0
    print(total_t)

if __name__ == "__main__":
    main()
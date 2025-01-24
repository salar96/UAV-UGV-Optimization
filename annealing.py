import numpy as np
from utils import *
from scipy.optimize import minimize
import time

def optimize_D(objective, init_guess, bounds, beta, method):
        # bounds=(np.min([np.min(drone[0]+drone[1]) for drone in drones]),np.max([np.max(drone[0]+drone[1]) for drone in drones]))*len(init_guess)
        result = minimize(objective, init_guess, args=(beta,), bounds=bounds, method=method)
        # self.params = result.x
        # self.cost_fun=result.fun
        params = result.x
        cost_fun=result.fun
        return params , cost_fun


def anneal(objective , init_stations, bounds, beta_init=1e-6, beta_f=100, alpha=1.5, purturb=0.1, method='powell', verbos=0):
        """ This function performs annealing loop on parameter optimization.
        The input is the initial guess value for stations, and the output is the optimal set
        of parameters and associations. """
        Y_s = []
        Betas = []
        params = np.ndarray.flatten(init_stations)
        beta = beta_init
        old_cost = my_inf
        strt = time.time()
        while beta <= beta_f:
            
            count=0
            params = params + np.random.normal(0, purturb, params.shape)
            params , cost_fun = optimize_D(objective, params, bounds, beta, method=method) 
            count += 1
            if verbos:
              print(f'Beta: {beta:.2e}  F.E.: {cost_fun:0.5e}')
            if abs(cost_fun-old_cost)/abs(old_cost) <= 1e-4:
                print("--Optimization Terminated--")
                break
            old_cost = cost_fun
            beta = beta * alpha
            Y_s.append(params.reshape(-1,2))
            Betas.append(beta)
        print(f"Elapsed time: {time.time()-strt:.2f}")
        return Y_s , Betas
        # self.calc_associations(beta)
        # self.calc_routs()
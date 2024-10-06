# Calculation about the dataset used for training the neural network, 
# This code uses the thermophysical model in our article. 
# You can modify the thermophysical model by changing the class Solve_PDE.

import numpy as np
import random
import scipy.optimize as optimize
from scipy.interpolate import interp1d
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Simulation of the thermophysical model
# xmax, tmax: the maximum values of deepth and time
# nx, nt: the numbers of deepth and time grid
# interp: the interpolation function of the radiation flux
# p: the thermal parameter
# vt: the value of the translation (will be introduced later)

class Solve_PDE():
    def __init__(self, xmax, tmax, nx, nt, interp, p, vt):
        self.xmax = xmax
        self.tmax = tmax
        self.nx = nx
        self.nt = nt
        self.interp = interp
        self.p = p
        self.vt = vt
        
    def __f_fn__(self, t):
        return self.interp(t)
    
    def __equation__(self, x, y, z):
        return x**4-self.p*(y-x)/(self.xmax/self.nx)-z
        
    def process(self):
        deepth, time = self.xmax, self.tmax
        nx, nt = self.nx, self.nt
        dx, dt = deepth/(nx-1), time/(nt-1)
        
        xarray = np.linspace(0, deepth, nx)
        tarray = np.linspace(0, time, nt)
        
        if self.vt <= -5:
            u = np.ones((nx, nt))*0.2
        else:
            u = np.ones((nx, nt))*0.6

        # 设置边界条件
        u[0, :] = optimize.newton(self.__equation__, u[0, :], args=(u[1, :], self.__f_fn__(tarray)))
        u[-1, :] = u[-2, :]

        flag = 1
        for k in range(4000):
            sol = optimize.newton(self.__equation__, u[0, :], args=(u[1, :], self.__f_fn__(tarray)), full_output=True, disp=False)
            u[0, :] = sol[0]
            u[-1, :] = u[-2, :]
            if False in sol[1]:
                flag = 0
                break
            for j in range(nt-1):
                u[:, 0] = u[:, nt-1]
                u[1:-1, j+1] = (1-2*dt/dx**2)*u[1:-1, j]+dt/dx**2*(u[2:, j]+u[:-2, j])
            if np.abs(np.mean(u[:, 0]-u[:, nt-1]))<1e-5:
                break
            
        return u, xarray, tarray, flag

# Generation of the random radiation flux function
# vt: the value of the translation beform inputting to sigmoid function

class Random_Flux():
    def __init__(self, vt):
        self.vt = vt
    
    def __Kernel__(self, x1, x2, params):
        output_scale, lengthscales = params
        diffs = np.expand_dims(x1, 1)-np.expand_dims(x2, 0)
        r2 = np.sum(diffs**2, axis=2)/lengthscales**2
        return output_scale*np.exp(-0.5*r2)

    def __sigmoid__(self, x):
        return 1/(1+np.exp(-x))

    def generate_func(self):
        N = 800
        X1 = np.linspace(0, 1, N)[:, None]
        X2 = np.linspace(0, 1, N)[:, None]
        K = self.__Kernel__(X1, X2, (1, 0.2))
        L = np.linalg.cholesky(K+1e-10*np.eye(N))
        X = np.linspace(0, 2*np.pi, N).flatten()
    
        gp_sample = np.dot(L, np.random.normal(0, 1, size=N))
        normalized_gp_sample = self.__sigmoid__(gp_sample+self.vt)
    
        # Randomly select certain ranges as 0
        # in order to simulate the shadow
        num_segments = np.random.randint(0, 6)
        segments = []
        for _ in range(num_segments):
            start = np.random.uniform(np.min(X), np.max(X))
            length = np.random.uniform(0.1, 2.0)
            end = min(start+length, np.max(X))
            segments.append((start, end))
        for start, end in segments:
            normalized_gp_sample[(X>=start)&(X<=end)] = 0
    
        interp1 = interp1d(np.linspace(0, 2*np.pi, N).flatten(), normalized_gp_sample)
        f_fn_number = interp1
        
        return f_fn_number

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    vt = 0 # you can change the value of translation (-6, -3, 0, 3, 6 are suggested)
    p = random.uniform(0, 10) # you can modify the scale ([0, 0.4], [0.4, 10], [10, 100] are suggested)
    Random_interp_func = Random_Flux(vt).generate_func()
    
    # Grid parameters (all can be modified)
    xmax = 10
    tmax = 2*np.pi
    nx = 40
    nt = 360
    
    PDE_Solution = Solve_PDE(10, 2*np.pi, 40, 360, Random_interp_func, p, vt)
    u_num, xarray, tarray, flag = PDE_Solution.process()
    interp = interp1d(xarray, u_num[:, 0])
    pde_num_u = interp
    
    if flag == 1:
        subdata = [Random_interp_func, pde_num_u, p]
        print(subdata)
    else:
        print('No solution!')
        
    # The subdata of the dataset is combined with:
    # Random_interp_func: the interpolation function of random radiation flux
    # pde_num_u: the interpolation function of temperature about deepth
    # p: random thermal parameter
    
    # You can loop through the above code to obtain a sufficient number of training sets.
    # We will introduce the usage of the dataset calculated here in the code about training

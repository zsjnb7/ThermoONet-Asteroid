# ThermoONet_Asteroid: A Deep Learning Framework for Asteroid Thermophysical Modeling and Yarkovsky Effect
ThermoONet_Asteroid is a PyTorch-based deep learning framework designed for asteroid thermophysical modeling and yarkovsky Effect. It uses a modified deep operator neural network architecture with channel attention mechanisms to model the thermal behavior of asteroid and estimate yarkovsky effect.
## Project Structure
The project consists of four main Python files:
1. Dataset Generation for Training (```dataset_making.py```)
2. DeepONet Architecture Definition (```network.py```)
3. Training Procedure (```training.py```)
4. Yarkovsky Calculation with DeepONet (```DeepONet and Application.ipynb```)
## Requirements
* Python 3.8+
* PyTorch (GPU version recommended)
* NumPy
* SciPy
* Matplotlib
* scikit-learn  

For GPU acceleration, please install the PyTorch version that matches your GPU's CUDA capability.
## Usage
### 1. Dataset Generation
Generate training data for the DeepONet model:
```python
from dataset_generation import Random_Flux, Solve_PDE

# Generate random flux function
vt = 0  # translation parameter (-6, -3, 0, 3, 6 recommended)
random_flux = Random_Flux(vt).generate_func()

# Solve PDE with generated flux
p = random.uniform(0, 10)  # thermal parameter
pde_solution = Solve_PDE(10, 2*np.pi, 40, 360, random_flux, p, vt)
u_num, xarray, tarray, flag = pde_solution.process()
```
### 2. Model Training
Train the DeepONet model:
```python
from training import Train_Dataset
from network_architecture import Branch, Trunk

# Prepare data loader
traindataset = Train_Dataset(num_sensors=128, num_input=128, batchsize=400, data_fup=data_fup)
train_loader, y_train = traindataset.data_loader()

# Initialize and train model
branch = Branch(num_sensors).cuda()
trunk = Trunk().cuda()
# ... training loop (see training.py for details)
```
### 3. Yarkovsky Effect Calculation
The main functionality calculates Yarkovsky acceleration for asteroids:
```python
from yarkovsky_calculation import Yarkovsky_database

# Initialize with pre-trained models and parameters
yd = Yarkovsky_database(branch, trunk, para, shape)
r_range = [0.05, 2.7, 128]  # min_r, max_r, num_points
yarkovsky_acc = yd.yarkovsky(r_range)
```
## File Details
### 1. dataset_generation.py
This file generates training data by:
* Creating random radiation flux functions using Gaussian processes
* Solving the heat conduction PDE for asteroid surfaces
* Generating pairs of input functions and output temperature profiles  

Key parameters:
* ```vt```: Translation parameter for sigmoid function
* ```p```: Thermal parameter (different ranges suggested for different regimes)
* Grid parameters: ```xmax```, ```tmax```, ```nx```, ```nt```
### 2. network_architecture.py
Defines the DeepONet architecture:
* SELayer: Channel attention mechanism
* Branch networks: Process radiation flux and thermal parameters
* Trunk network: Processes spatial coordinates
* Custom initialization methods
### 3. training.py
Implements:
* Data preprocessing and normalization
* Training dataset class
* Training loop with learning rate scheduling
* Model evaluation
### 4. yarkovsky_calculation.py
This file contains:
* DeepONet architecture with channel attention mechanisms
* Branch and trunk network definitions
* Yarkovsky database calculation class
* Functions for processing asteroid shape data
* Methods for calculating Yarkovsky acceleration in spherical coordinates 
## Example: Asteroid (3200) Phaethon
The code includes an example for calculating Yarkovsky effects on asteroid Phaethon:
```python
# Shape data files
Phaethon1 = "test_asteroid/Phaethon.txt"
Phaethon2 = "test_asteroid/Phaethon2.txt"
shape = [Phaethon1, Phaethon2]

# Physical parameters for Phaethon
omega = 2*np.pi/(3.6*3600)  # rotation speed
A = 0.122                   # Bond albedo
eps = 0.9                   # emissivity
# ... additional parameters
para = [omega, A, eps, S, sig, rho, C, kapa, R, m]

# Calculate Yarkovsky acceleration
yd = Yarkovsky_database(branch, trunk, para, shape)
yarkovsky_acc = yd.yarkovsky([0.05, 2.7, 128])
```
## Output
The Yarkovsky acceleration output has dimensions [beta, yita, r, 3], representing:
* ```beta```: Angle between the Sun and z-axis
* ```yita```: Angle between the Sun and x-axis
* ```r```: Heliocentric distance
* ```3```: Acceleration components (```ax```, ```ay```, ```az```) in the spin axis coordinate system

## Recommendations
1. For accurate results, retrain the network on your specific problem domain
2. Use GPU acceleration for significantly faster computation
3. Adjust grid sizes (```num_b```, ```num_y```, ```num_r```) based on accuracy requirements
4. Modify thermal parameter ranges based on asteroid characteristics

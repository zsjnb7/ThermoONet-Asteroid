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

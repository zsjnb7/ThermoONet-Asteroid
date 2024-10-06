# DeepONet-for-asteroids-temperature
## DeepONet and Training (folder Network_Generate)
These codes are for training a neural network and consists of three parts.  
1. **dataset_making.py**: the calculation for the dataset used for training the neural network, which uses the thermophysical model in our article. You can modify the thermophysical model by changing ```class Solve_PDE ```.
2. **network.py**: the architecture of DeepONet.
3. **training.py**: the program for training the network, ```class Train_Dataset``` is used for data preprocessing, where the original data comes from dataset_making.py.
4. **dataset/test_data**: an example of training data, needs to be unzipped.
## DeepONet and Yarkovsky effect (folder Application)
This code is for the DeepONet-based database of Yarkovsky effect applied to the asteroid orbital evolution in N-body system.  
1. **DeepONet and Application.ipynb**: the architecture of DeepONet and the calculation method for the database of Yarkovsky effect.  
2. **net_data**: the trained branch network and trunk network that can be directly applied, but they are only useful for understanding the process. We really suggest you to retrain the network through the folder **Network_Generate**.
3. **test_asteroid**: the shape data of Phaethon for testing.  
## Important
Note that it is necessary to run the code in GPU for rapid computation. This requires to install the PyTorch that matches with your own computer's GPU and you can get help from PyTorch official.  

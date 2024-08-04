# DeepONet-for-asteroids-temperature
This code is for the DeepONet-based database of Yarkovsky effect applied to the asteroid orbital evolution in N-body system.  
The code **DeepONet and Application.ipynb** contains the architecture of DeepONet and the calculation method for the database of Yarkovsky effect.  
The folder **net_data** contains the trained branch network and trunk network that can be directly applied in practice cases.  
The folder **test_asteroid** contains the shape data of Phaethon for testing.  
Note that it is necessary to run the code in GPU for rapid computation. This requires to install the PyTorch that matches with your own computer's GPU and you can get help from PyTorch official.  
Please see comments in the code for specific usage instructions.

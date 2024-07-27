import pdb
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from PreProcessing import PreprocessData

# Set Seeds For Randomness
torch.manual_seed(10)
np.random.seed(10)    
InputSize = 6  # Input Size
batch_size = 1 # Batch Size Of Neural Network
NumClasses = 1 # Output Size 
NumEpochs = 25
HiddenSize = 10

# Create The Neural Network Model
class Net(nn.Module):
    def __init__(self, InputSize,NumClasses):
        super(Net, self).__init__()
		#Define The Feed Forward Layers
        self.fc1 = nn.Linear(InputSize, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10,NumClasses)
        
    def forward(self, x):
		#Steps For Forward Pass
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

net = Net(InputSize, NumClasses)     

criterion = nn.MSELoss() # mean square error
optimizer = torch.optim.SGD(net.parameters(), lr=0.01) # optimize hidden layer by changing the parameter (live) to make it more accurate

if __name__ == "__main__":
        
    TrainSize,SensorNNData,SensorNNLabels = PreprocessData()   
    for j in range(NumEpochs):
        losses = 0
        
        for i in range(TrainSize):  
            input_values = Variable(SensorNNData[i])
            labels = Variable(SensorNNLabels[i])
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = net(input_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses += loss.item()
            
        print ('Epoch %d, Loss: %.4f' %(j+1, losses/SensorNNData.shape[0]))       
        torch.save(net.state_dict(), r'D:\Maze Robot Game\SavedNets\NNBot2.pkl')
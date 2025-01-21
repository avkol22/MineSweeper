
import torch
import torch.nn as nn
import torch.optim as optim



gamma=0.9
batch_size = 100
learning_rate = 0.0001
weight_decay = 0.
test_games=2000
buffer_len = 100000 ### tidligere 100000 lave den længere??
counter_max = 2000  
epsilon = 1
epsilon_min = 0.01
epsilon_reduction_factor = 0.01**(1/60000) ## tidligere 60000
iteration_max=400000
iteration_period=10000
steps_per_gradient_update = 10
max_episode_step = 36  # baseret på størrelse af pladen (dette for en 6X6 plade)

class Network2(nn.Module):
    def __init__(self, size):
        super(Network2, self).__init__()
        self.conv1 = nn.Conv2d(10, 24, kernel_size= 3, stride=1, padding = 1)  # 
        self.conv2 = nn.Conv2d(24, 16, kernel_size= 3, stride=1, padding = 1) #
        self.conv3 = nn.Conv2d(16, 64, kernel_size= 3, stride=1, padding = 1) #
        self.conv4 = nn.Conv2d(64, 1, kernel_size= 3, stride=1, padding = 1) #
        self.fc1 = nn.Linear(size**2,32)
        self.fc2 = nn.Linear(32,size**2)  # For at får returneret værdier tilsvarende til boarded 
        self.relu = nn.LeakyReLU()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        self.size = size
        self.loss_function = torch.nn.MSELoss()
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def add_buffer(self):
        self.buff_state = torch.zeros(size=(buffer_len,10,self.size,self.size))
        self.buff_new_state = torch.zeros(size=(buffer_len,10,self.size,self.size))
        self.buff_action = torch.zeros(size = (buffer_len,2))
        self.buff_reward = torch.zeros(size=(buffer_len,1))
        self.buff_is_terminal = torch.zeros(size = (buffer_len,1))


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))  
        x = torch.flatten(x,1)
        x = self.relu(self.fc1(x))
        x=self.fc2(x)
        return x
    
    def update_weights(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


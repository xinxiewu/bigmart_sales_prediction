"""
deep_learning.py contains neural networks:
    1. MyNetwork
    2. Training process
"""
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import torch
from util import *

class MyNetwork(nn.Module):
    def __init__(self, input_feature, dim1, dim2, drop, output_feature):
        super(MyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_feature, dim1, bias=True),
            nn.Dropout(drop),
            nn.ReLU(),
            nn.Linear(dim1, dim2, bias=True),
            # nn.Dropout(drop),
            nn.ReLU(),
            nn.Linear(dim2, 6, bias=True),
            # nn.Dropout(drop),
            nn.ReLU(),
            nn.Linear(6, output_feature, bias=True)
        )

    def forward(self, input):
        return self.model(input)
    
def training_nn(epochs=500, learning_rate=0.01, loss_func='MSE', optimizer='Adam', input_feature=35, dim1=64, dim2=16, dropout=0.2, output_feature=1,
                train_loader=None, test_loader=None, device=None, save=False, output=None):
    # Initialize model
    mynetwork = MyNetwork(input_feature=input_feature, dim1=dim1, dim2=dim2, drop=dropout, output_feature=output_feature)
    mynetwork.to(device)

    # Define loss function & optimizer
    if loss_func == 'MSE':
        loss_fn = nn.MSELoss()
        loss_fn.to(device)
    if optimizer == 'Adam':
        optimizer = optim.Adam(mynetwork.parameters(), lr=learning_rate)

    # Epoches, training & testing
    total_training_step, total_testing_step = 0, 0
    writer = SummaryWriter(output)
    for i in range(epochs):
        if (i+1)%100 == 0:
            print(f"------------epoch: {i+1}------------")

        """
        Training
        """
        mynetwork.train()
        for data in train_loader:
            input, output = data
            input, output = input.to(device), output.to(device)
            outputs = mynetwork(input)
            loss = loss_fn(outputs, output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            

            total_training_step += 1
            if total_training_step % 100 == 0:
                print(f"Traning: {total_training_step}; Loss: {loss.item()}")
                writer.add_scalar("train_loss", loss.item(), total_training_step)

        """
        Testing
        """
        mynetwork.eval()
        total_test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                input, output = data
                input, output = input.to(device), output.to(device)
                outputs = mynetwork(input)
                loss = loss_fn(outputs, output)
                total_test_loss += loss.item()
        print(f"total test loss: {total_test_loss}")
        writer.add_scalar("test_loss", total_test_loss, total_testing_step)
        total_testing_step += 1
        if save == True:
            torch.save(mynetwork, f"mynetwork_{i}.pth")

    writer.close()
    return
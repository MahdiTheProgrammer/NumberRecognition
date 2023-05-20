from torchvision import transforms
import torch
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import random
from PIL import Image
from torch import nn 
import torch.optim as optim
# Define the transformation to convert images to tensors
transform =  transforms.ToTensor()

# Initialize the 2D list to store the image tensores
image_tensors = [[None]*10772 for _ in range(10)]

# Loop over the folders and images to load the tensores
for x in range(10):
    for y in range(1000):
        # Load the image and apply the transformation
        img = Image.open(f"dataset/{x}/{x}/{y}.png")
        tensor = transform(img)
        # Store the tensor in the 2D list
        image_tensors[x][y] = tensor
        # Assuming your picture tensor is named "picture"
        batch_size = image_tensors[x][y].size(0)  # Get the batch size
        num_features = image_tensors[x][y].size(1) * image_tensors[x][y].size(2)  # Calculate the total number of features
        # Reshape the tensor to (batch_size, num_features)
        image_tensors[x][y] = image_tensors[x][y].view(batch_size, num_features)

# Split data    
train = [image_tensors[i][:800] for i in range(10)]
test = [image_tensors[i][800:] for i in range(10)]
print(len(test))  


# Lets create our model
class NumberRecognition(nn.Module):
    def __init__(self, num_features):
        super(NumberRecognition, self).__init__()
        self.layer_1 = nn.Linear(in_features = num_features, out_features = 700 )
        self.layer_2 = nn.Linear(in_features = 700, out_features = 10)
    def forward(self, x):
        return self.layer_2(self.layer_1(x))


# Lets create our train loop

num_classes = 10  # We have 10 classes for digit recognition (0-9)

# Define خعق neural network model
model = NumberRecognition(num_features)  
model = torch.load("save.pt")

# Define the loss function
criterion = nn.CrossEntropyLoss()
import random 
# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)  # You can adjust the learning rate as needed
epochs = 8000
folder = 0 
pic = 0 
train_loss_values = []
epoch_count = []
# Training loop
for epoch in range(epochs):


    model.train()
    x = random.randint(0,9)
    inputs = train[x][pic-1]
    
    outputs = model(train[x][pic-1])

    labels = torch.tensor([x] * outputs.size(0))
    
    loss = criterion(outputs, labels)
    print(loss)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    
    if (epoch)%800==0:
        folder +=1
        pic = 0 
        
    pic+=1
    train_loss_values.append(loss.detach().numpy())
    epoch_count.append(epoch)
    


    



torch.save(model, 'save.pt')

with torch.inference_mode():
    for x in range(10):

        outputs = model(test[x][0])
        labels = torch.tensor([x] * inputs.size(0))
        loss = criterion(outputs, labels)
        print(f"Loss of number {x} is {loss}")

plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.title("Training curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()
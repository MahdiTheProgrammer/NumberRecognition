from torchvision import transforms
import torch
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import random
from PIL import Image
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

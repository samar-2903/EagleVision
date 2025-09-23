import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network built from scratch using PyTorch.
    This model is designed for a basic image classification task.
    It accepts 3-channel (RGB) images of size 64x64 pixels.
    """
    def __init__(self, num_classes=10):
        """
        Initializes the layers of the network.
        
        Args:
            num_classes (int): The number of distinct object categories to identify.
        """
        super(SimpleCNN, self).__init__()
        
        # --- Convolutional Layer Block 1 ---
        # Input: 3 channels (RGB), Output: 16 channels
        # Kernel size: 3x3, Padding: 1 to maintain image dimensions
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Max pooling layer to reduce spatial dimensions by half
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 64x64 -> 32x32
        
        # --- Convolutional Layer Block 2 ---
        # Input: 16 channels, Output: 32 channels
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16

        # --- Convolutional Layer Block 3 ---
        # Input: 32 channels, Output: 64 channels
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 16x16 -> 8x8
        
        # --- Fully Connected (Classifier) Layers ---
        # The input features to the linear layer are calculated as:
        # output_channels * pooled_height * pooled_width = 64 * 8 * 8 = 4096
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        This is where the data flows through the layers.
        
        Args:
            x (torch.Tensor): The input batch of images.
                               Shape: (batch_size, 3, 64, 64)
                               
        Returns:
            torch.Tensor: The output logits for each class.
                          Shape: (batch_size, num_classes)
        """
        # Pass through Conv Block 1
        x = self.pool1(F.relu(self.conv1(x)))
        
        # Pass through Conv Block 2
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Pass through Conv Block 3
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Flatten the feature maps to a vector for the fully connected layers
        # The view function reshapes the tensor. -1 tells PyTorch to infer the batch size.
        x = x.view(-1, 64 * 8 * 8)
        
        # Pass through the first fully connected layer with ReLU and Dropout
        x = self.dropout(F.relu(self.fc1(x)))
        
        # Pass through the output layer
        # No activation function here, as it will be handled by the loss function (e.g., CrossEntropyLoss)
        x = self.fc2(x)
        
        return x

def train_model(model, num_epochs=5):
    """
    A dummy training function to simulate training the model.
    In a real scenario, you would use a real, labeled dataset.
    """
    print("\n--- Starting Dummy Training ---")
    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # In a real scenario, you would use a DataLoader here.
    # We will simulate it with random data.
    for epoch in range(num_epochs):
        # Create a dummy batch of images and labels
        dummy_images = torch.randn(16, 3, 64, 64) # Batch of 16 images
        dummy_labels = torch.randint(0, model.fc2.out_features, (16,)) # Random labels

        # Training step
        optimizer.zero_grad()
        outputs = model(dummy_images)
        loss = criterion(outputs, dummy_labels)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    print("--- Dummy Training Finished ---")
    return model

def predict_image(model, image_path, class_names):
    """
    Loads an image, preprocesses it, and makes a prediction using the trained model.
    """
    print(f"\n--- Making Prediction on '{image_path}' ---")
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: '{image_path}' not found. Please place the image in the same directory.")
        print("Using a random dummy image for demonstration instead.")
        # Create a dummy PIL image if the file is not found
        image = Image.fromarray((torch.randn(64, 64, 3).numpy() * 255).astype('uint8'), 'RGB')

    # Define the same transformations used for training
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Use same normalization as training
    ])
    
    # Preprocess the image and add a batch dimension
    image_tensor = transform(image).unsqueeze(0)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Make a prediction
    with torch.no_grad():
        output = model(image_tensor)
        
    # Get the predicted class index
    _, predicted_idx = torch.max(output, 1)
    predicted_class = class_names[predicted_idx.item()]
    
    print(f"\nPrediction Complete:")
    print(f"The model predicts the main object in the image is a: '{predicted_class}'")


# --- Demonstration of Training and Inference ---
if __name__ == "__main__":
    # Define the classes your model will learn to identify
    STREET_CLASSES = [
        'car', 'pedestrian', 'truck', 'bicycle', 'traffic_light', 
        'building', 'road_sign', 'bus', 'tree', 'motorcycle'
    ]
    NUM_CLASSES = len(STREET_CLASSES)
    
    # 1. Instantiate the model
    # The weights are initialized randomly by PyTorch; it is not pre-trained.
    model = SimpleCNN(num_classes=NUM_CLASSES)
    print("--- Model Architecture ---")
    print(model)
    
    # 2. Train the model (using dummy data for this example)
    # In a real project, you would train it on thousands of labeled images.
    trained_model = train_model(model, num_epochs=5)
    
    # 3. Make a prediction on your target image
    # Make sure 'streets.png' is in the same folder as this script.
    image_to_predict = 'streets.png'
    predict_image(trained_model, image_to_predict, STREET_CLASSES)


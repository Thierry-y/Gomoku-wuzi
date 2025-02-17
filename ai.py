import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import matplotlib.pyplot as plt

# Configure the device (use GPU if available, otherwise use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom dataset class
class GoDataset(Dataset):
    def __init__(self, x_path, y_path, size=7541):
        self.x_data = np.zeros((size, 361), dtype=np.float32)
        self.y_data = np.zeros((size, 1), dtype=np.float32)
        
        # Load feature data
        with open(x_path, 'r') as f:
            for i, line in enumerate(f):
                self.x_data[i] = np.array(
                    [int(s) for s in re.findall(r'\d+', line)],
                    dtype=np.float32
                )
        
        # Load label data
        with open(y_path, 'r') as f:
            for i, line in enumerate(f):
                self.y_data[i] = float(line.strip())
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.x_data[idx], device=device),
            torch.tensor(self.y_data[idx], device=device)
        )

# Neural network model definition
class NuNet(nn.Module):
    def __init__(self):
        super(NuNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(361, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.ReLU()  
        )
    
    def forward(self, x):
        return self.layers(x)

# Training logger class to track loss and MAE
class TrainingLogger:
    def __init__(self):
        self.losses = {'batch': [], 'epoch': []}
        self.mae = {'batch': [], 'epoch': []}
    
    def update_batch(self, loss, mae):
        self.losses['batch'].append(loss)
        self.mae['batch'].append(mae)
    
    def update_epoch(self, loss, mae):
        self.losses['epoch'].append(loss)
        self.mae['epoch'].append(mae)
    
    def plot_progress(self):
        plt.figure(figsize=(12, 5))
        
        # Loss curve
        plt.subplot(1, 2, 1)
        plt.plot(self.losses['epoch'], 'g-', label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend()
        
        # MAE curve
        plt.subplot(1, 2, 2)
        plt.plot(self.mae['epoch'], 'r-', label='MAE')
        plt.title('Mean Absolute Error')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Training function
def train_model():
    # Initialize dataset and dataloader
    dataset = GoDataset("data/x_train.txt", "data/y_train.txt")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    model = NuNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    logger = TrainingLogger()
    
    # Training loop
    for epoch in range(200):
        epoch_loss = 0.0
        epoch_mae = 0.0
        
        for inputs, labels in dataloader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record batch metrics
            batch_mae = torch.mean(torch.abs(outputs - labels)).item()
            logger.update_batch(loss.item(), batch_mae)
            
            epoch_loss += loss.item() * inputs.size(0)
            epoch_mae += batch_mae * inputs.size(0)
        
        # Compute epoch metrics
        epoch_loss /= len(dataset)
        epoch_mae /= len(dataset)
        logger.update_epoch(epoch_loss, epoch_mae)
        
        print(f'Epoch {epoch+1}/200 - Loss: {epoch_loss:.4f} - MAE: {epoch_mae:.4f}')
    
    # Save the trained model
    torch.save(model.state_dict(), 'go_model.pth')
    print("Model saved to go_model.pth")
    
    # Plot training curves
    logger.plot_progress()
    
    return model

# Model evaluation function
def evaluate_model(model, dataset, criterion):
    dataloader = DataLoader(dataset, batch_size=64)
    total_loss = 0.0
    total_mae = 0.0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item() * inputs.size(0)
            total_mae += torch.mean(torch.abs(outputs - labels)).item() * inputs.size(0)
    
    avg_loss = total_loss / len(dataset)
    avg_mae = total_mae / len(dataset)
    print(f'Evaluation - Loss: {avg_loss:.4f} - MAE: {avg_mae:.4f}')

class Model:
    def __init__(self, model_path='go_model.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = NuNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  

    def get_score_ANN(self, board, turn):
        # Convert the Go board into a flattened input format for the neural network
        x_input = np.zeros((1, 361), dtype=np.float32)
        opponent = 2 if turn == 1 else 1  
        
        for i in range(19):
            for j in range(19):
                pos = i * 19 + j  # Convert 2D position to 1D index
                val = board[i][j]  # Get board value at the current position
                
                if val == turn:
                    x_input[0, pos] = 1  # Mark player's piece
                elif val == opponent:
                    x_input[0, pos] = 2  # Mark opponent's piece
                else:
                    x_input[0, pos] = 0  # Mark empty space
        
        # Convert the input array to a PyTorch tensor and move it to the appropriate device
        x_tensor = torch.tensor(x_input, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():  # Disable gradient computation for efficiency
            prediction = self.model(x_tensor)  # Get model prediction
            score = prediction.item() * 20000 - 10000  # Scale the output to a meaningful range
        
        # Ensure the score is within the valid range (-10000 to 10000)
        return max(min(int(score), 10000), -10000)

if __name__ == "__main__":
    # Train the model
    trained_model = train_model()

    # Load model
    # model = NuNet().to(device)  
    # model.load_state_dict(torch.load("go_model.pth", map_location=device))  
    # model.eval()
    
    # Load test dataset
    dataset = GoDataset("data/x_test.txt", "data/y_test.txt")
    criterion = nn.MSELoss()
    
    # Evaluate the trained model
    # evaluate_model(model, dataset, criterion)
    evaluate_model(trained_model, dataset, criterion)

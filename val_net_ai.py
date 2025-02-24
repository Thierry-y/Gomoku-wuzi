import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import matplotlib.pyplot as plt

# Configure the device (use GPU if available, otherwise use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom dataset class for loading Go board data
class GoDataset(Dataset):
    def __init__(self, x_path, y_path, size=7541):
        # Initialize feature and label arrays
        self.x_data = np.zeros((size, 1, 19, 19), dtype=np.float32)  # Go board features
        self.y_data = np.zeros((size, 1), dtype=np.float32)          # Corresponding labels

        # Load feature data from file
        with open(x_path, 'r') as f:
            for i, line in enumerate(f):
                board = np.array(
                    [int(s) for s in re.findall(r'\d+', line)],
                    dtype=np.float32
                ).reshape(1, 19, 19)
                self.x_data[i] = board

        # Load label data from file
        with open(y_path, 'r') as f:
            for i, line in enumerate(f):
                self.y_data[i] = float(line.strip())

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        # Return feature and label tensors
        return (
            torch.from_numpy(self.x_data[idx]).to(device),
            torch.from_numpy(self.y_data[idx]).to(device)
        )

# Neural network model definition
class VaNet(nn.Module):
    def __init__(self):
        super(VaNet, self).__init__()
        # Convolutional layers for feature extraction
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Fully connected layers for final prediction
        self.fc = nn.Sequential(
            nn.Linear(128 * 19 * 19, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Forward pass through the network
        x = x.view(-1, 1, 19, 19)  # Reshape input to match conv layer
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the output for FC layers
        x = self.fc(x)
        return x

# Training logger class to track losses and MAE
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
        # Plot training loss and MAE over epochs
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.losses['epoch'], 'g-', label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend()

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
    # Load training data
    dataset = GoDataset("data/x_train.txt", "data/y_train.txt")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize model, loss function, optimizer, and scheduler
    model = VaNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    logger = TrainingLogger()

    # Training loop
    for epoch in range(200):
        epoch_loss = 0.0
        epoch_mae = 0.0

        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_mae = torch.mean(torch.abs(outputs - labels)).item()
            logger.update_batch(loss.item(), batch_mae)

            epoch_loss += loss.item() * inputs.size(0)
            epoch_mae += batch_mae * inputs.size(0)

        scheduler.step()

        epoch_loss /= len(dataset)
        epoch_mae /= len(dataset)
        logger.update_epoch(epoch_loss, epoch_mae)

        print(f'Epoch {epoch+1}/200 - Loss: {epoch_loss:.4f} - MAE: {epoch_mae:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'go_model_alphazero.pth')
    print("Model saved to go_model_alphazero.pth")
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

# Inference class for using the trained model
class Model_new:
    def __init__(self, model_path='go_model_alphazero.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VaNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def get_score_ANN(self, board, turn):
        # Convert Go board to (1, 1, 19, 19) format
        x_input = np.zeros((1, 1, 19, 19), dtype=np.float32)
        opponent = 2 if turn == 1 else 1

        for i in range(19):
            for j in range(19):
                val = board[i][j]
                if val == turn:
                    x_input[0, 0, i, j] = 1  # Mark current player's stones
                elif val == opponent:
                    x_input[0, 0, i, j] = 2  # Mark opponent's stones
                else:
                    x_input[0, 0, i, j] = 0  # Empty spaces

        # Convert to Tensor and move to device
        x_tensor = torch.tensor(x_input, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            prediction = self.model(x_tensor)  # Get model prediction
            score = prediction.item()  # Directly use the model output

        # Uncomment to clamp score between 0 and 1 if needed
        # score = max(min(score, 1.0), 0.0)
        print(score)
        return score

if __name__ == "__main__":
    trained_model = train_model()
    test_dataset = GoDataset("data/x_test.txt", "data/y_test.txt")
    evaluate_model(trained_model, test_dataset, nn.MSELoss())

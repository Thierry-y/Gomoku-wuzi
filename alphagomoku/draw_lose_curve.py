import json
import matplotlib.pyplot as plt
import os

# Define TrainingLogger class
class TrainingLogger:
    def __init__(self):
        self.losses = {'batch': [], 'epoch': []}
    
    def update_batch(self, loss):
        self.losses['batch'].append(loss)
    
    def update_epoch(self, loss):
        self.losses['epoch'].append(loss)
    
    def plot_progress(self):
        plt.figure(figsize=(8, 5))
        
        # Plot the loss curve
        plt.plot(self.losses['epoch'], 'g-', label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()


# Load train_losses.json file
def load_loss_data(file_path):
    """
    Load loss data from a JSON file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# Main program
if __name__ == '__main__':
    # File path
    log_dir = 'log'
    file_name = 'train_losses.json'
    file_path = os.path.join(log_dir, file_name)
    
    try:
        # Load loss data
        loss_data = load_loss_data(file_path)
        
        # Create a logger object and update data
        logger = TrainingLogger()
        for epoch, loss in loss_data:
            logger.update_epoch(loss)
        
        # Plot the loss curve
        logger.plot_progress()
    
    except FileNotFoundError as e:
        print(e)
    except json.JSONDecodeError:
        print(f"File {file_path} is not a valid JSON format")
    except Exception as e:
        print(f"An error occurred: {e}")

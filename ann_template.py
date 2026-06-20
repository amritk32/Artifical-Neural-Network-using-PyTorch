import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = fetch_california_housing()
X = df.data
y = df.target
BATCH_SIZE = 256

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Scale dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# Mini Batch
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)


# Create Neural Network
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initialize layers
        self.layer1 = nn.Linear(8, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 16)
        self.layer4 = nn.Linear(16, 1)

        # Batch Normalizers
        self.normalizer1 = nn.BatchNorm1d(256)
        self.normalizer2 = nn.BatchNorm1d(128)
        self.normalizer3 = nn.BatchNorm1d(16)

        # Dropout Regularization
        self.dropout = nn.Dropout(p=0.2)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.normalizer1(self.layer1(x))))
        x = self.dropout(self.relu(self.normalizer2(self.layer2(x))))
        x = self.dropout(self.relu(self.normalizer3(self.layer3(x))))
        # Final Output Layer 
        x = self.layer4(x)
        return x
    

# Initialize Network , Optimizer and Loss function
ann = ANN()
ann.to(device)
optimizer = torch.optim.Adam(ann.parameters(), lr=0.001)
criterion = nn.MSELoss()


# Different Optimizers
sgd_optimizer = torch.optim.SGD(ann.parameters(), lr=0.01, momentum=0.9)
nesterov_optimizer = torch.optim.SGD(ann.parameters(), lr=0.01, momentum=0.9, nesterov=True)
rms_optimizer = torch.optim.RMSprop(ann.parameters(), lr=0.01, alpha = 0.99)


# Training Loop
def train(epochs: int):
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):     
            # Train ANN
            ann.train()
            running_train_loss = 0.0
            
            # Mini Batch Loop
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()   # Reset Gradients
                outputs = ann(batch_X)  # Forward Propagation 
                loss = criterion(outputs, batch_y)   # Calculate Loss
                loss.backward()         # Backward Propagation
                optimizer.step()        #Gradient Descent

                running_train_loss += loss.item() * batch_X.size(0) # Aggregated Loss for each batch size
            
            # Average training loss per epocch
            epoch_train_loss = running_train_loss / len(train_loader.dataset)

            # Validation Phase
            ann.eval()
            with torch.no_grad():
                valid_outputs = ann(X_test_tensor)
                epoch_val_loss = criterion(valid_outputs, y_test_tensor).item() # Validation loss for current epoch

            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch: {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    

        return train_losses, val_losses

def plot_loss_curves(train_losses, valid_losses):
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_losses, label="Training Loss", color="blue", linewidth=3)
    plt.plot(valid_losses, label="Validation Loss", color="orange", linewidth=3)
    
    plt.title("Model Performance: Overfitting / Underfitting Detector", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=12)
    
    plt.savefig("loss_curve.png")
    plt.show()


if __name__ == "__main__":
    train_hist, valid_hist = train(epochs=900)
    plot_loss_curves(train_hist, valid_hist)

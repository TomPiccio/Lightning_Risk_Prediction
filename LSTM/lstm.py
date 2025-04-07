import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class LightningLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
        super(LightningLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Use the last output for prediction
        out = self.fc(out[:, -1, :])
        return out

def prepare_data(data, sequence_length):
    """
    Prepare data for LSTM model
    Args:
        data: DataFrame with columns [temperature, wind_speed, humidity, rainfall, wind_direction]
        sequence_length: Number of time steps to use for prediction
    Returns:
        X: Input sequences
        y: Target values
    """
    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(data['lightning_risk'].iloc[i + sequence_length])
    
    return torch.FloatTensor(X), torch.FloatTensor(y)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Train the LSTM model
    """
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.extend((outputs.squeeze() > 0.5).cpu().numpy())
            actuals.extend(batch_y.numpy())
    
    accuracy = accuracy_score(actuals, predictions)
    return accuracy

# Example usage:
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and prepare your data here
    # data = pd.read_csv('your_data.csv')
    # X, y = prepare_data(data, sequence_length=24)  # Example: 24 time steps
    
    # Split data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Create data loaders
    # train_dataset = TensorDataset(X_train, y_train)
    # val_dataset = TensorDataset(X_val, y_val)
    # test_dataset = TensorDataset(X_test, y_test)
    
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32)
    # test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model
    # model = LightningLSTM().to(device)
    
    # Define loss function and optimizer
    # criterion = nn.BCELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    # train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device)
    
    # Evaluate model
    # accuracy = evaluate_model(model, test_loader, device)
    # print(f'Test Accuracy: {accuracy:.4f}')


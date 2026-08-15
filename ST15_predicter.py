# ST15_predictor.py
# A simple LSTM that predicts future workload from recent history load.
# Testing on a sine wave first.

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# THE LSTM Use
# LSTM = a neural network good at sequences.
# Input: last N workload values. Output: next M predicted values.

class WorkloadPredictor(nn.Module):
    def __init__(self, input_len=10, output_len=5, hidden=32):
        super().__init__()
        self.output_len = output_len
        # The LSTM reads the sequence
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        # A layer that turns the LSTM's memory into predictions
        self.OPL = nn.Linear(hidden, output_len)

    def forward(self, x):
        # x shape: (batch, sequence_length, 1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]        # take the LSTM's final memory
        return self.OPL(last)         # predict the next M values



# Prepare TRAINING DATA (sine curve according to Supervisor instrctions)

def make_sine_data(num_points=2000):
    t = np.arange(num_points)
    wave = 0.5 + 0.4 * np.sin(t / 20.0)    # a smooth wave between 0 and 1, plus a tiny bit of noise
    noise = np.random.normal(0, 0.02, num_points)
    return np.clip(wave + noise, 0, 1)

# Break down the wave  past 10 - next 5    for traning
def make_windows(data, input_len=10, output_len=5):
    X, y = [], []
    for i in range(len(data) - input_len - output_len):
        X.append(data[i : i + input_len])
        y.append(data[i + input_len : i + input_len + output_len])
    return np.array(X), np.array(y)


# Training workload predicter

def train():
    print("Making sine-wave workload data...")
    data = make_sine_data(2000)

    # 70% train, 30% test (standard split)
    split = int(len(data) * 0.7)
    train_data, test_data = data[:split], data[split:]

    X_train, y_train = make_windows(train_data)
    X_test, y_test = make_windows(test_data)

    # turn into tensors (the shape torch wants: batch, length, 1)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    model = WorkloadPredictor(input_len=10, output_len=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    print("Training the predictor for 100 rounds...\n")
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Round {epoch+1:3d} | training error: {loss.item():.5f}")

    # ---- Test it ----
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t)
        test_error = loss_fn(test_pred, y_test_t).item()
        rmse = np.sqrt(test_error)

    print(f"\nTest RMSE : {rmse}")     # Lower RMSE like  0.05 better , 0.1 good , 0.2 bad 
    # ---- Save the trained predictor ----
    torch.save(model.state_dict(), "predictor.pt")
    print("Saved predictor as predictor.pt")

    # ---- Draw a picture: predicted vs actual ----
    with torch.no_grad():
        # predict across the test set, take the first predicted step each time
        preds = model(X_test_t)[:, 0].numpy()
        actual = y_test_t[:, 0].numpy()

    plt.figure(figsize=(11, 4))
    plt.plot(actual[:150], label="Actual workload", color="blue", linewidth=2)
    plt.plot(preds[:150], label="Predicted workload", color="orange",
             linewidth=2, linestyle="--")
    plt.title("Workload Predictor — Predicted vs Actual (sine test)")
    plt.xlabel("Time step")
    plt.ylabel("Workload (0 to 1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()



    if rmse < 0.05:
        print("RMSE is low, predictions track the wave.")
      
    else:
        print(" RMSE is a bit high - need to tune it")

if __name__ == "__main__":
    train() 
# ST16_predictor_patterns.py
# Train the predictor on multiple workload patterns and TEST generalisation
# on an unseen pattern (the step function).

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ======================================================================
# THE PREDICTOR (same LSTM as before)
# ======================================================================
class WorkloadPredictor(nn.Module):
    def __init__(self, input_len=10, output_len=5, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, output_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ======================================================================
# THE THREE WORKLOAD PATTERNS (your three sketches)
# ======================================================================

def make_sine(n=2000):
    """Pattern 1: smooth wave (daily load)."""
    t = np.arange(n)
    wave = 0.5 + 0.4 * np.sin(t / 20.0)
    return np.clip(wave + np.random.normal(0, 0.02, n), 0, 1)


def make_spikes(n=2000):
    """Pattern 2: calm baseline with sudden spikes that decay (traffic shocks)."""
    signal = np.full(n, 0.3)                       # calm baseline
    # add random spikes that fade away
    num_spikes = n // 200
    for _ in range(num_spikes):
        start = np.random.randint(0, n - 50)
        height = np.random.uniform(0.3, 0.6)
        # spike jumps up then decays over ~30 steps
        for i in range(30):
            if start + i < n:
                signal[start + i] += height * np.exp(-i / 10.0)
    return np.clip(signal + np.random.normal(0, 0.02, n), 0, 1)


def make_steps(n=2000):
    """Pattern 3: flat, then sudden jump to new flat level (step function)."""
    signal = np.zeros(n)
    level = 0.5
    i = 0
    while i < n:
        hold = np.random.randint(80, 200)          # how long to stay flat
        level = np.random.uniform(0.2, 0.8)        # new random level
        signal[i : i + hold] = level
        i += hold
    return np.clip(signal + np.random.normal(0, 0.02, n), 0, 1)


# ======================================================================
# HELPERS
# ======================================================================

def make_windows(data, input_len=10, output_len=5):
    X, y = [], []
    for i in range(len(data) - input_len - output_len):
        X.append(data[i : i + input_len])
        y.append(data[i + input_len : i + input_len + output_len])
    return np.array(X), np.array(y)


def to_tensors(X, y):
    return (torch.tensor(X, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(y, dtype=torch.float32))


def evaluate_rmse(model, data, name):
    """Test the model on a pattern, return RMSE and predictions."""
    X, y = make_windows(data)
    Xt, yt = to_tensors(X, y)
    model.eval()
    with torch.no_grad():
        pred = model(Xt)
        rmse = np.sqrt(nn.functional.mse_loss(pred, yt).item())
    print(f"  {name:20s} RMSE: {rmse:.4f}")
    return rmse, pred[:, 0].numpy(), yt[:, 0].numpy()


# ======================================================================
# MAIN: train on sine + spikes, TEST on steps (unseen)
# ======================================================================

def main():
    print("Generating workload patterns...\n")
    sine = make_sine()
    spikes = make_spikes()
    steps = make_steps()

    # ---- TRAINING DATA: sine + spikes combined ----
    # (the predictor learns smooth AND bursty patterns)
    train_sine_X, train_sine_y = make_windows(sine)
    train_spike_X, train_spike_y = make_windows(spikes)
    X_train = np.concatenate([train_sine_X, train_spike_X])
    y_train = np.concatenate([train_sine_y, train_spike_y])
    Xt, yt = to_tensors(X_train, y_train)

    # ---- BUILD + TRAIN ----
    model = WorkloadPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    print("Training on SINE + SPIKES (150 rounds)...\n")
    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 30 == 0:
            print(f"  Round {epoch+1:3d} | training error: {loss.item():.5f}")

    torch.save(model.state_dict(), "predictor.pt")
    print("\nSaved predictor as predictor.pt\n")

    # ---- TEST on all three (steps is UNSEEN) ----
    print("=" * 50)
    print("TESTING (steps = never seen during training)")
    print("=" * 50)
    _, sine_pred, sine_true = evaluate_rmse(model, sine, "Sine (trained)")
    _, spike_pred, spike_true = evaluate_rmse(model, spikes, "Spikes (trained)")
    _, step_pred, step_true = evaluate_rmse(model, steps, "Steps (UNSEEN test)")

    # ---- DRAW all three, side by side ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(sine_true[:150], color="blue", label="Actual")
    axes[0].plot(sine_pred[:150], color="orange", linestyle="--", label="Predicted")
    axes[0].set_title("Sine (trained on)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(spike_true[:150], color="blue", label="Actual")
    axes[1].plot(spike_pred[:150], color="orange", linestyle="--", label="Predicted")
    axes[1].set_title("Spikes (trained on)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(step_true[:150], color="blue", label="Actual")
    axes[2].plot(step_pred[:150], color="orange", linestyle="--", label="Predicted")
    axes[2].set_title("Steps (UNSEEN — the hard test)")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 50)
    print("HOW TO READ THIS")
    print("=" * 50)
    print("""
  - Sine & Spikes: predictor was TRAINED on these -> should fit well.
  - Steps: predictor NEVER saw these -> tests generalisation.
    Watch how it handles the sudden jumps: it will lag at each jump
    (nobody can predict a surprise), then catch up. That LAG is the
    interesting, honest finding for your thesis.
""")


if __name__ == "__main__":
    main()
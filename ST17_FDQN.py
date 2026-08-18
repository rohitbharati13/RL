# ST17_predictive_placer.py
# OPTION 1: a DQN placer that uses workload PREDICTION to make
# energy-aware packing decisions. The predictor now has a REAL job.
#
# Needs: ST10_9B1.py (environment), ST15/16 predictor (predictor.pt)

import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
from ST10_9B1 import TwoAgentEnv, PowerState


# ======================================================================
# LOAD THE TRAINED PREDICTOR (already built, we just use it)
# ======================================================================
class WorkloadPredictor(nn.Module):
    def __init__(self, input_len=10, output_len=5, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, output_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def load_predictor():
    model = WorkloadPredictor()
    model.load_state_dict(torch.load("predictor.pt"))
    model.eval()
    return model


# ======================================================================
# THE DQN BRAIN (same recipe as before)
# ======================================================================
class DQNBrain(nn.Module):
    def __init__(self, input_size, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, num_actions),
        )
    def forward(self, x):
        return self.net(x)


class Memory:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def add(self, *experience):
        self.buffer.append(experience)
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (torch.tensor(np.array(s), dtype=torch.float32),
                torch.tensor(a, dtype=torch.long),
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(np.array(ns), dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32))
    def size(self):
        return len(self.buffer)


# ======================================================================
# THE PREDICTIVE, ENERGY-AWARE PLACER
# ======================================================================

class PredictivePlacer:
    def __init__(self, state_size, num_actions):
        self.num_actions = num_actions
        self.brain = DQNBrain(state_size, num_actions)
        self.memory = Memory()
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=0.0005)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        with torch.no_grad():
            st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return int(torch.argmax(self.brain(st)).item())

    def learn(self, batch_size=64):
        if self.memory.size() < batch_size:
            return
        s, a, r, ns, d = self.memory.sample(batch_size)
        q = self.brain(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nq = self.brain(ns).max(1)[0]
        target = r + self.gamma * nq * (1 - d)
        loss = nn.functional.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ======================================================================
# STATE: includes the PREDICTED future load (the key connection)
# ======================================================================

def get_state(env, predictor, load_history):
    """Placer sees each server + queue + PREDICTED future load."""
    obs = []
    for s in env.servers:
        obs.append(s.utilization())
        obs.append(1.0 if s.is_ready() else 0.0)
    obs.append(min(len(env.waiting_queue) / 20.0, 1.0))

    # THE KEY LINE: ask the predictor what load is coming
    if len(load_history) >= 10:
        recent = np.array(load_history[-10:], dtype=np.float32).reshape(1, 10, 1)
        with torch.no_grad():
            predicted = predictor(torch.tensor(recent))[0, 0].item()
    else:
        predicted = 0.5   # default until we have history
    obs.append(np.clip(predicted, 0, 1))   # the forecast as a state input

    return np.array(obs, dtype=np.float32)


# ======================================================================
# ENERGY-AWARE REWARD: reward PACKING onto fewer servers
# ======================================================================

def energy_aware_reward(env, placed_ok, chose_used_server):
    """
    Reward good placement AND packing onto fewer servers.
    This is what makes prediction matter: packing when low load is coming
    saves energy; the predictor tells the placer when it's safe to pack.
    """
    reward = 0.0
    if placed_ok:
        reward += 3.0
        # BONUS for packing: placing on an already-busy server (consolidation)
        # instead of spreading to a fresh one. Fewer active servers = less energy.
        if chose_used_server:
            reward += 2.0    # good packing
    else:
        reward -= 3.0

    # small penalty for how many servers are active (encourages consolidation)
    active = sum(1 for s in env.servers if s.cpu_used > 0)
    reward -= 0.2 * active
    return reward


# ======================================================================
# TRAIN
# ======================================================================

def train(num_episodes=300):
    env = TwoAgentEnv(num_servers=10, arrival_rate=1.0, episode_length=200)
    predictor = load_predictor()

    # state = 2 per server + queue + 1 prediction
    state_size = env.num_servers * 2 + 1 + 1
    num_actions = env.num_servers + 1
    placer = PredictivePlacer(state_size, num_actions)

    # power manager: simple fixed rule (we're NOT learning power now)
    # keep enough awake for current load
    def fixed_power(env):
        return max(3, min(env.num_servers, 3 + len(env.waiting_queue) // 2))

    history = []
    print(f"Training predictive placer for {num_episodes} episodes...\n")

    for episode in range(num_episodes):
        env.reset(seed=episode)
        load_history = []
        state = get_state(env, predictor, load_history)
        ep_reward = 0
        done = False

        while not done:
            # record current load for the predictor
            cur_load = np.mean([s.utilization() for s in env.servers])
            load_history.append(cur_load)

            action = placer.choose_action(state)
            power_target = fixed_power(env)

            # was the chosen server already in use? (for packing bonus)
            chose_used = (action < env.num_servers and
                          env.servers[action].cpu_used > 0)

            pr, wr, done = env.step(action, power_target)
            placed_ok = pr > 0   # rough: positive placer reward = placed ok

            # our energy-aware reward replaces the raw placement reward
            reward = energy_aware_reward(env, placed_ok, chose_used)

            next_state = get_state(env, predictor, load_history)
            placer.memory.add(state, action, reward, next_state, float(done))
            placer.learn()

            state = next_state
            ep_reward += reward

        history.append(ep_reward)
        if (episode + 1) % 20 == 0:
            print(f"  Ep {episode+1:3d} | avg reward (last 20): "
                  f"{np.mean(history[-20:]):7.1f} | explore: {placer.epsilon:.2f}")

    torch.save(placer.brain.state_dict(), "predictive_placer.pt")
    print("\nSaved as predictive_placer.pt")

    early = np.mean(history[:20])
    late = np.mean(history[-20:])
    print("\n" + "=" * 50)
    print("DID THE PREDICTIVE PLACER LEARN?")
    print("=" * 50)
    print(f"  First 20: {early:7.1f}   Last 20: {late:7.1f}   "
          f"{'✅ UP' if late > early else '⚠️ check'}")
    print("=" * 50)
    print("\nThe placer now uses the forecast to make packing decisions.")
    print("That's the REAL connection between predictor and DQN.")


if __name__ == "__main__":
    train()
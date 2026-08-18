# ST18_ablation.py
# THE PROOF: does the predictor actually help placement?
# Trains 3 identical placers differing ONLY in future-info:
#   A) full predictor forecast
#   B) current load only (control)
#   C) no info (constant dummy)
# Then compares them on REAL metrics: energy, SLA, packing.

import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
from ST10_9B1 import TwoAgentEnv, PowerState


# ---- predictor (loaded, used only by Version A) ----
class WorkloadPredictor(nn.Module):
    def __init__(self, input_len=10, output_len=5, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, output_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def load_predictor():
    m = WorkloadPredictor(); m.load_state_dict(torch.load("predictor.pt")); m.eval()
    return m


# ---- DQN brain + memory (same recipe) ----
class DQNBrain(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,64), nn.ReLU(),
                                 nn.Linear(64,64), nn.ReLU(), nn.Linear(64,n_out))
    def forward(self, x): return self.net(x)

class Memory:
    def __init__(self, cap=10000): self.b = deque(maxlen=cap)
    def add(self, *e): self.b.append(e)
    def sample(self, n):
        s,a,r,ns,d = zip(*random.sample(self.b, n))
        return (torch.tensor(np.array(s),dtype=torch.float32),
                torch.tensor(a,dtype=torch.long),
                torch.tensor(r,dtype=torch.float32),
                torch.tensor(np.array(ns),dtype=torch.float32),
                torch.tensor(d,dtype=torch.float32))
    def size(self): return len(self.b)


class Placer:
    def __init__(self, state_size, n_actions):
        self.n = n_actions
        self.brain = DQNBrain(state_size, n_actions)
        self.mem = Memory()
        self.opt = torch.optim.Adam(self.brain.parameters(), lr=0.0005)
        self.gamma=0.99; self.eps=1.0; self.eps_min=0.05; self.eps_decay=0.9995
    def act(self, s):
        if random.random() < self.eps:
            return random.randint(0, self.n-1)
        with torch.no_grad():
            return int(torch.argmax(self.brain(torch.tensor(s,dtype=torch.float32).unsqueeze(0))).item())
    def learn(self, bs=64):
        if self.mem.size() < bs: return
        s,a,r,ns,d = self.mem.sample(bs)
        q = self.brain(s).gather(1,a.unsqueeze(1)).squeeze(1)
        with torch.no_grad(): nq = self.brain(ns).max(1)[0]
        loss = nn.functional.mse_loss(q, r + self.gamma*nq*(1-d))
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.eps = max(self.eps_min, self.eps*self.eps_decay)


# ======================================================================
# THE KEY DIFFERENCE: the future-info number, per version
# ======================================================================

def future_info(version, env, predictor, load_history):
    """The ONLY thing that differs between the 3 versions."""
    if version == "A_predictor":
        if len(load_history) >= 10:
            recent = np.array(load_history[-10:], dtype=np.float32).reshape(1,10,1)
            with torch.no_grad():
                return float(np.clip(predictor(torch.tensor(recent))[0,0].item(), 0, 1))
        return 0.5
    elif version == "B_current":
        # current load instead of a forecast (the honest control)
        return float(np.mean([s.utilization() for s in env.servers]))
    else:  # C_none
        return 0.5   # constant dummy — no information


def get_state(version, env, predictor, load_history):
    obs = []
    for s in env.servers:
        obs.append(s.utilization())
        obs.append(1.0 if s.is_ready() else 0.0)
    obs.append(min(len(env.waiting_queue)/20.0, 1.0))
    obs.append(future_info(version, env, predictor, load_history))  # the one differing number
    return np.array(obs, dtype=np.float32)


def energy_aware_reward(env, placed_ok, chose_used):
    r = 0.0
    if placed_ok:
        r += 3.0
        if chose_used: r += 2.0
    else:
        r -= 3.0
    active = sum(1 for s in env.servers if s.cpu_used > 0)
    r -= 0.2 * active
    return r


def fixed_power(env):
    return max(3, min(env.num_servers, 3 + len(env.waiting_queue)//2))


# ======================================================================
# TRAIN one version
# ======================================================================
def train_version(version, predictor, episodes=250):
    env = TwoAgentEnv(num_servers=10, arrival_rate=1.0, episode_length=200)
    state_size = env.num_servers*2 + 1 + 1
    placer = Placer(state_size, env.num_servers + 1)

    for ep in range(episodes):
        env.reset(seed=ep)
        load_hist = []
        state = get_state(version, env, predictor, load_hist)
        done = False
        while not done:
            load_hist.append(np.mean([s.utilization() for s in env.servers]))
            action = placer.act(state)
            chose_used = (action < env.num_servers and env.servers[action].cpu_used > 0)
            pr, wr, done = env.step(action, fixed_power(env))
            reward = energy_aware_reward(env, pr > 0, chose_used)
            nxt = get_state(version, env, predictor, load_hist)
            placer.mem.add(state, action, reward, nxt, float(done))
            placer.learn()
            state = nxt
    return placer


# ======================================================================
# EVALUATE one version on REAL metrics (energy, SLA, packing)
# ======================================================================
def evaluate_version(version, placer, predictor, seeds):
    results = []
    for seed in seeds:
        env = TwoAgentEnv(num_servers=10, arrival_rate=1.0, episode_length=200)
        env.reset(seed=seed)
        load_hist = []
        active_track = []
        placer.eps = 0.0   # no exploration during eval
        done = False
        while not done:
            load_hist.append(np.mean([s.utilization() for s in env.servers]))
            state = get_state(version, env, predictor, load_hist)
            action = placer.act(state)
            _, _, done = env.step(action, fixed_power(env))
            active_track.append(sum(1 for s in env.servers if s.cpu_used > 0))
        results.append({
            "energy": env.total_energy,
            "sla": env.sla_violations,
            "completed": env.total_completed,
            "avg_active": np.mean(active_track),
        })
    return {k: np.mean([r[k] for r in results]) for k in results[0]}


# ======================================================================
# MAIN
# ======================================================================
def main():
    predictor = load_predictor()
    seeds = [42, 123, 456, 789, 1011]

    versions = {
        "A_predictor": "Full predictor (forecast)",
        "B_current":   "Current load only (control)",
        "C_none":      "No info (dummy)",
    }

    trained = {}
    for v, label in versions.items():
        print(f"Training {label}...")
        trained[v] = train_version(v, predictor)
    print()

    print("=" * 74)
    print(f"{'Version':<30} {'Energy':>9} {'SLA':>8} {'Completed':>11} {'AvgActive':>10}")
    print("-" * 74)
    metrics = {}
    for v, label in versions.items():
        m = evaluate_version(v, trained[v], predictor, seeds)
        metrics[v] = m
        print(f"{label:<30} {m['energy']:>9.2f} {m['sla']:>8.1f} "
              f"{m['completed']:>11.1f} {m['avg_active']:>10.2f}")
    print("=" * 74)

    # ---- Honest interpretation ----
    A, B, C = metrics["A_predictor"], metrics["B_current"], metrics["C_none"]
    print("\n" + "=" * 60)
    print("WHAT THE ABLATION SHOWS (honest read)")
    print("=" * 60)
    print(f"""
  Predictor energy: {A['energy']:.2f} Wh, packing: {A['avg_active']:.2f} active servers
  Current-only    : {B['energy']:.2f} Wh, packing: {B['avg_active']:.2f}
  No info         : {C['energy']:.2f} Wh, packing: {C['avg_active']:.2f}

  Does prediction beat current-load-only?
    -> If A uses less energy / packs tighter than B: prediction ADDS value.
    -> If A and B are similar: forecasting doesn't beat knowing current load
       (honest, important finding).
    -> If all similar: extra info doesn't help placement here.

  Lower energy = better. Lower AvgActive = tighter packing = more energy saving.
  Whatever the result, this is a REAL, defensible thesis finding.
""")
    
# ============================================================
    # VISUALIZATION — the three-way ablation
    # ============================================================
    import matplotlib.pyplot as plt

    labels = ["Full\npredictor", "Current\nload only", "No info\n(dummy)"]
    energy_vals = [A['energy'], B['energy'], C['energy']]
    sla_vals    = [A['sla'], B['sla'], C['sla']]
    active_vals = [A['avg_active'], B['avg_active'], C['avg_active']]
    colors = ["#D85A30", "#378ADD", "#888888"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].bar(labels, energy_vals, color=colors, edgecolor="black")
    axes[0].set_title("Energy (Wh) — lower is better")
    axes[0].set_ylabel("Energy (Wh)")

    axes[1].bar(labels, sla_vals, color=colors, edgecolor="black")
    axes[1].set_title("SLA Violations — lower is better")
    axes[1].set_ylabel("Violations")

    axes[2].bar(labels, active_vals, color=colors, edgecolor="black")
    axes[2].set_title("Avg Active Servers (packing)")
    axes[2].set_ylabel("Servers")

    for ax in axes:
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("ST18 Ablation: prediction did NOT help placement (flat reward)",
                 fontsize=13)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
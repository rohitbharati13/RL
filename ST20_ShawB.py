# ST20_shaw_baselines.py
# Q-learning and SARSA (Shaw's algorithms) implemented in YOUR environment.
# These become baselines: your DQN should beat them.
# Tabular methods need a SMALL discretised state.

import numpy as np
import random
from ST10_9B1 import TwoAgentEnv, PowerState


# ======================================================================
# DISCRETISE the state into a small number of buckets (for the table)
# ======================================================================
# Shaw did this too — continuous values chopped into LOW/MED/HIGH buckets.

def discretise_state(env):
    """
    Turn the situation into a small tuple the table can look up.
    We use just 3 features, each in a few buckets, to keep the table small.
    """
    # feature 1: how full is the queue? (0=empty, 1=some, 2=full)
    q = len(env.waiting_queue)
    queue_bucket = 0 if q == 0 else (1 if q < 8 else 2)

    # feature 2: average server load (0=low, 1=med, 2=high)
    load = np.mean([s.utilization() for s in env.servers])
    load_bucket = 0 if load < 0.33 else (1 if load < 0.66 else 2)

    # feature 3: how many servers have room? (0=none, 1=few, 2=many)
    free = sum(1 for s in env.servers if s.is_ready() and s.cpu_used < s.cpu_capacity)
    free_bucket = 0 if free == 0 else (1 if free < 5 else 2)

    return (queue_bucket, load_bucket, free_bucket)


# ======================================================================
# A TABULAR AGENT (works as both Q-learning AND SARSA)
# ======================================================================

class TabularAgent:
    def __init__(self, num_actions, algorithm="qlearning"):
        self.num_actions = num_actions
        self.algorithm = algorithm       # "qlearning" or "sarsa"
        self.Q = {}                      # the table: Q[(state, action)] = value
        self.alpha = 0.1                 # learning rate
        self.gamma = 0.99                # discount
        self.epsilon = 1.0               # exploration
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995

    def get_q(self, state, action):
        return self.Q.get((state, action), 0.0)   # unseen = 0

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        # pick action with highest Q-value for this state
        qs = [self.get_q(state, a) for a in range(self.num_actions)]
        return int(np.argmax(qs))

    def update(self, state, action, reward, next_state, next_action=None):
        """
        THE ONLY DIFFERENCE between Q-learning and SARSA is here.
        """
        old_q = self.get_q(state, action)

        if self.algorithm == "qlearning":
            # Q-LEARNING: assume we play the BEST next action (optimistic)
            next_best = max(self.get_q(next_state, a) for a in range(self.num_actions))
            target = reward + self.gamma * next_best
        else:
            # SARSA: use the action we ACTUALLY take next (realistic)
            target = reward + self.gamma * self.get_q(next_state, next_action)

        # nudge the table value toward the target
        self.Q[(state, action)] = old_q + self.alpha * (target - old_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ======================================================================
# TRAIN a tabular agent
# ======================================================================

def train_tabular(algorithm, episodes=500):
    env = TwoAgentEnv(num_servers=10, arrival_rate=1.0, episode_length=200)
    num_actions = env.num_servers + 1     # place on server, or reject
    agent = TabularAgent(num_actions, algorithm)

    # power manager: same simple fixed rule as your DQN tests (fair comparison)
    def fixed_power(env):
        return max(3, min(env.num_servers, 3 + len(env.waiting_queue) // 2))

    print(f"Training {algorithm} for {episodes} episodes...")

    for ep in range(episodes):
        env.reset(seed=ep)
        state = discretise_state(env)
        action = agent.choose_action(state)
        done = False

        while not done:
            pr, wr, done = env.step(action, fixed_power(env))
            reward = pr   # use the placer reward
            next_state = discretise_state(env)
            next_action = agent.choose_action(next_state)

            if algorithm == "qlearning":
                agent.update(state, action, reward, next_state)
            else:  # sarsa needs the next action
                agent.update(state, action, reward, next_state, next_action)

            state, action = next_state, next_action
        agent.decay_epsilon()

    return agent


# ======================================================================
# EVALUATE a trained tabular agent
# ======================================================================

def evaluate_tabular(agent, seeds):
    def fixed_power(env):
        return max(3, min(env.num_servers, 3 + len(env.waiting_queue) // 2))

    results = []
    for seed in seeds:
        env = TwoAgentEnv(num_servers=10, arrival_rate=1.0, episode_length=200)
        env.reset(seed=seed)
        agent.epsilon = 0.0    # no exploration during eval
        done = False
        while not done:
            state = discretise_state(env)
            action = agent.choose_action(state)
            _, _, done = env.step(action, fixed_power(env))
        results.append({
            "energy": env.total_energy,
            "sla": env.sla_violations,
            "completed": env.total_completed,
        })
    return {k: np.mean([r[k] for r in results]) for k in results[0]}


# ======================================================================
# MAIN
# ======================================================================

def main():
    seeds = [42, 123, 456, 789, 1011]

    print("Building Shaw's baselines in your environment...\n")

    q_agent = train_tabular("qlearning", episodes=500)
    s_agent = train_tabular("sarsa", episodes=500)
    print()

    q_metrics = evaluate_tabular(q_agent, seeds)
    s_metrics = evaluate_tabular(s_agent, seeds)

    print("=" * 60)
    print(f"{'Algorithm':<20} {'Energy':>9} {'SLA':>8} {'Completed':>11}")
    print("-" * 60)
    print(f"{'Q-learning (Shaw)':<20} {q_metrics['energy']:>9.2f} "
          f"{q_metrics['sla']:>8.1f} {q_metrics['completed']:>11.1f}")
    print(f"{'SARSA (Shaw)':<20} {s_metrics['energy']:>9.2f} "
          f"{s_metrics['sla']:>8.1f} {s_metrics['completed']:>11.1f}")
    print("=" * 60)
    print(f"""
  Table size: Q-learning learned {len(q_agent.Q)} state-action values,
              SARSA learned {len(s_agent.Q)}.
  (Tiny compared to a neural network — that's the point.)

  These are your Shaw-method baselines. In your thesis, compare them
  against your DQN: the DQN should handle the richer state better,
  since tabular methods are limited by discretisation.
""")


if __name__ == "__main__":
    main()
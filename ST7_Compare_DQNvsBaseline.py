# Compare DQN with baseline by running same job on every scheduler  and compare scores.

import numpy as np
from stable_baselines3 import DQN
from ST5_Gymnasium import CloudSchedulingEnv

# ---- Three "brains" we will compare ----

# Dumbest baseline: pick any action at random
def random_policy(env, obs):
    return env.action_space.sample()

# Heuristic baseline: deal jobs in a circle 0,1,2,3,4,0,1,....

def round_robin_policy(env, obs, state):
    action = state["next_server"]
    state["next_server"] = (state["next_server"] + 1) % env.num_servers
    return action

def make_dqn_policy(model):
    def dqn_policy(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return dqn_policy

# ---- Run ONE scheduler for ONE episode on a given seed ----

def run_one_episode(policy_fn, seed, is_round_robin=False):
    env = CloudSchedulingEnv(num_servers=10, arrival_rate=1.5, episode_length=200)
    obs, _ = env.reset(seed=seed)
    rr_state = {"next_server": 0}    # only used by round robin
    total_reward = 0
    done = False

    while not done:
        if is_round_robin:
            action = policy_fn(env, obs, rr_state)
        else:
            action = policy_fn(env, obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    # Return this episode's scorecard
    return {
        "reward": total_reward,
        "energy": env.total_energy,
        "sla": env.sla_violations,
        "completed": env.total_completed,
    }

# ---- Run each scheduler across 5 SEEDS and average (this is real science) ----

def evaluate(name, policy_fn, seeds, is_round_robin=False):
    results = [run_one_episode(policy_fn, s, is_round_robin) for s in seeds]
    # average each metric across the 5 runs
    avg = {k: np.mean([r[k] for r in results]) for k in results[0]}
    std = {k: np.std([r[k] for r in results]) for k in results[0]}
    return name, avg, std


# ---- Main ----

SEEDS = [42, 123, 456, 789, 1011]   # 5 different job streams

print("Loading the trained DQN brain...")
model = DQN.load("dqn_scheduler")
dqn_policy = make_dqn_policy(model)

print(f"Running each scheduler on {len(SEEDS)} seeds (this is the fair fight)...\n")

table = []
table.append(evaluate("Random",      random_policy,      SEEDS))
table.append(evaluate("Round Robin", round_robin_policy, SEEDS, is_round_robin=True))
table.append(evaluate("DQN (ours)",  dqn_policy,         SEEDS))

# ---- Print a clean comparison table ----

print("=" * 70)
print(f"{'Method':<14} {'Reward':>12} {'Energy(Wh)':>12} {'SLA':>8} {'Completed':>11}")
print("-" * 70)
for name, avg, std in table:
    print(f"{name:<14} "
          f"{avg['reward']:>8.1f}±{std['reward']:<3.0f} "
          f"{avg['energy']:>9.2f} "
          f"{avg['sla']:>8.1f} "
          f"{avg['completed']:>11.1f}")
print("=" * 70)

print("\nHow to read this:")
print("  - Higher reward = better overall")
print("  - Lower energy  = better (greener)")
print("  - Lower SLA     = better (fewer missed deadlines)")
print("  - DQN should beat Random clearly. Beating Round Robin = tuning goal.")
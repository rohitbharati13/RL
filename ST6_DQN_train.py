import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from ST5_Gymnasium import CloudSchedulingEnv
import matplotlib.pyplot as plt

class RewardRecorder(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self):
        # Monitor stores finished-episode info under "episode"
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True

env = CloudSchedulingEnv(num_servers=10, arrival_rate=0.6, episode_length=200)
env = Monitor(env)

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.0005,
    buffer_size=100000,
    learning_starts=5000,        # collect experience before learning starts
    exploration_fraction=0.5,    # explore for the FIRST HALF of training
    exploration_final_eps=0.05,
    verbose=1,
)


recorder = RewardRecorder()

model.learn(total_timesteps=300000)
model.save("dqn_scheduler")
print("\nDone. Saved as dqn_scheduler.zip.")
rewards = recorder.episode_rewards
print(f"\nRecorded {len(rewards)} episodes. Drawing the learning curve...")

plt.figure(figsize=(10, 5))
plt.plot(rewards, color="lightblue", alpha=0.5, label="Reward per episode")
if len(rewards) >= 20:    # the clear trend (rolling average)
    window = 20
    smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.plot(range(window - 1, len(rewards)), smooth,
             color="blue", linewidth=2, label=f"Smoothed (avg of {window})")

plt.title("DQN Training — Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# Wrap Daacenter in the standard Gimnasium Format , so RL brain can plug into it.

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Job:
    def __init__(self, job_id, cpu_need , deadline, duration):
        self.job_id = job_id
        self.cpu_need = cpu_need
        self.deadline = deadline
        self.duration = duration
        self.finish_time = None

class Server:
    def __init__(self, server_id, cpu_capacity):
        self.server_id = server_id
        self.cpu_capacity = cpu_capacity
        self.cpu_used = 0
        self.running_jobs = []

    def FreeCPU(self, job):
        return (self.cpu_capacity - self.cpu_used) >= job.cpu_need

    def Run_Job(self, job, current_time):
        self.cpu_used += job.cpu_need
        job.finish_time = current_time + job.duration
        self.running_jobs.append(job)

    def complete_finished_jobs(self, current_time):
        finished = []
        for job in self.running_jobs[:]:
            if job.finish_time <= current_time:
                self.cpu_used -= job.cpu_need
                self.running_jobs.remove(job)
                finished.append(job)
        return finished

    def utilization(self):
        return self.cpu_used / self.cpu_capacity

    def power_use(self) :
    # how much electricity this Server burns Right Now.
        Idle_power = 100    # watts when Server On but empty
        Peak_power = 250    # watts when 100% busy

        return Idle_power + (Peak_power - Idle_power) * self.utilization()      # Linear power model from "Shaw 2016" , HP proliant servers.
    

# ---- The Gymnasium environment

class CloudSchedulingEnv(gym.Env) :
     
     def __init__(self, num_servers=5, arrival_rate=1.5, episode_length=200):
        super().__init__()
        self.num_servers = num_servers
        self.arrival_rate = arrival_rate
        self.episode_length = episode_length

        # WHAT THE AGENT SEES (observation_space):
        # for each Server: [utilisation], plus one number: [queue fullness]
        # So total numbers = num_servers + 1. All between 0.0 and 1.0.

        obs_size = num_servers + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # WHAT THE AGENT CAN DO (action_space):
        # pick a Server (0 to num_servers-1) to place the front job,
        # OR reject the job (action = num_servers).
        self.action_space = spaces.Discrete(num_servers + 1)

    #Start a fresh episode. Called before every training run."""
     def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.current_time = 0
        self.servers = [Server(i, 100) for i in range(self.num_servers)]
        self.waiting_queue = []
        self.next_job_id = 0

        # Scorecard for this episode
        self.total_energy = 0.0
        self.total_completed = 0
        self.sla_violations = 0

        self.generate_jobs()
        return self.get_observation(), {}
     
     def generate_jobs(self):
        num_arrivals = self.rng.poisson(self.arrival_rate)
        for _ in range(num_arrivals):
            job = Job(
                job_id=self.next_job_id,
               cpu_need =self.rng.integers(10, 40),
                deadline=self.current_time + self.rng.integers(10, 50),
                duration=self.rng.integers(3, 8),
            )
            self.next_job_id += 1
            self.waiting_queue.append(job)

    # Build the list of numbers the agent sees right now.
     def get_observation(self):
        
        obs = [s.utilization() for s in self.servers]      # each Server's fullness
        queue_fullness = min(len(self.waiting_queue) / 20.0, 1.0)  # capped at 1.0
        obs.append(queue_fullness)
        return np.array(obs, dtype=np.float32)
     
# The agent picks ONE action. We apply it and move time forward by one.

     def step(self, action):
        reward = 0.0

        # --- Apply the agent's decision on the FRONT job in the queue ---
        if len(self.waiting_queue) > 0:
            job = self.waiting_queue[0]

            if action < self.num_servers:
                # Agent chose a Server. Try to place the job there.
                Server = self.servers[action]
                if Server.FreeCPU(job):
                    Server.Run_Job(job, self.current_time)
                    self.waiting_queue.pop(0)        # job placed, leave queue
                    reward += 1.0                    # small reward for placing a job
                else:
                    reward -= 2.0                    # chose a full Server — bad move
            else:
                # Agent chose to reject the job.
                self.waiting_queue.pop(0)
                self.sla_violations += 1
                reward -= 10.0                       # rejecting is a big penalty

        # --- Move the clock forward by one timestep ---
        self.current_time += 1

        # Finish jobs (and check for late completions)
        for Server in self.servers:
            done = Server.complete_finished_jobs(self.current_time)
            for j in done:
                self.total_completed += 1
                if j.finish_time > j.deadline:
                    self.sla_violations += 1
                    reward -= 5.0

        # Drop expired waiting jobs
        still_waiting = []
        for j in self.waiting_queue:
            if self.current_time > j.deadline:
                self.sla_violations += 1
                reward -= 5.0
            else:
                still_waiting.append(j)
        self.waiting_queue = still_waiting

        # New arrivals
        self.generate_jobs()

        # Energy this step → penalty (we want LESS energy)
        step_power = sum(s.power_use() for s in self.servers)
        energy_wh = step_power / 3600
        self.total_energy += energy_wh
        reward -= 0.01 * step_power      # the more watts, the more penalty

        # Is the episode over?
        truncated = self.current_time >= self.episode_length
        terminated = False

        return self.get_observation(), reward, terminated, truncated, {}
     

# ---- Quick test: does our socket work with RANDOM actions? ----

if __name__ == "__main__":
    env = CloudSchedulingEnv(num_servers=5)
    obs, _ = env.reset(seed=42)

    print(f"What the agent sees (observation): {obs}")
    print(f"Number of things it sees: {len(obs)}")
    print(f"Number of actions it can pick: {env.action_space.n}")
    print("\nRunning 200 steps with RANDOM actions (no brain yet)...\n")

    total_reward = 0
    obs, _ = env.reset(seed=42)
    done = False
    while not done:
        action = env.action_space.sample()      # random action
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Total reward (random agent): {total_reward:.1f}")
    print(f"Energy used: {env.total_energy:.2f} Wh")
    print(f"SLA violations: {env.sla_violations}")
    print(f"Jobs completed: {env.total_completed}")
   

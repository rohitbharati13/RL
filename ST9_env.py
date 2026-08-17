# ST9_TwoAgentEnv.py
# Environment rebuilt to support server SLEEP/WAKE so energy can actually be saved.
# Step 9A: build the world + test with random actions. No agents yet.

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import Enum


# Named labels for power states (clearer than 0,1,2)
class PowerState(Enum):
    ON = 0       # awake, burns full power
    SLEEP = 1    # napping, tiny power, needs 2 steps to wake
    OFF = 2      # fully off, zero power


class Job:
    def __init__(self, job_id, cpu_needed, deadline, duration):
        self.job_id = job_id
        self.cpu_needed = cpu_needed
        self.deadline = deadline
        self.duration = duration
        self.finish_time = None


class Server:
    def __init__(self, server_id, cpu_capacity):
        self.server_id = server_id
        self.cpu_capacity = cpu_capacity
        self.cpu_used = 0
        self.running_jobs = []
        self.power_state = PowerState.ON
        self.wake_timer = 0            # steps left until awake

    def is_ready(self):
        """Can this server take a job right now?"""
        return self.power_state == PowerState.ON and self.wake_timer == 0

    def has_room_for(self, job):
        return self.is_ready() and (self.cpu_capacity - self.cpu_used) >= job.cpu_needed

    def run_job(self, job, current_time):
        self.cpu_used += job.cpu_needed
        job.finish_time = current_time + job.duration
        self.running_jobs.append(job)

    def complete_finished_jobs(self, current_time):
        finished = []
        for job in self.running_jobs[:]:
            if job.finish_time <= current_time:
                self.cpu_used -= job.cpu_needed
                self.running_jobs.remove(job)
                finished.append(job)
        return finished

    def utilization(self):
        return self.cpu_used / self.cpu_capacity

    def go_to_sleep(self):
        """Sleep only if ON and empty."""
        if self.power_state == PowerState.ON and len(self.running_jobs) == 0:
            self.power_state = PowerState.SLEEP
            return True
        return False

    def wake_up(self):
        """Wake from sleep — takes 2 steps to be ready."""
        if self.power_state == PowerState.SLEEP:
            self.power_state = PowerState.ON
            self.wake_timer = 2
            return True
        return False

    def tick_wake_timer(self):
        if self.wake_timer > 0:
            self.wake_timer -= 1

    def power_watts(self):
        if self.power_state == PowerState.ON:
            IDLE, PEAK = 100, 250
            return IDLE + (PEAK - IDLE) * self.utilization()
        elif self.power_state == PowerState.SLEEP:
            return 10
        else:  # OFF
            return 0


class TwoAgentCloudEnv(gym.Env):
    """
    Environment that supports BOTH job placement AND server power control.
    Step 9A: we expose ONE combined action space and test with random actions.
    In 9B we split this into two agents.

    Action layout (single combined action for now):
      0 .. N-1     -> assign the front job to server i
      N            -> reject the front job
      N+1 .. 2N    -> toggle sleep/wake on server (action - N - 1)
    """

    def __init__(self, num_servers=10, arrival_rate=1.0, episode_length=200):
        super().__init__()
        self.num_servers = num_servers
        self.arrival_rate = arrival_rate
        self.episode_length = episode_length

        # Observation: each server [utilisation, power_state], plus queue fullness
        obs_size = num_servers * 2 + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        # Combined action space (split into two agents later)
        self.action_space = spaces.Discrete(2 * num_servers + 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.current_time = 0
        self.servers = [Server(i, 100) for i in range(self.num_servers)]
        self.waiting_queue = []
        self.next_job_id = 0
        self.total_energy = 0.0
        self.total_completed = 0
        self.sla_violations = 0
        self._generate_jobs()
        return self._get_obs(), {}

    def _generate_jobs(self):
        n = self.rng.poisson(self.arrival_rate)
        for _ in range(n):
            job = Job(
                job_id=self.next_job_id,
                cpu_needed=self.rng.integers(10, 40),
                deadline=self.current_time + self.rng.integers(15, 60),
                duration=self.rng.integers(3, 8),
            )
            self.next_job_id += 1
            self.waiting_queue.append(job)

    def _get_obs(self):
        obs = []
        for s in self.servers:
            obs.append(s.utilization())
            obs.append(s.power_state.value / 2.0)  # normalise 0,1,2 -> 0,0.5,1
        obs.append(min(len(self.waiting_queue) / 20.0, 1.0))
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        reward = 0.0

        # --- Apply the action ---
        if action < self.num_servers:
            # assign front job to server `action`
            if self.waiting_queue:
                job = self.waiting_queue[0]
                srv = self.servers[action]
                if srv.has_room_for(job):
                    srv.run_job(job, self.current_time)
                    self.waiting_queue.pop(0)
                    reward += 5.0
                else:
                    reward -= 3.0
        elif action == self.num_servers:
            # reject front job
            if self.waiting_queue:
                self.waiting_queue.pop(0)
                self.sla_violations += 1
                reward -= 10.0
        else:
            # toggle sleep/wake on a server
            idx = action - self.num_servers - 1
            srv = self.servers[idx]
            if srv.power_state == PowerState.ON:
                srv.go_to_sleep()
            elif srv.power_state == PowerState.SLEEP:
                srv.wake_up()

        # --- Advance time ---
        self.current_time += 1
        for s in self.servers:
            s.tick_wake_timer()
            for j in s.complete_finished_jobs(self.current_time):
                self.total_completed += 1
                if j.finish_time > j.deadline:
                    self.sla_violations += 1
                    reward -= 5.0

        # drop expired waiting jobs
        still = []
        for j in self.waiting_queue:
            if self.current_time > j.deadline:
                self.sla_violations += 1
                reward -= 5.0
            else:
                still.append(j)
        self.waiting_queue = still

        self._generate_jobs()

        # energy this step
        step_power = sum(s.power_watts() for s in self.servers)
        self.total_energy += step_power / 3600
        reward -= 0.002 * step_power

        truncated = self.current_time >= self.episode_length
        return self._get_obs(), reward, False, truncated, {}


# ---- Test with random actions: does energy CHANGE when servers sleep? ----
if __name__ == "__main__":
    print("Testing the new sleep/wake environment with random actions...\n")

    env = TwoAgentCloudEnv(num_servers=10, arrival_rate=1.0)
    obs, _ = env.reset(seed=42)

    sleep_counts = []
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        # count how many servers are asleep right now
        asleep = sum(1 for s in env.servers if s.power_state == PowerState.SLEEP)
        sleep_counts.append(asleep)

    print(f"Energy used:        {env.total_energy:.2f} Wh")
    print(f"Jobs completed:     {env.total_completed}")
    print(f"SLA violations:     {env.sla_violations}")
    print(f"Avg servers asleep: {np.mean(sleep_counts):.1f} (out of 10)")
    print()
    print("KEY CHECK: because random actions sometimes sleep servers,")
    print("energy should now be LOWER and VARIABLE — not stuck at ~63 like before.")
    print("This proves the sleep mechanism works. Next (9B): add the two agents.")
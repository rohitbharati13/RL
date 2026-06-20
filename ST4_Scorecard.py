# Scorecard will represent the Enery , missed deadlines (SLA violation ) and utilisation

import numpy as np

class Job :
    def __init__(self , job_id , cpu_need , deadline , duration) :
        self.job_id = job_id
        self.cpu_need = cpu_need
        self.deadline = deadline
        self.duration = duration
        self.finish_time = None


class Server :
    def __init__(self , server_id , cpu_capacity):
        self.server_id = server_id
        self.cpu_capacity = cpu_capacity
        self.cpu_used = 0
        self.running_jobs = []

    def FreeCPU(self , job) :
        return (self.cpu_capacity - self.cpu_used ) >= job.cpu_need

    def Run_Job(self , job , current_time) :
        self.cpu_used  += job.cpu_need
        job.finish_time = current_time + job.duration
        self.running_jobs.append(job)

    def complete_finished_jobs(self , current_time) :
        finished = []
        for job in self.running_jobs[ : ] :
            if job.finish_time <= current_time :
               self.cpu_used -= job.cpu_need
               self.running_jobs.remove(job)
               finished.append(job)
        return finished
    
    def utilization(self) :
        # how much server is busy now ? A number from 0.0 to 1.0
        return self.cpu_used / self.cpu_capacity
    
    def power_use(self) :
        # how much electricity this server burns Right Now.
        Idle_power = 100    # watts when server On but empty
        Peak_power = 250    # watts when 100% busy

        return Idle_power + (Peak_power - Idle_power) * self.utilization()      # Linear power model from "Shaw 2016" , HP proliant servers.


class Datacenter : 
    def __init__(self , num_servers , arrival_rate , seed=42):
        self.servers = [Server(i , 100) for i in range(num_servers)]
        self.waiting_queue = []
        self.current_time = 0
        self.arrival_rate = arrival_rate
        self.next_job_id = 0
        self.rng = np.random.default_rng(seed)


        self.total_energy = 0.0   # total electricity used 
        self.total_completed = 0   #job finished
        self.sla_violation = 0       # Job that have missed their deadline
        self.utilization_history = []      # Server utlization each timestep

    
    # Generating Synthetic Data (Random Fake Jobs)
    def generate_new_jobs(self) :
        num_arrival = self.rng.poisson(self.arrival_rate)
        for _ in range(num_arrival) :
            job = Job(
                job_id = self.next_job_id,
                cpu_need = self.rng.integers(10, 40),
                deadline = self.current_time + self.rng.integers(10 , 50),
                duration = self.rng.integers(3,8)
                )
            
            self.next_job_id += 1
            self.waiting_queue.append(job)

    def tick(self) :
        self.current_time += 1

         # 1. Finish jobs. If a job finished LATE, that's an SLA violation.
        for server in self.servers:
            done = server.complete_finished_jobs(self.current_time)
            for job in done:
                self.total_completed += 1
                if job.finish_time > job.deadline:
                    self.sla_violations += 1

        # 2. Drop jobs that waited too long (missed deadline in the queue).
        still_waiting = []
        for job in self.waiting_queue:
            if self.current_time > job.deadline:
                self.sla_violations += 1     # never even got to run — a violation
            else:
                still_waiting.append(job)
        self.waiting_queue = still_waiting

        # 3. New customers arrive.
        self.generate_new_jobs()

        # 4. MEASURE energy + utilisation for this timestep.
        step_power = sum(s.power_use() for s in self.servers)   # total watts now
        self.total_energy += step_power / 3600                 # convert to watt-hours
        avg_util = np.mean([s.utilization() for s in self.servers])
        self.utilization_history.append(avg_util)

class RoundRobinScheduler:
    def __init__(self, num_servers):
        self.num_servers = num_servers
        self.next_server = 0

    def assign_jobs(self, datacenter):
        still_waiting = []
        for job in datacenter.waiting_queue:
            placed = False
            for _ in range(self.num_servers):
                server = datacenter.servers[self.next_server]
                self.next_server = (self.next_server + 1) % self.num_servers
                if server.FreeCPU(job):
                    server.Run_Job(job, datacenter.current_time)
                    placed = True
                    break
            if not placed:
                still_waiting.append(job)
        datacenter.waiting_queue = still_waiting

# ---- Run for 100 timesteps and print the FULL SCORECARD ----

dc = Datacenter(num_servers=5, arrival_rate=1.5)
scheduler = RoundRobinScheduler(num_servers=5)

for step in range(100):
    dc.tick()
    scheduler.assign_jobs(dc)

avg_utilization = np.mean(dc.utilization_history)

print("-" * 45)
print("ROUND ROBIN — SCORECARD")
print("-" * 45)
print(f"Total energy used: {dc.total_energy:.2f}")
print(f"Jobs completed: {dc.total_completed}")
print(f"SLA violations:  {dc.sla_violation}")
print(f"Average utilisation:  {avg_utilization * 100:.1f}%")
print("-" * 45)
print("\nThis is your BASELINE scorecard.")
print("Your RL agent's job: lower energy + lower SLA violations.")    
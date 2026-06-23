# Now : Servers Finish Jobs over time, and Round Robin Scheduler Assigns them.

import numpy as np


# ---- Job now also knows how long it taked to run.
class Job :
    def __init__(self, job_id , cpu_need , deadline , duration):
        self.job_id = job_id
        self.cpu_need = cpu_need
        self.deadline = deadline
        self.duration = duration
        self.finish_time = None

#  Server can now run jobs and finish them later

class Server :
    def __init__(self, server_id , cpu_capacity):
        self.server_id = server_id
        self.cpu_capacity = cpu_capacity
        self.cpu_used = 0
        self.running_jobs = []                  # jobs currently cooking on these server

    def FreeCPU(self, job) :
        return (self.cpu_capacity   - self.cpu_used)  >= job.cpu_need
    

    def Run_Job(self, job , current_time) :
        # start cooking a job . It will finish later.
        self.cpu_used += job.cpu_need
        job.finish_time = current_time + job.duration
        self.running_jobs.append(job)

    def Complete_Finish_job(self, current_time) :
        # Check which jobs are done. Finished jobs free up their CPU.
        finished = []
        for job in self.running_jobs[ : ] :       # the [:] makes a safe copy to loop over
            if job.finish_time <= current_time :
                self.cpu_used -= job.cpu_need
                self.running_jobs.remove(job)
                finished.append(job)
        return finished
    
class Datacenter:
    def __init__(self, num_servers, arrival_rate, seed=42):
        self.servers = [Server(i, 100) for i in range(num_servers)]
        self.waiting_queue = []
        self.current_time = 0
        self.arrival_rate = arrival_rate
        self.next_job_id = 0
        self.rng = np.random.default_rng(seed)

        # Stats we will report at the end
        self.total_completed = 0

    def generate_new_jobs(self):
        num_arrivals = self.rng.poisson(self.arrival_rate)
        for _ in range(num_arrivals):
            job = Job(
                job_id=self.next_job_id,
                cpu_need=self.rng.integers(10, 40),
                deadline=self.current_time + self.rng.integers(10, 50),
                duration=self.rng.integers(3, 8),     # runs for 3-7 timesteps
            )
            self.next_job_id += 1
            self.waiting_queue.append(job)

    def tick(self):
        self.current_time += 1
        # 1. Servers finish whatever is done
        for server in self.servers:
            done = server.Complete_Finish_job(self.current_time)
            self.total_completed += len(done)
        # 2. New customers arrive
        self.generate_new_jobs()

# The Round Robin Scheduler (our first decision-maker)

class RoundRobinScheduler:
    """Hands jobs to servers in turn: server 0, 1, 2, 0, 1, 2, ... like dealing cards."""

    def __init__(self, num_servers):
        self.num_servers = num_servers
        self.next_server = 0      # whose turn is it to receive a job

    def assign_jobs(self, datacenter):

        # Try to place each waiting job, starting from the front of the queue.

        still_waiting = []
        for job in datacenter.waiting_queue:
            placed = False

            # Try each server once, starting at whoever's turn it is.

            for _ in range(self.num_servers):
                server = datacenter.servers[self.next_server]
                self.next_server = (self.next_server + 1) % self.num_servers  # rotate
                if server.FreeCPU(job):
                    server.Run_Job(job, datacenter.current_time)
                    placed = True
                    break
            if not placed:
                still_waiting.append(job)   # nobody had room, keep waiting
        datacenter.waiting_queue = still_waiting

# ---- Run the whole thing for 30 timesteps ----

print("Building datacenter (5 servers) + Round Robin scheduler...\n")
dc = Datacenter(num_servers=5, arrival_rate=1.5)
scheduler = RoundRobinScheduler(num_servers=5)

for step in range(30):
    dc.tick()                    # clock moves, jobs finish, new jobs arrive
    scheduler.assign_jobs(dc)    # head waiter assigns waiting jobs

    if dc.current_time % 5 == 0:   # print every 5 steps to keep it short
        busy = sum(1 for s in dc.servers if s.cpu_used > 0)
        print(f"Time {dc.current_time:2d}: "
              f"{len(dc.waiting_queue)} waiting | "
              f"{busy}/5 servers busy | "
              f"{dc.total_completed} jobs completed so far")

print(f"\nDone! Total jobs completed: {dc.total_completed}")
print(f"Jobs still waiting: {len(dc.waiting_queue)}")
print("The queue stays small now — the scheduler is clearing it. The world works!")
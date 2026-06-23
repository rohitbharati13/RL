import numpy as np

class Job :
    def __init__(self , job_id , cpu_need , deadline):
        self.job_id = job_id
        self.cpu_need = cpu_need
        self.deadline = deadline
    
class Server :
    def __init__(self , server_id, cpu_capacity):
        self.server_id = server_id
        self.cpu_capacity = cpu_capacity
        self.cpu_used = 0

    def FreeCPU(self, job):
        return( self.cpu_capacity - self.cpu_used ) >= job.cpu_need


# Datacenter with many servers  + a queue of waiting jobs + a clock
class Datacenter :
    
    # Make a list of servers, This is a "This is a List comprehension" 
    # It's just a quick way to build a list of N servers.
    def __init__(self, num_servers , arrival_rate , seed = 42):
        self.servers = [
            Server(server_id= i, cpu_capacity= 100)
            for i in range(num_servers)
        ]

        self.waiting_queue = []
        self.current_time = 0
        self.arrival_rate = arrival_rate
        self.next_job_id = 0
    
    # A random-number generator with a fixed "seed" so results repeat.
        self.rng = np.random.default_rng(seed)
    
    def generate_new_jobs(self) :
        # This is synthetic data, Jobs arrive following  poisson pattern.
        # Poisson gives us a random count of arrivals this timestep
        # averging around arrival_rate. Soome timesteps 0 , some 3 , etc.

        num_arrivals = self.rng.poisson(self.arrival_rate)

        for _ in range(num_arrivals):
            job = Job(
                job_id=self.next_job_id,
                cpu_need =self.rng.integers(10, 40),       # random 10-39 CPU
                deadline=self.current_time + self.rng.integers(10, 50),
            )
            self.next_job_id += 1
            self.waiting_queue.append(job)

    def tick(self):
        """Move the clock forward by one step."""
        self.current_time += 1
        self.generate_new_jobs()

# ---- Let's run the clock for 10 timesteps and watch jobs arrive ----

print("Building a datacenter with 5 servers...")
dc = Datacenter(num_servers=5, arrival_rate=1.5)

print("Running the clock for 10 steps. Nobody assigns jobs yet — watch the queue grow.\n")

for step in range(10):
    dc.tick()
    print(f"Time {dc.current_time}: "
          f"{len(dc.waiting_queue)} jobs waiting in the queue")

print(f"\nAfter 10 steps, {len(dc.waiting_queue)} jobs are stuck waiting.")
print("Problem: no one is assigning them to servers! That's what we build next.")


# Lets Start with the two smallest LEGO bricks : 
# A job and a Server

class Job :

# single job is comming to the server
    def __init__(self, job_id, cpu_need , deadline):
        self.job_id = job_id
        self.cpu_need = cpu_need
        self.deadline = deadline


class Server :

# Server features
    def __init__(self, server_id , cpu_capacity):
        self.server_id = server_id
        self.cpu_capacity = cpu_capacity
        self.cpu_used = 0
# Can server fits the Job?
    def FreeCPU(self, job):
        free_cpu = self.cpu_capacity - self.cpu_used
        return free_cpu >= job.cpu_need

# PUt the job into Server
    def Run_Job(self , job) :
        self.cpu_used = self.cpu_used + job.cpu_need
        print(f" Server {self.server_id} is now running job {job.job_id}."
              f"CPU used : {self.cpu_used} / {self.cpu_capacity}")
        



# Test the Bricks

print("Making the Server with 100 CPU")
Sev = Server( server_id=1 , cpu_capacity= 100)

print("Making Job that need 60 CPU power")
My_job = Job(job_id=1, cpu_need=60 , deadline= 10)

print("Does the server have room?")
if Sev.FreeCPU(My_job):
    print("  Yes! Let's run it.")
    Sev.Run_Job(My_job)
else:
    print("  No room.")
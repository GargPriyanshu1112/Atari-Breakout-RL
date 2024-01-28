import numpy as np
import itertools
import time
import threading
import multiprocessing


class Worker:
    def __init__(self, id_, global_counter):
        self.id = id_
        self.global_counter = global_counter
        self.local_counter = itertools.count()

    def run(self):
        while True:
            time.sleep(np.random.rand() * 3)
            global_step = next(self.global_counter)
            local_step = next(self.local_counter)
            print(f"Worker({self.id}): {local_step}")
            if global_step >= 20:
                break


global_counter = itertools.count()
num_workers = multiprocessing.cpu_count()
print(f"Num cpu cores: {num_workers}")

# Create the workers
workers = []
for worker_id in range(num_workers):
    worker = Worker(worker_id, global_counter)
    workers.append(worker)

# Start the threads
worker_threads = []
for worker in workers:
    worker_fn = lambda: worker.run()
    t = threading.Thread(target=worker_fn)
    t.start()
    worker_threads.append(t)

for t in worker_threads:
    t.join()

print("Done")

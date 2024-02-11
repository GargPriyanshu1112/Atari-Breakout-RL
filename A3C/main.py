"""
When to close env...
"""

import itertools
import threading
import tensorflow as tf

from networks import get_networks
from workers import Worker
from config import (
    INP_SHAPE,
    NUM_ACTIONS,
    NUM_WORKERS,
    DISCOUNT_FACTOR,
    MAX_STEPS,
    STEPS_BEFORE_UPDATE,
)


if __name__ == "__main__":
    # Global value and policy networks
    value_network, policy_network = get_networks(INP_SHAPE, NUM_ACTIONS)

    returns_list = []
    steps_counter = itertools.count()
    coordinator = tf.train.Coordinator()  # threads coordinator

    workers = []
    for worker_id in range(NUM_WORKERS):
        worker = Worker(
            f"worker_#{worker_id+1}",
            steps_counter,
            value_network,
            policy_network,
            returns_list,
            DISCOUNT_FACTOR,
            MAX_STEPS,
        )
        workers.append(worker)

    worker_threads = []
    for worker in workers:
        worker_func = lambda: worker.run(coordinator, STEPS_BEFORE_UPDATE)
        t = threading.Thread(target=worker_func)
        t.start()
        worker_threads.append(t)

    # Wait for all workers to finish
    coordinator.join(
        worker_threads, stop_grace_period_secs=300
    )  # raises error if threads don't terminate within 5 minutes

    # Add plot function

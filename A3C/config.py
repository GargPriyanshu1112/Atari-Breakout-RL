MAX_STEPS = 5e6
STEPS_BEFORE_UPDATE = 5  # No. of steps each worker performs before sending their gradients to the global networks
NUM_ACTIONS = 4
INP_SHAPE = (84, 84, 4)
DISCOUNT_FACTOR = 0.99
NUM_STACKED_FRAMES = 4

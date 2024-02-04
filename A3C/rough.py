# class PolicyNetwork:
#     def __init__(self, reg_const=0.01):
#         self.a = reg_const


# class ValueNetwork(PolicyNetwork):
#     def __init__(self):
#         super().__init__()


# p = PolicyNetwork()
# v = ValueNetwork()

# print(v.a)


import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# Define logits
logits = np.log([0.1, 0.5, 0.4])

# Apply softmax to logits
probs = tf.nn.softmax(logits)

# Create a Categorical distribution using probabilities
dist_probs = tfp.distributions.Categorical(probs=probs)

# Create a Categorical distribution using logits
dist_logits = tfp.distributions.Categorical(logits=logits)

# Sample from the distributions
sample_probs = dist_probs.sample()
sample_logits = dist_logits.sample()

# Get log probabilities of the samples
log_prob_probs = dist_probs.log_prob(sample_probs)
log_prob_logits = dist_logits.log_prob(sample_logits)

print("Sampled outcome (from probs):", sample_probs.numpy())
print("Log Probability (from probs):", log_prob_probs.numpy())

print("Sampled outcome (from logits):", sample_logits.numpy())
print("Log Probability (from logits):", log_prob_logits.numpy())


"""
(5, 84, 84, 4)
[<tf.Tensor: shape=(), dtype=int32, numpy=2>, <tf.Tensor: shape=(), dtype=int32, numpy=2>, <tf.Tensor: shape=(), dtype=int32, numpy=1>, <tf.Tensor: shape=(), dtype=int32, numpy=2>, <tf.Tensor: shape=(), dtype=int32, numpy=3>]
[<tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[-4.4956163e-05]], dtype=float32)>, <tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[-2.2664055e-05]], dtype=float32)>, <tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[-2.6610243e-05]], dtype=float32)>, <tf.Tensor: shape=(1, 1), dtype=float32, 
numpy=array([[-5.539361e-05]], dtype=float32)>, <tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[-9.778974e-05]], dtype=float32)>]
[<tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[0.00035296]], dtype=float32)>, <tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[0.00034943]], 
dtype=float32)>, <tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[0.00034594]], dtype=float32)>, <tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[0.00034248]], dtype=float32)>, <tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[0.00033905]], dtype=float32)>]
"""

C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\gym\utils\passive_env_checker.py:233: DeprecationWarning: `np.bool8` is a deprecated alias for `np.bool_`.  
(Deprecated NumPy 1.24)
  if not isinstance(terminated, (bool, np.bool8)):
Exception in thread Thread-4 (<lambda>):
Exception in thread Thread-1 (<lambda>):
Traceback (most recent call last):
Traceback (most recent call last):
Exception in thread Thread-3 (<lambda>):
Traceback (most recent call last):
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\threading.py", line 1038, in _bootstrap_inner
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\threading.py", line 1038, in _bootstrap_inner
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\threading.py", line 1038, in _bootstrap_inner
    self.run()
    self.run()
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\threading.py", line 975, in run
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\threading.py", line 975, in run
    self.run()
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\threading.py", line 975, in run
    self._target(*self._args, **self._kwargs)
    self._target(*self._args, **self._kwargs)
  File "H:\Reinforcement Learning\Project\Atari-Breakout-RL\A3C\main.py", line 44, in <lambda>
  File "H:\Reinforcement Learning\Project\Atari-Breakout-RL\A3C\main.py", line 44, in <lambda>
    self._target(*self._args, **self._kwargs)
    worker_func = lambda: worker.run(coordinator, STEPS_BEFORE_UPDATE)
  File "H:\Reinforcement Learning\Project\Atari-Breakout-RL\A3C\main.py", line 44, in <lambda>
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    worker_func = lambda: worker.run(coordinator, STEPS_BEFORE_UPDATE)
    worker_func = lambda: worker.run(coordinator, STEPS_BEFORE_UPDATE)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "H:\Reinforcement Learning\Project\Atari-Breakout-RL\A3C\workers.py", line 126, in run
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "H:\Reinforcement Learning\Project\Atari-Breakout-RL\A3C\workers.py", line 126, in run
    self.update(steps_data)
  File "H:\Reinforcement Learning\Project\Atari-Breakout-RL\A3C\workers.py", line 153, in update
    self.update(steps_data)
  File "H:\Reinforcement Learning\Project\Atari-Breakout-RL\A3C\workers.py", line 126, in run
  File "H:\Reinforcement Learning\Project\Atari-Breakout-RL\A3C\workers.py", line 153, in update
    self.update_global_weights(np.array(states), actions, advantages, value_targets)
  File "H:\Reinforcement Learning\Project\Atari-Breakout-RL\A3C\workers.py", line 55, in update_global_weights
    self.update_global_weights(np.array(states), actions, advantages, value_targets)
  File "H:\Reinforcement Learning\Project\Atari-Breakout-RL\A3C\workers.py", line 55, in update_global_weights
    self.global_value_network.optimizer.apply_gradients(
    self.update(steps_data)
    self.global_value_network.optimizer.apply_gradients(
  File "H:\Reinforcement Learning\Project\Atari-Breakout-RL\A3C\workers.py", line 153, in update
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\optimizers\optimizer.py", line 1223, in apply_gradients
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\optimizers\optimizer.py", line 1223, in apply_gradients
    self.update_global_weights(np.array(states), actions, advantages, value_targets)
  File "H:\Reinforcement Learning\Project\Atari-Breakout-RL\A3C\workers.py", line 55, in update_global_weights
    return super().apply_gradients(grads_and_vars, name=name)
    return super().apply_gradients(grads_and_vars, name=name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\optimizers\optimizer.py", line 652, in apply_gradients
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\optimizers\optimizer.py", line 652, in apply_gradients
    self.global_value_network.optimizer.apply_gradients(
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\optimizers\optimizer.py", line 1223, in apply_gradients
    iteration = self._internal_apply_gradients(grads_and_vars)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    iteration = self._internal_apply_gradients(grads_and_vars)
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\optimizers\optimizer.py", line 1253, in _internal_apply_gradients
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    return super().apply_gradients(grads_and_vars, name=name)
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\optimizers\optimizer.py", line 1253, in _internal_apply_gradients
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\optimizers\optimizer.py", line 652, in apply_gradients
    return tf.__internal__.distribute.interim.maybe_merge_call(
    return tf.__internal__.distribute.interim.maybe_merge_call(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\tensorflow\python\distribute\merge_call_interim.py", line 51, in maybe_merge_call
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    iteration = self._internal_apply_gradients(grads_and_vars)
    return fn(strategy, *args, **kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\optimizers\optimizer.py", line 1253, in _internal_apply_gradients
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\optimizers\optimizer.py", line 1345, in _distributed_apply_gradients_fn
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\tensorflow\python\distribute\merge_call_interim.py", line 51, in maybe_merge_call
    distribution.extended.update(
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\tensorflow\python\distribute\distribute_lib.py", line 3013, in update
    return fn(strategy, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\optimizers\optimizer.py", line 1345, in _distributed_apply_gradients_fn
    return tf.__internal__.distribute.interim.maybe_merge_call(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\tensorflow\python\distribute\merge_call_interim.py", line 51, in maybe_merge_call
    return fn(strategy, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\optimizers\optimizer.py", line 1345, in _distributed_apply_gradients_fn
    return self._update(var, fn, args, kwargs, group)
    distribution.extended.update(
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\tensorflow\python\distribute\distribute_lib.py", line 3013, in update
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    distribution.extended.update(
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\tensorflow\python\distribute\distribute_lib.py", line 4083, in _update
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\tensorflow\python\distribute\distribute_lib.py", line 3013, in update
    return self._update(var, fn, args, kwargs, group)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\tensorflow\python\distribute\distribute_lib.py", line 4083, in _update
    return self._update(var, fn, args, kwargs, group)
    return self._update_non_slot(var, fn, (var,) + tuple(args), kwargs, group)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\tensorflow\python\distribute\distribute_lib.py", line 4083, in _update
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\tensorflow\python\distribute\distribute_lib.py", line 4089, in _update_non_slot
    return self._update_non_slot(var, fn, (var,) + tuple(args), kwargs, group)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\tensorflow\python\distribute\distribute_lib.py", line 4089, in _update_non_slot
    return self._update_non_slot(var, fn, (var,) + tuple(args), kwargs, group)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\tensorflow\python\distribute\distribute_lib.py", line 4089, in _update_non_slot
    result = fn(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\tensorflow\python\autograph\impl\api.py", line 596, in wrapper
    result = fn(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^
    result = fn(*args, **kwargs)
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\tensorflow\python\autograph\impl\api.py", line 596, in wrapper
             ^^^^^^^^^^^^^^^^^^^
    return func(*args, **kwargs)
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\tensorflow\python\autograph\impl\api.py", line 596, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\optimizers\optimizer.py", line 1342, in apply_grad_to_update_var
           ^^^^^^^^^^^^^^^^^^^^^
    return func(*args, **kwargs)
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\optimizers\optimizer.py", line 1342, in apply_grad_to_update_var
    return self._update_step(grad, var)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\optimizers\optimizer.py", line 1342, in apply_grad_to_update_var
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\optimizers\optimizer.py", line 233, in _update_step
    return self._update_step(grad, var)
    return self._update_step(grad, var)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\optimizers\optimizer.py", line 233, in _update_step
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    raise KeyError(
  File "C:\Users\lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\optimizers\optimizer.py", line 233, in _update_step
KeyError: 'The optimizer cannot recognize variable conv2d_6/kernel:0. This usually means you are trying to call the optimizer to update different parts of the model separately. Please call `optimizer.build(variables)` with the full list of trainable variables before the training loop or use legacy optimizer `tf.keras.optimizers.legacy.RMSprop.'
    raise KeyError(
KeyError: 'The optimizer cannot recognize variable conv2d_8/kernel:0. This usually means you are trying to call the optimizer to update different parts of the model separately. Please call `optimizer.build(variables)` with the full list of trainable variables before the training loop or use legacy optimizer `tf.keras.optimizers.legacy.RMSprop.'
    raise KeyError(
KeyError: 'The optimizer cannot recognize variable conv2d_2/kernel:0. This usually means you are trying to call the optimizer to update different parts of the model separately. Please call `optimizer.build(variables)` with the full list of trainable variables before the training loop or use legacy optimizer `tf.keras.optimizers.legacy.RMSprop.'
Episode Reward: 2.0 - worker_#2
Episode Reward: 1.0 - worker_#2
Episode Reward: 0.0 - worker_#2
Episode Reward: 3.0 - worker_#2
Episode Reward: 2.0 - worker_#2
Episode Reward: 2.0 - worker_#2
Episode Reward: 1.0 - worker_#2
Episode Reward: 0.0 - worker_#2
Episode Reward: 2.0 - worker_#2
Episode Reward: 1.0 - worker_#2
Episode Reward: 0.0 - worker_#2
Episode Reward: 3.0 - worker_#2
Episode Reward: 1.0 - worker_#2
Episode Reward: 1.0 - worker_#2
Episode Reward: 3.0 - worker_#2
Episode Reward: 4.0 - worker_#2
Episode Reward: 0.0 - worker_#2
Episode Reward: 0.0 - worker_#2
Episode Reward: 1.0 - worker_#2
Episode Reward: 0.0 - worker_#2
Episode Reward: 0.0 - worker_#2
Episode Reward: 0.0 - worker_#2
Episode Reward: 0.0 - worker_#2
Episode Reward: 1.0 - worker_#2
Episode Reward: 2.0 - worker_#2
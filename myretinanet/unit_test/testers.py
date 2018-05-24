# import tensorflow as tf
import matplotlib.pyplot as plt



# A collection of Learning Rate schecules:
# third_party/tensorflow_models/object_detection/utils/learning_schedules.py
def learning_rate_schedule(base_learning_rate, lr_warmup_init, lr_warmup_step,
                            lr_drop_step, global_step):
  """Handles linear scaling rule, gradual warmup, and LR decay."""
  # lr_warmup_init is the starting learning rate; the learning rate is linearly
  # scaled up to the full learning rate after `lr_warmup_steps` before decaying.
  lr_warmup_remainder = 1.0 - lr_warmup_init
  linear_warmup = [
      (lr_warmup_init + lr_warmup_remainder * (float(step) / lr_warmup_step),
       step) for step in range(0, lr_warmup_step, max(1, lr_warmup_step // 100))
  ]
  lr_schedule = linear_warmup + [[1.0, lr_warmup_step], [0.1, lr_drop_step]]
  learning_rate = base_learning_rate
  for mult, start_global_step in lr_schedule:
    if global_step >= start_global_step:
      learning_rate = base_learning_rate * mult

  return learning_rate

if __name__ == '__main__':
  lr = []
  for step in range(0, 100000):
    lr.append(learning_rate_schedule(base_learning_rate=0.08,
                                     lr_warmup_init=0.1,
                                     lr_warmup_step=2000,
                                     lr_drop_step=15000,
                                     global_step=step))

  plt.plot(range(0, 100000), lr)
  plt.pause(-1)
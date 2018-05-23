# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model defination for the RetinaNet Model.

Defines model_fn of RetinaNet for TF Estimator. The model_fn includes RetinaNet
model architecture, loss function, learning rate schedule, and evaluation
procedure.

T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar
Focal Loss for Dense Object Detection. arXiv:1708.02002
"""

import tensorflow as tf


def default_hparams():
  return tf.contrib.training.HParams(
      image_size=640,
      input_rand_hflip=True,

      # dataset specific parameters
      num_classes=90,
      skip_crowd=True,

      # model architecture
      min_level=3,
      max_level=7,
      num_scales=3,
      aspect_ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
      anchor_scale=4.0,
      resnet_depth=50,

      # is batchnorm training mode
      is_training_bn=True,

      # optimization
      momentum=0.9,
      learning_rate=0.08,
      weight_decay = 1e-4,
      lr_warmup_init=0.1,
      lr_warmup_step=2000,
      lr_drop_step=15000,

      # classification loss
      alpha=0.25,
      gamma=1.5,

      # localization loss
      delta=0.1,
      box_loss_weight=50.0,

      # resnet checkpoint
      resnet_checkpoint=None,

      # output detection
      box_max_detected=100,
      box_iou_threshold=0.5,
      use_bfloat16=False,
  )


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
    learning_rate = tf.where(global_step < start_global_step, learning_rate,
                             base_learning_rate * mult)
  return learning_rate


def focal_loss(logits, targets, alpha, gamma, normalizer):
  """Compute the focal loss between `logits` and the golden `target` values.

  Focal loss = -(1-alpha)^gamma * log(pt)
  where pt is the probability of being classified to the true class.

  Args:
    logits: A float32 tensor of size
      [batch, height_in, width_in, num_predictions].
    targets: A float32 tensor of size
      [batch, height_in, width_in, num_predictions].
    alpha: A float32 scalar multiplying alpha to the loss from positive examples
      and (1-alpha) to the loss from negative examples.
    gamma: A float32 scalar modulating loss from hard and easy examples.
    normalizer: A float32 scalar normalizes the total loss from all examples.
  Returns:
    loss: A float32 scalar representing normalized total loss.
  """
  with tf.name_scope('focal_loss'):
    positive_label_mask = tf.equal(targets, 1.0)
    cross_entropy = (
        tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
    probs = tf.sigmoid(logits)
    probs_gt = tf.where(positive_label_mask, probs, 1.0 - probs)
    # With small gamma, the implementation could produce NaN during back prop.
    modulator = tf.pow(1.0 - probs_gt, gamma)
    loss = modulator * cross_entropy
    weighted_loss = tf.where(positive_label_mask, alpha * loss,
                             (1.0 - alpha) * loss)
    total_loss = tf.reduce_sum(weighted_loss)
    total_loss /= normalizer
  return total_loss


def _classification_loss(cls_outputs,
                         cls_targets,
                         num_positives,
                         alpha=0.25,
                         gamma=2.0):
  """Computes classification loss."""
  normalizer = num_positives
  classification_loss = focal_loss(cls_outputs, cls_targets, alpha, gamma,
                                   normalizer)
  return classification_loss


def _box_loss(box_outputs, box_targets, num_positives, delta=0.1):
  """Computes box regression loss."""
  # delta is typically around the mean value of regression target.
  # for instances, the regression targets of 512x512 input with 6 anchors on
  # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
  normalizer = num_positives * 4.0
  mask = tf.not_equal(box_targets, 0.0)
  box_loss = tf.losses.huber_loss(
      box_targets,
      box_outputs,
      weights=mask,
      delta=delta,
      reduction=tf.losses.Reduction.SUM)
  box_loss /= normalizer
  return box_loss


def detection_loss(cls_outputs, box_outputs, labels, params):
  """Computes total detection loss.

  Computes total detection loss including box and class loss from all levels.
  Args:
    cls_outputs: an OrderDict with keys representing levels and values
      representing logits in
      [batch_size, height, width, num_anchors * num_classes].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4].
    labels: the dictionary that returned from dataloader that includes
      groundturth targets.
    params: the dictionary including training parameters specified in
      default_haprams function in this file.
  Returns:
    total_loss: an integar tensor representing total loss reducing from
      class and box losses from all levels.
    cls_loss: an integar tensor representing total class loss.
    box_loss: an integar tensor representing total box regression loss.
  """
  # Sum all positives in a batch for normalization and avoid zero
  # num_positives_sum, which would lead to inf loss during training
  num_positives_sum = tf.reduce_sum(labels['mean_num_positives']) + 1.0
  levels = cls_outputs.keys()

  cls_losses = []
  box_losses = []
  for level in levels:
    # Onehot encoding for classification labels.
    cls_targets_at_level = tf.one_hot(
        labels['cls_targets_%d' % level],
        params['num_classes'])
    bs, width, height, _, _ = cls_targets_at_level.get_shape().as_list()
    cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                      [bs, width, height, -1])
    box_targets_at_level = labels['box_targets_%d' % level]
    cls_losses.append(
        _classification_loss(
            cls_outputs[level],
            cls_targets_at_level,
            num_positives_sum,
            alpha=params['alpha'],
            gamma=params['gamma']))
    box_losses.append(
        _box_loss(
            box_outputs[level],
            box_targets_at_level,
            num_positives_sum,
            delta=params['delta']))

  # Sum per level losses to total loss.
  cls_loss = tf.add_n(cls_losses)
  box_loss = tf.add_n(box_losses)
  total_loss = cls_loss + params['box_loss_weight'] * box_loss
  return total_loss, cls_loss, box_loss
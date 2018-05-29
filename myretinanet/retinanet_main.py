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
"""Training script for RetinaNet.
"""

import tensorflow as tf
from absl import flags
from tensorflow.contrib import slim
from myretinanet import retinanet_model
from myretinanet.dataset.cocodataset import CoCoDataset
from myretinanet.network.retinanet_arch import retinanet
from myretinanet.utils import anchors, coco_metric
from collections import namedtuple

help_dict={
  'resnet_checkpoint'    : 'Location of the ResNet50 checkpoint to use for model initialization.',
  'retinanet_checkpoint' : 'Location of the retinanet checkpoint to use for model initialization.',
  'training_file_pattern': 'Glob for training data files (e.g., COCO train - minival set)',
  'hparams'              : 'Comma separated k=v pairs of hyperparameters.',
  'train_batch_size'     : 'training batch size',
  'log_step'  : 'Number of iterations per TPU training loop',
  'num_epochs'           : 'Number of epochs for training',
  'examples_per_epoch'   : 'Number of examples in one epoch',

  'eval_after_training'  : 'Run one eval after the training finishes.',
  'eval_log_dir'         : 'Location of log_dir in evaluation',
  'eval_num'             : 'Maximum evaluation number.',
  'eval_steps'           : 'evaluation steps',
  'eval_file_pattern'    : 'Glob for evaluation tfrecords (e.g., COCO val2017 set)',
  'eval_json_file'       : 'COCO validation JSON containing golden bounding boxes.',
  'eval_batch_size'      : 'batch size used in evaluation mode.',

  'mode'                 : 'Mode to run: train or eval (default: train)',
  'model_dir'            : 'Location of model_dir',
  'use_xla'              : 'Use XLA even if use_tpu is false.  If use_tpu is true, we always use XLA, and this flag has no effect.'
}


def arg_def(name, default_val):
  return name, default_val, help_dict[name]


Param = namedtuple('ParamStruct', [
  'resnet_checkpoint',
  'retinanet_checkpoint',
  'training_file_pattern',
  'hparams',
  'train_batch_size',
  'log_step',
  'num_epochs',
  'examples_per_epoch',
  'eval_after_training',
  'eval_log_dir',
  'eval_num',
  'eval_file_pattern',
  'eval_json_file',
  'eval_batch_size',
  'mode',
  'model_dir',
  'use_xla',

  'image_size',
  'input_rand_hflip',
  'num_classes',
  'skip_crowd',
  'min_level',
  'max_level',
  'num_scales',
  'aspect_ratios',
  'anchor_scale',
  'resnet_depth',
  'is_training_bn',
  'momentum',
  'learning_rate',
  'weight_decay',
  'alpha',
  'gamma',
  'pi',
  'box_loss_weight',
  'box_max_detected',
  'box_iou_threshold'
])


def inputParam():

  # For train
  flags.DEFINE_string(*arg_def('resnet_checkpoint', ''))
  flags.DEFINE_string(*arg_def('retinanet_checkpoint', ''))
  flags.DEFINE_string(*arg_def('training_file_pattern', None))
  flags.DEFINE_string(*arg_def('hparams', ''))
  flags.DEFINE_integer(*arg_def('train_batch_size', 16))
  flags.DEFINE_integer(*arg_def('log_step', 100))
  flags.DEFINE_integer(*arg_def('num_epochs', 15))
  flags.DEFINE_integer(*arg_def('examples_per_epoch', 120000))

  # For Eval mode
  flags.DEFINE_bool(*arg_def('eval_after_training', False))
  flags.DEFINE_string(*arg_def('eval_log_dir', None))
  flags.DEFINE_integer(*arg_def('eval_num', None))
  flags.DEFINE_integer(*arg_def('eval_batch_size', 1))
  flags.DEFINE_string(*arg_def('eval_file_pattern', None))
  flags.DEFINE_string(*arg_def('eval_json_file',''))

  # For both
  flags.DEFINE_string(*arg_def('mode','train'))
  flags.DEFINE_string(*arg_def('model_dir', None))
  flags.DEFINE_bool(*arg_def('use_xla', False))

  return flags.FLAGS


def checkInputParam(FLAGS):
  if FLAGS.mode is 'train' and FLAGS.training_file_pattern is None:
    raise RuntimeError('You must specify --training_file_pattern for training.')

  if FLAGS.mode is 'eval':
    if FLAGS.valid_data_dir is None:
      raise RuntimeError('You must specify --valid_data_dir for evaluation.')

    if FLAGS.eval_json_file is None:
      raise RuntimeError('You must specify --eval_json_file for evaluation.')


def initParam(input_flag):
  params = Param(
    # For train
    resnet_checkpoint=input_flag.resnet_checkpoint,
    retinanet_checkpoint=input_flag.retinanet_checkpoint,

    training_file_pattern=input_flag.training_file_pattern,
    hparams=input_flag.hparams,
    train_batch_size=input_flag.train_batch_size,
    log_step=input_flag.log_step,
    num_epochs=input_flag.num_epochs,
    examples_per_epoch=input_flag.examples_per_epoch,

    # For eval
    eval_after_training=input_flag.eval_after_training,
    eval_log_dir=input_flag.eval_log_dir,
    eval_num=input_flag.eval_num,
    eval_file_pattern=input_flag.eval_file_pattern,
    eval_json_file=input_flag.eval_json_file,
    eval_batch_size=input_flag.eval_batch_size,

    # Shared  settings
    mode=input_flag.mode,
    model_dir=input_flag.model_dir,
    use_xla=input_flag.use_xla,

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
    learning_rate=0.01,
    weight_decay=1e-4,
    # lr_warmup_init=0.1,
    # lr_warmup_step=2000,
    # lr_drop_step=15000,

    # classification loss
    alpha=0.25,
    gamma=2,
    pi=0.01,

    # localization loss
    box_loss_weight=10.0,

    # output detection
    box_max_detected=100,
    box_iou_threshold=0.5
  )

  return params


def metric_fn(mt_input, params):
  """Evaluation metric fn. Performed on CPU, do not reference TPU ops."""
  eval_anchors = anchors.Anchors(params.min_level,
                                 params.max_level,
                                 params.num_scales,
                                 params.aspect_ratios,
                                 params.anchor_scale,
                                 params.image_size)
  anchor_labeler = anchors.AnchorLabeler(eval_anchors,
                                         params.num_classes)
  cls_loss = tf.metrics.mean(mt_input['cls_loss_repeat'])
  box_loss = tf.metrics.mean(mt_input['box_loss_repeat'])

  # add metrics to output
  cls_outputs = {}
  box_outputs = {}
  for level in range(params.min_level, params.max_level + 1):
    cls_outputs[level] = mt_input['cls_outputs_%d' % level]
    box_outputs[level] = mt_input['box_outputs_%d' % level]

  detections = anchor_labeler.generate_detections(
    cls_outputs, box_outputs, mt_input['source_ids'])
  eval_metric = coco_metric.EvaluationMetric(params.eval_json_file)
  values, updates = eval_metric.estimator_metric_fn(detections,
                                                 mt_input['image_scales'])
  # Add metrics to output.
  output_values = {
    'cls_loss': cls_loss,
    'box_loss': box_loss,
  }
  output_values.update(values)
  return output_values, updates

FLAGS = inputParam()


def main(_):

  checkInputParam(FLAGS)

  params = initParam(FLAGS)

  # Config session.
  # config_proto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  if params.mode == 'train':
    is_training = True
  else:
    is_training = False

  if params.mode == 'train':
    # Prepare input data
    coco_train = CoCoDataset(record_path=params.training_file_pattern,
                             is_training=is_training,
                             batch_size=params.train_batch_size,
                             params=params)
    imgs, glabels = coco_train.get_next()

    # Create network
    logits, pboxes = retinanet(imgs, params.weight_decay,
                               ckpt_file=params.resnet_checkpoint,
                               num_classes=params.num_classes,
                               is_training=is_training,
                               num_anchors=len(params.aspect_ratios) * params.num_scales)

    # Select trainable variables.
    vars_train = tf.trainable_variables(scope='resnet_v2_50')
    vars_train += tf.trainable_variables(scope='retinanet')
    vars_train += tf.trainable_variables(scope='resnet_fpn')


    # Compute loss
    # cls_loss and box_loss are for logging. only total_loss is optimized.
    total_loss, cls_loss, box_loss = retinanet_model.detection_loss(logits, pboxes, glabels, params)
    weight_loss = params.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in vars_train if 'bias' not in v.name])
    tf.losses.add_loss(total_loss)
    tf.losses.add_loss(weight_loss)

    # Get loss
    loss  = tf.losses.get_total_loss()

    # Create optimizer
    tf.train.create_global_step()
    global_step = tf.train.get_global_step()
    lr = retinanet_model.learning_rate_schedule(params.learning_rate, global_step)
    optimizer = tf.train.MomentumOptimizer(lr, params.momentum)

    # Create train operation.
    train_op = slim.learning.create_train_op(total_loss=loss,
                                             optimizer=optimizer,
                                             global_step=global_step,
                                             variables_to_train=vars_train)
    # Learn using GPU
    max_steps = int((params.num_epochs * params.examples_per_epoch) / params.train_batch_size)
    slim.learning.train(train_op=train_op,
                        logdir=params.model_dir,
                        log_every_n_steps=params.log_step,
                        global_step=global_step,
                        number_of_steps=max_steps,
                        init_feed_dict={lr: params.learning_rate},
                        save_interval_secs=1000 * 50,  # 1000 step save once
                        save_summaries_secs=10 * 50    # ten step
    )

  elif params.mode == 'eval':
    # eval only runs on CPU or GPU host with batch_size = 1
    # Prepare input data
    coco_train = CoCoDataset(record_path=params.eval_file_pattern,
                             is_training=is_training,
                             batch_size=1,
                             params=params)
    imgs, glabels = coco_train.get_next()

    # Create network
    logits, pboxes = retinanet(imgs, params.weight_decay,
                               ckpt_file=params.resnet_checkpoint,
                               num_classes=params.num_classes,
                               is_training=is_training,
                               num_anchors=len(params.aspect_ratios) * params.num_scales)


    # Compute loss
    # cls_loss and box_loss are for logging. only total_loss is optimized.
    total_loss, cls_loss, box_loss = retinanet_model.detection_loss(logits, pboxes,
                                                                    glabels, params)

    cls_loss_repeat = tf.reshape(
        tf.tile(tf.expand_dims(cls_loss, 0), [
            params.eval_batch_size,
        ]), [params.eval_batch_size, 1])
    box_loss_repeat = tf.reshape(
        tf.tile(tf.expand_dims(box_loss, 0), [
            params.eval_batch_size,
        ]), [params.eval_batch_size, 1])

    metric_fn_inputs = {
        'cls_loss_repeat': cls_loss_repeat,
        'box_loss_repeat': box_loss_repeat,
        'source_ids': glabels['source_ids'],
        'image_scales': glabels['image_scales'],
    }
    for level in range(params.min_level, params.max_level + 1):
      metric_fn_inputs['cls_outputs_%d' % level] = logits[level]
      metric_fn_inputs['box_outputs_%d' % level] = pboxes[level]

    value_ops,  update_ops= metric_fn(metric_fn_inputs, params)

    slim.evaluation.evaluate_once('',
                                  params.retinanet_checkpoint,
                                  logdir=params.eval_log_dir,
                                  num_evals=params.eval_num,
                                  eval_op=update_ops,
                                  final_op=value_ops)

  else:
    tf.logging.info('Mode not found.')

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)

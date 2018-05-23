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

import os

import tensorflow as tf
from collections import namedtuple
from absl import flags
from tensorflow.contrib import slim
from myretinanet import retinanet_model
from myretinanet.dataset.cocodataset import CoCoDataset
from myretinanet.network.retinanet_arch import retinanet
from myretinanet.unit_test import testers

def inputParam():

  flags.DEFINE_bool(
      'use_xla',
      False,
      """Use XLA even if use_tpu is false.  If use_tpu is true, we always use XLA, and this flag has no effect.""")
  flags.DEFINE_string('model_dir',
                      None,
                      'Location of model_dir')

  flags.DEFINE_string('resnet_checkpoint',
                      '',
                      'Location of the ResNet50 checkpoint to use for model initialization.')
  flags.DEFINE_string('hparams',
                      '',
                      'Comma separated k=v pairs of hyperparameters.')
  flags.DEFINE_integer('train_batch_size',
                       64,
                       'training batch size')
  flags.DEFINE_integer('eval_steps',
                       5000,
                       'evaluation steps')
  flags.DEFINE_integer('iterations_per_loop',
                       100,
                       'Number of iterations per TPU training loop')
  flags.DEFINE_string('training_file_pattern',
                      None,
                      'Glob for training data files (e.g., COCO train - minival set)')
  flags.DEFINE_string('validation_file_pattern',
                      None,
                      'Glob for evaluation tfrecords (e.g., COCO val2017 set)')
  flags.DEFINE_string('val_json_file',
                      '',
                      'COCO validation JSON containing golden bounding boxes.')
  flags.DEFINE_integer('num_examples_per_epoch',
                       120000,
                       'Number of examples in one epoch')
  flags.DEFINE_integer('num_epochs',
                       15,
                       'Number of epochs for training')
  flags.DEFINE_string('mode',
                      'train',
                      'Mode to run: train or eval (default: train)')
  flags.DEFINE_bool('eval_after_training',
                    False,
                    'Run one eval after the '
                    'training finishes.')

  # For Eval mode
  flags.DEFINE_integer('min_eval_interval',
                       180,
                       'Minimum seconds between evaluations.')
  flags.DEFINE_integer('eval_timeout',
                       None,
                       'Maximum seconds between checkpoints before evaluation terminates.')

  return flags.FLAGS

def checkInputParam(FLAGS):
  if FLAGS.mode is 'train' and FLAGS.training_file_pattern is None:
    raise RuntimeError('You must specify --training_file_pattern for training.')

  if FLAGS.mode is 'eval':
    if FLAGS.valid_data_dir is None:
      raise RuntimeError('You must specify --valid_data_dir for evaluation.')

    if FLAGS.val_json_file is None:
      raise RuntimeError('You must specify --val_json_file for evaluation.')

FLAGS = inputParam()

def main(_):

  checkInputParam(FLAGS)

  # Parse hparams
  hparams = retinanet_model.default_hparams()
  hparams.parse(FLAGS.hparams)

  params = dict(
      hparams.values(),
      resnet_checkpoint=FLAGS.resnet_checkpoint,
      val_json_file=FLAGS.val_json_file,
      mode=FLAGS.mode,
  )

  # Config session.
  # config_proto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  if params['mode'] == 'train':
    is_training = True
  else:
    is_training = False

  if FLAGS.mode == 'train':
    # Prepare input data
    coco_train = CoCoDataset(record_path=FLAGS.training_file_pattern,
                             is_training=is_training,
                             batch_size=FLAGS.train_batch_size,
                             params=params)
    imgs, glabels = coco_train.get_next()

    # Create network
    logits, pboxes = retinanet(imgs, params['weight_decay'],
                               ckpt_file=params['resnet_checkpoint'],
                               num_classes=params['num_classes'],
                               is_training=is_training,
                               num_anchors=len(params['aspect_ratios']) * params['num_scales'])

    # Compute loss
    # cls_loss and box_loss are for logging. only total_loss is optimized.
    total_loss, cls_loss, box_loss = retinanet_model.detection_loss(logits, pboxes,
                                                                    glabels, params)
    tf.losses.add_loss(total_loss)

    # Get loss
    loss  = tf.losses.get_total_loss()

    # Create optimizer
    tf.train.create_global_step()
    global_step = tf.train.get_global_step()
    lr = retinanet_model.learning_rate_schedule(params['learning_rate'], params['lr_warmup_init'],
      params['lr_warmup_step'], params['lr_drop_step'], global_step)
    optimizer = tf.train.MomentumOptimizer(lr, params['momentum'])

    # Select trainable variables.
    vars_train = tf.trainable_variables(scope='retinanet')
    vars_train += tf.trainable_variables(scope='resnet_fpn')

    # Create train operation.
    train_op = slim.learning.create_train_op(total_loss=loss,
                                             optimizer=optimizer,
                                             global_step=global_step,
                                             variables_to_train=vars_train)
    # Learn using GPU
    max_step = FLAGS.num_epochs * FLAGS.num_examples_per_epoch
    slim.learning.train(train_op=train_op,
                        logdir=FLAGS.model_dir,
                        log_every_n_steps=FLAGS.iterations_per_loop,
                        global_step=global_step,
                        number_of_steps=1,
                        init_feed_dict={lr: params['learning_rate']})

  elif FLAGS.mode == 'eval':
    # eval only runs on CPU or GPU host with batch_size = 1

    # Override the default options: disable randomization in the input pipeline
    # and don't run on the TPU.
    eval_params = dict(
        params,
        input_rand_hflip=False,
        skip_crowd=False,
        resnet_checkpoint=None,
        is_training_bn=False,
    )

    eval_estimator = estimator.Estimator(
        model_fn=retinanet_model.retinanet_model_fn,
        config=run_conf,
        params=eval_params)

    def terminate_eval():
      tf.logging.info('Terminating eval after %d seconds of no checkpoints' %
                      FLAGS.eval_timeout)
      return True

    # Run evaluation when there's a new checkpoint
    for ckpt in evaluation.checkpoints_iterator(
        FLAGS.model_dir,
        min_interval_secs=FLAGS.min_eval_interval,
        timeout=FLAGS.eval_timeout,
        timeout_fn=terminate_eval):

      tf.logging.info('Starting to evaluate.')
      try:
        eval_results = eval_estimator.evaluate(
            input_fn=dataloader.InputReader(FLAGS.validation_file_pattern,
                                            is_training=False),
            steps=FLAGS.eval_steps)
        tf.logging.info('Eval results: %s' % eval_results)

        # Terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(ckpt).split('-')[1])
        total_step = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) /
                         FLAGS.train_batch_size)

        if current_step >= total_step:
          tf.logging.info('Evaluation finished after training step %d' %
                          current_step)
          break

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        tf.logging.info('Checkpoint %s no longer exists, skipping checkpoint' %
                        ckpt)
  else:
    tf.logging.info('Mode not found.')

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)

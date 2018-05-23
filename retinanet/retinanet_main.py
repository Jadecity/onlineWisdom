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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import tensorflow as tf

from retinanet import dataloader
from retinanet import retinanet_model
from tensorflow.python.estimator import run_config
from tensorflow.python.estimator import estimator
from tensorflow.contrib.training.python.training import evaluation

flags.DEFINE_bool(
    'use_xla', False,
    'Use XLA even if use_tpu is false.  If use_tpu is true, we always use XLA, '
    'and this flag has no effect.')
flags.DEFINE_string('model_dir', None, 'Location of model_dir')
flags.DEFINE_string('resnet_checkpoint', '',
                    'Location of the ResNet50 checkpoint to use for model '
                    'initialization.')
flags.DEFINE_string('hparams', '',
                    'Comma separated k=v pairs of hyperparameters.')
flags.DEFINE_integer('train_batch_size', 64, 'training batch size')
flags.DEFINE_integer('eval_steps', 5000, 'evaluation steps')
flags.DEFINE_integer(
    'iterations_per_loop', 100, 'Number of iterations per TPU training loop')
flags.DEFINE_string(
    'training_file_pattern', None,
    'Glob for training data files (e.g., COCO train - minival set)')
flags.DEFINE_string(
    'validation_file_pattern', None,
    'Glob for evaluation tfrecords (e.g., COCO val2017 set)')
flags.DEFINE_string(
    'val_json_file',
    '',
    'COCO validation JSON containing golden bounding boxes.')
flags.DEFINE_integer('num_examples_per_epoch', 120000,
                     'Number of examples in one epoch')
flags.DEFINE_integer('num_epochs', 15, 'Number of epochs for training')
flags.DEFINE_string('mode', 'train',
                    'Mode to run: train or eval (default: train)')
flags.DEFINE_bool('eval_after_training', False, 'Run one eval after the '
                  'training finishes.')

# For Eval mode
flags.DEFINE_integer('min_eval_interval', 180,
                     'Minimum seconds between evaluations.')
flags.DEFINE_integer(
    'eval_timeout', None,
    'Maximum seconds between checkpoints before evaluation terminates.')


FLAGS = flags.FLAGS


def main(_):

  if FLAGS.mode is 'train' and FLAGS.training_file_pattern is None:
    raise RuntimeError('You must specify --training_file_pattern for training.')

  if FLAGS.mode is 'eval':
    if FLAGS.valid_data_dir is None:
      raise RuntimeError('You must specify --valid_data_dir for evaluation.')

    if FLAGS.val_json_file is None:
      raise RuntimeError('You must specify --val_json_file for evaluation.')

  # Parse hparams
  hparams = retinanet_model.default_hparams()
  hparams.parse(FLAGS.hparams)

  params = dict(
      hparams.values(),
      resnet_checkpoint=FLAGS.resnet_checkpoint,
      val_json_file=FLAGS.val_json_file,
      mode=FLAGS.mode,
  )

  config_proto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  if FLAGS.use_xla:
    config_proto.graph_options.optimizer_options.global_jit_level = (
        tf.OptimizerOptions.ON_1)

  run_conf = run_config.RunConfig(
      model_dir=FLAGS.model_dir,
      log_step_count_steps=FLAGS.iterations_per_loop,
      session_config=config_proto)

  # GPU Estimator
  if FLAGS.mode == 'train':
    train_estimator = estimator.Estimator(
        model_fn=retinanet_model.retinanet_model_fn,
        config=run_conf,
        params=params)

    train_estimator.train(
        input_fn=dataloader.InputReader(FLAGS.training_file_pattern, is_training=True),
        max_steps=int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) / FLAGS.train_batch_size))

    if FLAGS.eval_after_training:
      # Run evaluation after training finishes.
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

      eval_results = eval_estimator.evaluate(
          input_fn=dataloader.InputReader(FLAGS.validation_file_pattern, is_training=False),
          steps=FLAGS.eval_steps)

      tf.logging.info('Eval results: %s' % eval_results)

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

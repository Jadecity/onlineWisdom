from myretinanet.utils import tf_example_decoder
import tensorflow as tf
from myretinanet.utils import preprocessor
from myretinanet.utils import anchors

def _normalize_image(image):
  """Normalize the image to zero mean and unit variance."""
  offset = tf.constant([0.485, 0.456, 0.406])
  offset = tf.expand_dims(offset, axis=0)
  offset = tf.expand_dims(offset, axis=0)
  image -= offset

  scale = tf.constant([0.229, 0.224, 0.225])
  scale = tf.expand_dims(scale, axis=0)
  scale = tf.expand_dims(scale, axis=0)
  image /= scale
  return image


class CoCoDataset():
  def __init__(self, record_path, is_training, batch_size, params, skip_crowd=True):
    self._skip_crowd = skip_crowd
    self._batch_size = batch_size
    self._params = params
    input_anchors = anchors.Anchors(params.min_level, params.max_level,
                                    params.num_scales,
                                    params.aspect_ratios,
                                    params.anchor_scale,
                                    params.image_size)
    self._anchor_labeler = anchors.AnchorLabeler(input_anchors, params.num_classes)

    self._dataset = self._build_dataset(record_path, is_training, batch_size)
    self._itr = self._dataset.make_one_shot_iterator()

  def get_next(self, min_level=3, max_level=7):
    (images, cls_targets, box_targets, num_positives, source_ids, image_scales) = self._itr.get_next()

    # Post process.
    # count num_positives in a batch
    labels = {}
    num_positives_batch = tf.reduce_mean(num_positives)
    labels['mean_num_positives'] = tf.reshape(
      tf.tile(tf.expand_dims(num_positives_batch, 0), [self._batch_size]), [self._batch_size, 1])

    for level in range(min_level, max_level + 1):
      labels['cls_targets_%d' % level] = cls_targets[level]
      labels['box_targets_%d' % level] = box_targets[level]

    labels['source_ids'] = source_ids
    labels['image_scales'] = image_scales
    return images, labels

  def _parse_func(self, value):
    example_decoder = tf_example_decoder.TfExampleDecoder()

    data = example_decoder.decode(value)


    image = data['image']
    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']
    classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])

    # Handle crowd annotations. As crowd annotations are not large
    # instances, the model ignores them in training.
    if self._skip_crowd:
      indices = tf.where(tf.logical_not(data['groundtruth_is_crowd']))
      classes = tf.gather_nd(classes, indices)
      boxes = tf.gather_nd(boxes, indices)

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Preprocessing
    image_original_shape = tf.shape(image)
    image, boxes = self._preprocess(image, boxes,
                                    self._params.image_size,
                                    self._params.input_rand_hflip)

    # Compute output
    image_scale = tf.to_float(image_original_shape[0]) / tf.to_float(tf.shape(image)[0])
    source_id = tf.string_to_number(data['source_id'], out_type=tf.float32)

    (cls_targets, box_targets, num_positives) = self._anchor_labeler.label_anchors(boxes, classes)

    return (image, cls_targets, box_targets, num_positives, source_id, image_scale)


  def _preprocess(self, img, boxes, img_size, rand_flip=False):
    image = _normalize_image(img)

    if rand_flip:
      image, boxes = preprocessor.random_horizontal_flip(image, boxes=boxes)

    image_original_shape = tf.shape(image)
    image, _ = preprocessor.resize_to_range(image,
                                            min_dimension=img_size,
                                            max_dimension=img_size)

    image, boxes = preprocessor.scale_boxes_to_pixel_coordinates(image,
                                                                 boxes,
                                                                 keypoints=None)

    image = tf.image.pad_to_bounding_box(image, 0, 0, img_size, img_size)

    return image, boxes


  def _build_dataset(self, record_path, is_training, batch_size):

    dataset = tf.data.Dataset.list_files(record_path, shuffle=False)

    dataset = dataset.shuffle(buffer_size=1024)
    if is_training:
      dataset = dataset.repeat()

    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
          lambda filename:tf.data.TFRecordDataset(filename).prefetch(1),
          cycle_length=32,
          sloppy=True))
    dataset = dataset.shuffle(20)

    dataset = dataset.map(self._parse_func, num_parallel_calls=64)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.apply(
      tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.prefetch(1)

    return dataset




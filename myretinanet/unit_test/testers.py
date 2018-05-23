import tensorflow as tf

def test_dataset_and_exit(imgs, glabels):
  with tf.Session() as sess:
    imgs_val, glabels_val = sess.run([imgs, glabels])
    batch_size = len(imgs_val)
    for i in range(batch_size):
      img = imgs_val[i]
      label = glabels_val[i]
      a = 0

  exit(0)
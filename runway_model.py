import sys
sys.path.insert(0, 'src')
import os, random, subprocess, evaluate, shutil
from utils import exists, list_files
import pdb
import numpy as np
import transform, vgg, pdb, os
import tensorflow as tf
import runway


def load_checkpoint(checkpoint, sess):
    saver = tf.train.Saver()
    try:
        saver.restore(sess, checkpoint)
        return True
    except:
        print("checkpoint %s not loaded correctly" % checkpoint)
        return False


@runway.setup(options={"checkpoint_path": runway.file(is_directory=True) })
def setup(options):
    global sess
    global img_placeholder
    global preds
    h, w = 480, 640
    img_shape = (h, w, 3)
    batch_shape = (1,) + img_shape
    g = tf.Graph()
    g.as_default()
    g.device('/gpu:0')
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    sess = tf.Session(config=soft_config)
    img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')
    preds = transform.net(img_placeholder)
    load_checkpoint(options['checkpoint_path'], sess)
    return sess


@runway.command('convert', inputs={'image': runway.image}, outputs={'output': runway.image})
def convert(sess, inp):
    img = np.array(inp['image'].resize((640, 480)))
    img = np.expand_dims(img, 0)
    output = sess.run(preds, feed_dict={img_placeholder: img})
    output = np.clip(output[0], 0, 255).astype(np.uint8)
    return dict(output=output)


if __name__ == '__main__':
    runway.run()


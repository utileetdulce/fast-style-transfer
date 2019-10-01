import sys
sys.path.insert(0, 'src')
import os, random, subprocess, evaluate, shutil
from utils import exists, list_files
import pdb
import numpy as np
import transform, vgg, pdb, os
import tensorflow as tf
from PIL import Image
import runway


def load_checkpoint(checkpoint, sess):
    saver = tf.train.Saver()
    try:
        saver.restore(sess, checkpoint)
        return True
    except:
        print("checkpoint %s not loaded correctly" % checkpoint)
        return False

g = None

@runway.setup(options={"checkpoint_path": runway.file(is_directory=True) })
def setup(options):
    global sess
    global img_placeholder
    global preds
    global g
    h, w = 480, 640
    img_shape = (h, w, 3)
    batch_shape = (1,) + img_shape
    g = tf.get_default_graph()
    sess = tf.Session(graph=g)
    img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')
    preds = transform.net(img_placeholder)
    load_checkpoint(os.path.join(options['checkpoint_path'], 'fns.ckpt'), sess)
    return sess


@runway.command('stylize', inputs={'image': runway.image}, outputs={'output': runway.image})
def stylize(sess, inp):
    img = inp['image']
    original_size = img.size
    img = np.array(img.resize((640, 480)))
    img = np.expand_dims(img, 0)
    with g.as_default():
        output = sess.run(preds, feed_dict={img_placeholder: img})
    output = np.clip(output[0], 0, 255).astype(np.uint8)
    output = Image.fromarray(output).resize(original_size)
    return dict(output=output)


if __name__ == '__main__':
    runway.run(model_options={'checkpoint_path': 'models/Cubist'})


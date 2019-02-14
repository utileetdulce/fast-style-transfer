import sys
sys.path.insert(0, 'src')
import os, random, subprocess, evaluate, shutil
from utils import exists, list_files
import pdb
import numpy as np
import transform, vgg, pdb, os
import tensorflow as tf
from runway import RunwayModel


models=[
    {"ckpt":"models/ckpt_cubist_b20_e4_cw05/fns.ckpt", "style":"styles/cubist-landscape-justineivu-geanina.jpg"},
	{"ckpt":"models/ckpt_hokusai_b20_e4_cw15/fns.ckpt", "style":"styles/hokusai.jpg"},
	{"ckpt":"models/ckpt_kandinsky_b20_e4_cw05/fns.ckpt", "style":"styles/kandinsky2.jpg"},
	{"ckpt":"models/ckpt_liechtenstein_b20_e4_cw15/fns.ckpt", "style":"styles/liechtenstein.jpg"},
	{"ckpt":"models/ckpt_wu_b20_e4_cw15/fns.ckpt", "style":"styles/wu4.jpg"},
	{"ckpt":"models/ckpt_elsalahi_b20_e4_cw05/fns.ckpt", "style":"styles/elsalahi2.jpg"},
	{"ckpt":"models/scream/scream.ckpt", "style":"styles/the_scream.jpg"},
	{"ckpt":"models/udnie/udnie.ckpt", "style":"styles/udnie.jpg"},
	{"ckpt":"models/ckpt_maps3_b5_e2_cw10_tv1_02/fns.ckpt", "style":"styles/maps3.jpg"}
]

faststyletransfer = RunwayModel()
idx_model = 0


def load_checkpoint(checkpoint, sess):
	saver = tf.train.Saver()
	try:
		saver.restore(sess, checkpoint)
		return True
	except:
		print("checkpoint %s not loaded correctly" % checkpoint)
		return False


@faststyletransfer.setup
def setup(alpha=0.5):
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
    load_checkpoint(models[idx_model]["ckpt"], sess)
    return sess


@faststyletransfer.command('convert', inputs={'image': 'image'}, outputs={'output': 'image'})
def convert(sess, inp):
    img = np.array(inp['image'])
    img = np.expand_dims(img, 0)
    output = sess.run(preds, feed_dict={img_placeholder: img})
    output = np.clip(output[0], 0, 255).astype(np.uint8)
    return dict(output=output)


if __name__ == '__main__':
    faststyletransfer.run()

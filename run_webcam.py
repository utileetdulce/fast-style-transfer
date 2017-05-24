from __future__ import print_function
import sys
sys.path.insert(0, 'src')
import argparse
import numpy as np
import transform, vgg, pdb, os
import tensorflow as tf
import cv2


models=[{"ckpt":"models/ckpt_cubist_b20_e4_cw05/fns.ckpt", "style":"styles/cubist-landscape-justineivu-geanina.jpg"},
	{"ckpt":"models/ckpt_hokusai_b20_e4_cw15/fns.ckpt", "style":"styles/hokusai.jpg"},
	{"ckpt":"models/ckpt_kandinsky_b20_e4_cw05/fns.ckpt", "style":"styles/kandinsky2.jpg"},
	{"ckpt":"models/ckpt_liechtenstein_b20_e4_cw15/fns.ckpt", "style":"styles/liechtenstein.jpg"},
	{"ckpt":"models/ckpt_maps3_b5_e2_cw10_tv1_02/fns.ckpt", "style":"styles/maps3.jpg"},
	{"ckpt":"models/ckpt_wu_b20_e4_cw15/fns.ckpt", "style":"styles/wu4.jpg"},
	{"ckpt":"models/ckpt_elsalahi_b20_e4_cw05/fns.ckpt", "style":"styles/elsalahi2.jpg"}]

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--width', type=int, help='width to resize camera feed to (default 320)', required=False, default=320)
parser.add_argument('--disp_width', type=int, help='width to display output (default 640)', required=False, default=640)
parser.add_argument('--display_source', type=int, help='whether to display content and style images next to output, default 1', required=True, default=1)


def load_checkpoint(checkpoint, sess):
	saver = tf.train.Saver()
	try:
		saver.restore(sess, checkpoint)
		style = cv2.imread(checkpoint)
		return True
	except:
		print("checkpoint %s not loaded correctly" % checkpoint)
		return False


def make_triptych(disp_width, frame, style, output):
	ch, cw, _ = frame.shape
	sh, sw, _ = style.shape
	oh, ow, _ = output.shape
	h = int(ch * disp_width * 0.5 / cw)
	full_img = np.concatenate([cv2.resize(frame, (int(0.5 * disp_width), h)), cv2.resize(style, (int(0.5 * disp_width), h))], axis=1)
	full_img = np.concatenate([full_img, cv2.resize(output, (disp_width, disp_width * oh / ow))], axis=0)
	return full_img


def main(width, disp_width, display_source):
	idx_model = 0
	device_t='/gpu:0'
	g = tf.Graph()
	soft_config = tf.ConfigProto(allow_soft_placement=True)
	soft_config.gpu_options.allow_growth = True
	with g.as_default(), g.device(device_t), tf.Session(config=soft_config) as sess:	
		
		cam = cv2.VideoCapture(0)
		cam_width, cam_height = cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH), cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

		width = width if width % 4 == 0 else width + 4 - (width % 4) # must be divisible by 4
		height = int(width * float(cam_height/cam_width))
		height = height if height % 4 == 0 else height + 4 - (height % 4) # must be divisible by 4
		img_shape = (height, width, 3)
		batch_shape = (1,) + img_shape
		print("batch shape",batch_shape)
		img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')
		preds = transform.net(img_placeholder)
		
		# load checkpoint
		load_checkpoint(models[idx_model]["ckpt"], sess)
		style = cv2.imread(models[idx_model]["style"])
		
		# enter cam loop
		while True:
			ret, frame = cam.read()
			frame = cv2.resize(frame, (width, height))
			#frame = cv2.flip(frame, 0)
			X = np.zeros(batch_shape, dtype=np.float32)
			X[0] = frame
			
			output = sess.run(preds, feed_dict={img_placeholder:X})
			output = output[:, :, :, [2,1,0]].reshape(img_shape)
			output = np.clip(output, 0, 255).astype(np.uint8)
			output = cv2.resize(output, (width, height))
			print("disp source is ",display_source)
			if display_source:
				full_img = make_triptych(disp_width, frame, style, output)
				cv2.imshow('frame', full_img)
			else:
				oh, ow, _ = output.shape
				output = cv2.resize(output, (disp_width, int(oh * disp_width / ow)))
				cv2.imshow('frame', output)

			key_ = cv2.waitKey(1)	
			if key_ == 27:
				break
			elif key_ == ord('a'):
				idx_model = (idx_model +len(models) - 1) % len(models)
				print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
				load_checkpoint(models[idx_model]["ckpt"], sess)
				style = cv2.imread(models[idx_model]["style"])
			elif key_ == ord('s'):
				idx_model = (idx_model + 1) % len(models)
				print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
				load_checkpoint(models[idx_model]["ckpt"], sess)
				style = cv2.imread(models[idx_model]["style"])
		
		# done
		cam.release()
		cv2.destroyAllWindows()


if __name__ == '__main__':
	opts = parser.parse_args()
	main(opts.width, opts.disp_width, opts.display_source==1)


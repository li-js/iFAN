import numpy as np
from PIL import Image

def encode_pts(pts, h0, w0, radius=1, add_bg_map=False):
	nb_pts=pts.shape[1]/2
	batch_size=pts.shape[0]
	pts=(pts.reshape(batch_size, 2, nb_pts)+0.5)*h0
	radius2=radius**2
	if add_bg_map:
		map_all=np.zeros((batch_size, h0,w0, nb_pts+1), dtype=np.float32)
	else:
		map_all=np.zeros((batch_size, h0,w0, nb_pts), dtype=np.float32)
	for im_i in xrange(batch_size):
		map_i=np.zeros((h0,w0, nb_pts), dtype=np.float32)
		for pt_k in xrange(nb_pts):
			pt=np.round(pts[im_i,:,pt_k]).astype(np.int)
			for kx in range(pt[0]-radius, pt[0]+radius):
				for ky in range(pt[1]-radius, pt[1]+radius):
					if kx>=0 and ky >=0 and kx < w0 and ky < h0 and np.square(kx-pt[0])+np.square(ky-pt[1])<radius2:
						map_i[ky, kx, pt_k]=1
		if add_bg_map: map_i=np.dstack((map_i.sum(axis=2)==0, map_i))
		map_all[im_i]=map_i
	return map_all
	

def encode_pose(pose, h0, w0):
	sh=pose.shape
	if len(sh)==1:
		assert(sh[0]==5)
		return np.tile(pose, [h0,w0,1])
	elif len(sh)==2:
		assert(sh[1]==5)
		return np.tile(pose, [h0,w0,1,1]).transpose((2,0,1,3))

def encode_emo(emo, h0, w0):
	sh=emo.shape
	if len(sh)==1:
		assert(sh[0]==7)
		return np.tile(emo, [h0,w0,1])
	elif len(sh)==2:
		assert(sh[1]==7)
		return np.tile(emo, [h0,w0,1,1]).transpose((2,0,1,3))


def encode_seg(seg, h0, w0):
	if seg.shape[1:3]==(h0,w0):
		return seg
	else:
		seg2=np.zeros((seg.shape[0], h0,w0, seg.shape[3]))
		for im_i in xrange(seg.shape[0]):
			for c in xrange(seg.shape[3]):
				seg2[im_i, :,:, c]=np.array(Image.fromarray(seg[im_i,:,:,c]).resize((w0,h0)))
		return seg2
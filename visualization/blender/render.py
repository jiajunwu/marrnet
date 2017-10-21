import os
import sys
import numpy as np
import time
import bpy
import socket
from mathutils import Vector
import util_voxel
import util_isosurface
# [0, 200, 0], [240, 0, 0], [150, -150, 75], [0, 240, 50]
def render_file(ifilename, ofile_path, ofile_prefix, ofile_ext, views=[[240, 0, 0], [150, -150, 75], [0, 240, 50]], resolution=[512, 512], pooling_step=1, method='iso', color=(0.75, 0.75, 0.75), iname='voxels', overwrite=True):
	tempfilename = ifilename[:-4] + '.npy'
	os.system(' '.join(('python', 'convert.py', ifilename, iname)))
	mat = np.load(tempfilename)
	os.system('rm ' + tempfilename)
	if not os.path.isdir(ofile_path):
		os.system('mkdir -p ' + ofile_path)
	for i in range(mat.shape[0]):
		for j, cam_dis in enumerate(views):
			ofilename = os.path.join(ofile_path, '%s_%01d_view_%d.%s' % (ofile_prefix, i, j + 1, ofile_ext))
			if not overwrite and os.path.isfile(ofilename):
				print('==> file ' + ofilename + ' skipped')
			else:
				render_voxel(mat[i][0], method=method, filename=ofilename, resolution=resolution, pooling_step=pooling_step, color=color, cam_dis=cam_dis)
				print('==> file '+ ofilename +' rendered')
	return mat.shape[0]


def render_voxel(mat, filename=None, resolution=[512, 512], pooling_step=1, method='iso', color=(0.75, 0.75, 0.75), cam_dis=(240, 0, 0)):
	''' Entrance of voxel rendering functions. render the object in the scene. '''
	assert len(mat.shape) == 3, 'Input must be a 3D matrix'
	assert mat.min() >= 0, 'Matrix min must >= 0'
	assert mat.max() <= 1, 'Matrix max must <= 1'

	scn = bpy.context.scene;

	# Clear
	bpy.ops.object.select_all(action='SELECT');
	bpy.ops.object.delete();

	if pooling_step > 1:
		mat = util_voxel.pooling(mat, pooling_step, 'max')

	material = bpy.data.materials.new('Color-Material')
	material.diffuse_color = color
	material.specular_color = (1, 1, 1)

	if method == 'iso':
		p0, p1, threshold = render_iso(mat, material=material)
	elif method == 'cube':
		p0, p1, threshold = render_cube(mat, material=material)
	else:
		error('Unknown rendering method. Only iso and cube are supported')

	# set camera view and lightling
	bpy.ops.object.camera_add()
	scn.camera = bpy.data.objects['Camera']
	scn.camera.data.type = 'ORTHO'
	scn.camera.data.ortho_scale = 120

	# Constrain camera to object
	b0,b1,bc = find_bound(mat, threshold)   # bound corners and bound center
	if bc is None:
		b0 = np.zeros(3)
		b1 = np.array(mat.shape)
		bc = (b0+b1)/2
	bpy.ops.object.constraint_add(type="TRACK_TO")
	empty = bpy.ops.object.empty_add(type='CUBE', radius=0.1, location=bc.tolist())
	scn.camera.constraints["Track To"].target = bpy.data.objects['Empty']
	scn.camera.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
	scn.camera.constraints["Track To"].up_axis = 'UP_Y'

	cam_factor = 0.6
	scn.camera.location = Vector(bc.tolist()) + Vector(cam_dis) * cam_factor
	# scn.camera.location = Vector(cam_dis) * cam_factor
	scn.camera.data.clip_end = max(p1)*10   # camera clipping range

	# Set camera light
	bpy.ops.object.lamp_add(type='POINT')
	camlamp = bpy.data.objects['Point']
	camlamp.location = scn.camera.location
	camlamp.data.use_specular = False
	camlamp.data.energy = 1000 / (pooling_step ** 2) / ((1.7 / cam_factor) ** 2)
	camlamp.data.distance = 100
	bpy.ops.object.constraint_add(type="COPY_TRANSFORMS")
	camlamp.constraints['Copy Transforms'].target = scn.camera

	# Set 6 background light
	p0_ = np.array(p0)
	p1_ = np.array(p1)
	sun_dis = (p1_-p0_).max()
	sun_locs = np.zeros((6, 3))
	sun_locs[:3, :] = np.dot(((p1_ + p0_) / 2).reshape(3, 1), np.ones((1, 3))) + np.eye(3) * sun_dis
	sun_locs[3:, :] = np.dot(((p1_ + p0_) / 2).reshape(3, 1), np.ones((1, 3))) - np.eye(3) * sun_dis
	sun_locs = sun_locs.tolist()
	for i in range(6):
		bpy.ops.object.lamp_add(type='POINT')
	sun_counter = 0
	for lamp in [obj for obj in bpy.data.objects if obj.name[:5] == 'Point']:
		if lamp == camlamp:
			continue
		lamp.location = Vector(sun_locs[sun_counter])
		lamp.data.use_specular = False
		lamp.data.energy = 40
		sun_counter += 1

	# Render
	if filename != None:
		bpy.data.scenes[0].render.resolution_x = resolution[0]
		bpy.data.scenes[0].render.resolution_y = resolution[1]
		bpy.data.scenes[0].render.alpha_mode = 'TRANSPARENT'
		bpy.data.scenes[0].render.image_settings.color_mode = 'RGBA'
		bpy.data.scenes[0].render.image_settings.file_format = filename.split('.')[-1].upper()
		bpy.data.scenes[0].render.filepath = filename
		print(bpy.data.scenes[0].render.filepath)
		bpy.ops.render.render(write_still=True)


def render_iso(mat, threshold=0.1, padding=4, material=None):
	if padding > 0:
		mat_ = np.zeros((mat.shape[0] + padding * 2, mat.shape[1] + padding * 2, mat.shape[2] + padding * 2))
		mat_[padding:-padding, padding:-padding, padding:-padding] = mat
		mat = mat_

	resolution = mat.shape
	p0 = [0, 0, 0]
	p1 = [x - 1 for x in resolution]
	def scalarfield(pos):
		x, y, z = pos[0], pos[1], pos[2]
		return mat[int(x), int(y), int(z)]

	start = time.time()
	block = util_isosurface.isosurface(p0, p1, resolution, threshold, scalarfield)

	if material != None:
		block.active_material = material

	elapsed = time.time() - start
	print("%r seconds" % elapsed)
	return p0, p1, threshold

def render_cube(mat, threshold=0.1, material=None):
	for i in range(mat.shape[0]):
		for j in range(mat.shape[1]):
			for k in range(mat.shape[2]):
				prob = mat[i][j][k]
				if prob > threshold:
					bpy.ops.mesh.primitive_cube_add(radius=0.5 * prob,location=(i + 0.5, j + 0.5, k + 0.5))
	if material != None:
		for obj in bpy.data.objects:
			if obj.name[:4] == 'Cube':
				obj.active_material = material
	return (0, 0, 0), mat.shape, threshold

def find_bound(mat, threshold):
	voxel_used = (mat >= threshold).astype(float);
	p0 = np.zeros(3)
	p1 = np.zeros(3)
	range_x = voxel_used.sum(1).sum(1).nonzero()[0]
	range_y = voxel_used.sum(0).sum(1).nonzero()[0]
	range_z = voxel_used.sum(0).sum(0).nonzero()[0]
	if range_x.size == 0:	   # no value above threshold
		return None, None, None
	p0[0] = range_x.min()
	p0[1] = range_y.min()
	p0[2] = range_z.min()
	p1[0] = range_x.max()
	p1[1] = range_y.max()
	p1[2] = range_z.max()
	return p0, p1, (p0 + p1) / 2

assert len(sys.argv) >= 7
file_path = sys.argv[5]
output_dir = sys.argv[6]
output_prefix = 'im'
if len(sys.argv) >= 8: 
	output_prefix = sys.argv[7]

render_file(file_path, output_dir, output_prefix, 'png')


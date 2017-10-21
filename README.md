# MarrNet: 3D Shape Reconstruction via 2.5D Sketches

This repository contains pre-trained models and testing code for MarrNet presented at NIPS 2017.

http://marrnet.csail.mit.edu

<table>
<tr>
<td><img src="http://marrnet.csail.mit.edu/repo/input/chair_1.png" width="210"></td>
<td><img src="http://marrnet.csail.mit.edu/repo/output/chair_1.png" width="210"></td>
<td><img src="http://marrnet.csail.mit.edu/repo/input/chair_2.png" width="210"></td>
<td><img src="http://marrnet.csail.mit.edu/repo/output/chair_2.png" width="210"></td>
<td><img src="http://marrnet.csail.mit.edu/repo/input/chair_3.png" width="210"></td>
<td><img src="http://marrnet.csail.mit.edu/repo/output/chair_3.png" width="210"></td>
</tr>
</table>

## Prerequisites
#### Torch
We use Torch 7 (http://torch.ch) for our implementation with these additional packages:

- [`cutorch`](https://github.com/torch/cutorch) , [`cudnn`](https://github.com/soumith/cudnn.torch): we use cutorch and cudnn for our model.
- [`fb.mattorch`](https://github.com/facebook/fblualib/tree/master/fblualib/mattorch): we use `.mat` file for saving voxelized shapes.

#### Visualization
- Basic visualization: MATLAB (tested on R2016b)
- Advanced visualization: Blender with bundled Python and packages `numpy`, `scipy`, `mathutils`, `itertools`, `bpy`, `bmesh`.

## Installation
Our current release has been tested on Ubuntu 14.04.

#### Cloning the repository
```sh
git clone git@github.com:jiajunwu/marrnet.git
```
#### Downloading pretrained models (920M) 
```sh
cd marrnet
./download_models.sh
```

## Guide
#### 3D shape reconstruction (`main.lua`)
We show how to reconstruct shapes from 2D images by using our pre-trained models. The file (`main.lua`) has the following options.
- `-imgname`:The name of test image, which should be stored in `image` folder.

Usage
```sh
th main.lua -imgname chair_1.png
```
The output is saved under folder `./output`, named `%imgname.mat`, where `%imgname` is the name of the image without the filename extension.

Note that the input image will be automatically resized to 256 x 256. The model works the best if the chair is centered and fills up about half of the whole image (like `image/chair_1.png`).

#### Visualization
We offer two ways of visualizing results, one in MATLAB and the other in Blender. We used the Blender visualization in our paper. The MATLAB visualization is easier to install and run, but it is also slower, and its output has a lower quality compared with Blender.

**MATLAB**:
Please use the function `visualization/matlab/visualize.m` for visualization. The MATLAB code allows users to either display rendered objects or save them as images. The script also supports downsampling and thresholding for faster rendering. The color of voxels represents the confidence value. 

Options include
- `inputfile`: the .mat file that saves the voxel matrices
- `indices`: the indices of objects in the inputfile that should be rendered. The default value is 0, which stands for rendering all objects.
- `step (s)`: downsample objects via a max pooling of step s for efficiency. The default value is 4 (128 x 128 x 128 -> 32 x 32 x 32).
- `threshold`: voxels with confidence lower than the threshold are not displayed
- `outputprefix`: 
    - when not specified, Matlab shows figures directly.
    - when specified, Matlab stores rendered images of objects at `../result/outputprefix_%i_front.bmp` and `../result/outputprefix_%i_side.bmp`, where `%i` is the index of objects. `front` view should be the same view with the input image, and `side` view would be a more understandable view for us.

Usage (after running `th main.lua -imgname chair_1.png`, in MATLAB, in folder `visualization/matlab`):

```matlab
visualize('../../output/chair_1.mat', 0, 2, 0.03, 'chair_1')
```

The code will visualize the shape in 2 different views. The visualization might take a while. Please change the `step` parameter if it is too slow.

**Blender**:
Options for the Blender visualization include

- `file_path`: the path to the `.mat` file.
- `output_dir`: the path of the output directory. 
- `outputprefix`: Blender stores rendered images of objects at `outputprefix_%i_view_%j.png`, where `%i` is the index of objects, `%j` is the index of views. When not specified, Blender will use 'im' by default.

Usage (after running `th main.lua -imgname chair_1.png`):
```sh
cd visualization/blender
blender --background --python render.py -- ../../output/chair_1.mat ../result/ chair_1
```

Blender will render 3 different views, where the first view should be the same as the input image.

## Reference

    @inproceedings{marrnet,
      title={{MarrNet: 3D Shape Reconstruction via 2.5D Sketches}},
      author={Wu, Jiajun and Wang, Yifan and Xue, Tianfan and Sun, Xingyuan and Freeman, William T and Tenenbaum, Joshua B},
      booktitle={Advances In Neural Information Processing Systems},
      year={2017}
    }

For any questions, please contact Jiajun Wu (jiajunwu@mit.edu) and Yifan Wang (wangyifan1995@gmail.com).

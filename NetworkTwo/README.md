# Signal2PC

Use UWB signal to reconstruct 3D point cloud

## Get Started

### Environment

It works well under dependencies as follows:

- Ubuntu 22.04
- Python 3.8
- PyTorch 2.0.0
- CUDA 11.8
- OpenCV 4.9
- Scipy 1.9.1
- Scikit-Image 0.20.0
- open3d 0.18.0
- cupy 12.3.0

Some other dependencies :

> easydict, pyyaml, tensorboardx, trimesh, shapely

Three external modules need to be compiled manually:

`python setup.py install` in directory [external/chamfer](external/chamfer), [external/neural_renderer](external/neural_renderer) and  [external/PyTorchEMD](external/PyTorchEMD) to compile the modules.


### Usage

#### Configuration

You can modify configuration in a `yml` file for training/evaluation. It overrides dsefault settings in `options.py`. We provide some examples in the `experiments` directory. 

#### Training

```
python entrypoint_train.py --name xxx --options path/to/yaml
```
#### Evaluation

```shell
python entrypoint_eval.py --name xxx --options path/to/yml --checkpoint path/to/checkpoint
```

#### 深度图渲染

1. /render/render_fixed.sh
2. /render/convertEXR.sh
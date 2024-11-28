# Fast Object Compositional Neural Radiance Fields (FOC-NeRF)
This repository contains:
* A pytorch implementation of Object-Level reconstruction with scene editting capability using a modified form of NeRF
neural network inspired by Instant Neural Graphics Primitives[instant-ngp](https://github.com/NVlabs/instant-ngp).



FOC-NeRF editting capability on Table-top dataset:

https://github.com/user-attachments/assets/45fbcb0a-0414-433c-8f23-d8cf8b64b919




### Other related projects

* [Object-NeRF](https://github.com/zju3dv/object_nerf): An Object-NeRF benchmark.

# Create environment
```bash
conda create -y --name foc_nerf python=3.10
conda activate foc_nerf
```

# Install
```bash
git clone --recursive https://github.com/ashawkey/torch-ngp.git
cd FOC-NeRF
```

### Install with pip
```bash
pip install -r requirements.txt

# (optional) install the tcnn backbone
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```


### Additional pip installation
```bash
pip install -r requirements.txt

# (optional) install the tcnn backbone
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```


### Build extension (optional)
By default, we use [`load`](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load) to build the extension at runtime.
However, this may be inconvenient sometimes.
Therefore, we also provide the `setup.py` to build each extension:
```bash
# install all extension modules
bash scripts/install_ext.sh

# if you want to install manually, here is an example:
cd raymarching
python setup.py build_ext --inplace # build ext only, do not install (only can be used in the parent directory)
pip install . # install to python path (you still need the raymarching/ folder, since this only install the built extension.)
```

### Tested environments
* Ubuntu 22 with torch 2.4.1 & CUDA 12.4 on a RTX 3090 RTX.



# Usage

We use the same data format as instant-ngp, [fox](https://github.com/NVlabs/instant-ngp/tree/master/data/nerf/fox). 


<details>
  <summary> Supported datasets </summary>

  * [Table-top](https://drive.google.com/drive/u/1/folders/11G4Jg7iP85TAJ8SCxPZMhhSFFwjEByys) 


  * [Cube Diorama Dataset](https://github.com/jc211/nerf-cube-diorama-dataset)

</details>

First time running will take some time to compile the CUDA extensions.

```bash
### FOC NeRF 
# training FOC-NERF

python3 main_nerf.py /path/to/dataset --workspace /path/to/workspace --fp16 --tcnn  --iters 30000 --yolo_model /path/to/yoloseg/checkpoint.pt


### Combining Objects to a single scene
python3 COMBINED.py   /path/to/dataset  --workspace /path/to/workspace    --objects_of_interest book cup  --ckpt_dir /path/to/checkpoint


###Editting the scene 
python3 editable.py   /path/to/datset  --workspace /path/to/workspace  --ckpt_dir /path/to/trained_checkpoints  --objects_of_interest book cup  --edit-object book  --offset_x 0.01 --offset_y 0.01 --offset_z 0.60 








# Tips

**Q**: How to install the environment and dependences

**A**: Follow the torch-ngp repository installation and install this additional repository

**Q**: CUDA Out Of Memory for my dataset.

**A**: You could try to reduce the number of object to be combined. Another solution is to manually set `downscale` in `NeRFDataset` to lower the image resolution.

# Citation

If you find this work useful, a citation will be appreciated via:
```
@misc{foc-nerf,
    Author = {Jeffrey Eiyike, Elvis Gyaase, Vikas Dhiman},
    Year = {2024},
    Note = {https://github.com/ashawkey/torch-ngp},
    Title = {FOC-NeRF:Fast Object Compositional Neural Radiance Fields}
}

```

# Acknowledgement

* Credits to [Thomas Müller](https://tom94.net/) for the amazing [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) and [instant-ngp](https://github.com/NVlabs/instant-ngp):
    ```
    @misc{tiny-cuda-nn,
        Author = {Thomas M\"uller},
        Year = {2021},
        Note = {https://github.com/nvlabs/tiny-cuda-nn},
        Title = {Tiny {CUDA} Neural Network Framework}
    }

    @article{mueller2022instant,
        title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
        author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
        journal = {arXiv:2201.05989},
        year = {2022},
        month = jan
    }

    @misc{torch-ngp,
    Author = {Jiaxiang Tang},
    Year = {2022},
    Note = {https://github.com/ashawkey/torch-ngp},
    Title = {Torch-ngp: a PyTorch implementation of instant-ngp}
}
    ```

* The framework of NeRF is adapted from [nerf_pl](https://github.com/ashawkey/torch-ngp):
    ```
    @misc{torch-ngp,
    Author = {Jiaxiang Tang},
    Year = {2022},
    Note = {https://github.com/ashawkey/torch-ngp},
    Title = {Torch-ngp: a PyTorch implementation of instant-ngp}
}
    ```


### Training Machine 
 The training computer is used to run this repository, and is a powerful workstation with NVIDIA RTX 3090, and 24 GB of DDR4 RAM. More GPU is required for combining more than 4 objects back into a single scene


### License

### Acknowledgements
- Special thanks to the contributors [Torch-ngp](https://github.com/ashawkey/torch-ngp) 
- [Ultralytics](https://docs.ultralytics.com/models/yolov8/)
 for their invaluable tools and resources.





   

      
   

   





# FOCNeRF

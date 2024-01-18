# Perceptual Learned Video Compression with Recurrent Conditional GAN

The project page for the paper:

> Ren Yang, Radu Timofte and Luc Van Gool, "Perceptual Learned Video Compression with Recurrent Conditional GAN", in Processings of the International Joint Conference on Artificial Intelligence (IJCAI), 2022. [[Paper]](https://arxiv.org/abs/2109.03082)

If our paper is useful for your research, please cite:
```
@inproceedings{yang2022perceptual,
  title={Perceptual Video Compression with Recurrent Conditional GAN},
  author={Yang, Ren and Timofte, Radu and Van Gool, Luc},
  booktitle={Processings of the International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2022}
}
```

If you have questions, please contact:

Ren Yang @ ETH Zurich, Switzerland   

Email: r.yangchn@gmail.com

#### Acknowledgement

Our discriminator, included in the folder "hific", is built upon [HiFiC](https://hific.github.io/).

## Codes

### Training data

- Download the training data. We train the models on the [Vimeo90k dataset](https://github.com/anchen1011/toflow) ([Download link](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip)) (82G). After downloading, please run the following codes to generate "folder.npy" which contains the directories of all training samples.
```
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(root)
    return result

folder = find('im1.png', 'path_to_vimeo90k/vimeo_septuplet/sequences/')
np.save('folder.npy', folder)
```

- Compress I-frames. In PLVC, we compress the I-frames by HiFiC (https://hific.github.io/). Please first encode the first frame in each training clip, i.e., im1.png, by HiFiC, and name the compressed images as im1_lo.png, im1_mi.png and im1_hi.png for the low-, medium- and high-quality configurations of HiFiC, respectively.

### Dependency

- Tensorflow 1.12
  
  (*Since we train the models on tensorflow-compression 1.0, which is only compatibable with tf 1.12, the pre-trained models are not compatible with higher versions.*)

- Tensorflow-compression 1.0 ([Download link](https://github.com/tensorflow/compression/releases/tag/v1.0))

  (*After downloading, put the folder "tensorflow_compression" to the same directory as the codes.*)
  
- SciPy 1.2.0

  (*Since we use misc.imread, do not use higher versions in which misc.imread is removed.*)
  
 - Pre-trained models by MSE loss ([Download link](https://data.vision.ee.ethz.ch/reyang/model.zip))
 
      (*Download the folder "model" to the same directory as the codes.*)
  
 - compare-gan 3.0
 
 - lpips-tensorflow (https://github.com/alexlee-gk/lpips-tensorflow)
 
### Training
 
```
python PLVC_training.py --q mi
```

The PLVC has three quality configurations, i.e., "lo", "mi" and "hi" for low-, medium- and high-quality models, respectively.

### Test

Pre-trained models: [https://data.vision.ee.ethz.ch/reyang/PLVC_model.zip](https://drive.google.com/file/d/1ctVgz7uZh3Oz9uV99cc1cQeDZPOyVeQr/view?usp=sharing)

Test code: 

The backbone (generator) of PLVC has the same architecture as [RLVC](https://github.com/RenYang-home/RLVC/). Therefore the pre-trained PLVC models can be directly loaded and tested by the inference code of RLVC, with the following adjustments:

1. In PLVC, we use bi-IPPP with GOP = 9, hence set both "--f_p" and "--b_p" to 4 in [RLVC.py](https://github.com/RenYang-home/RLVC/blob/master/RLVC.py#L8).
2. The PLVC models are fine-tuned from the PSNR-optimized RLVC models, and the "lo", "mi" and "hi" quality models corresponds to lambda = 256, 512 and 1024, respectively. Therefore, set "--l" in [RLVC.py](https://github.com/RenYang-home/RLVC/blob/master/RLVC.py#L14) to 256, 512 or 1024 according to the quality setting of PLVC.
3. Compress the I-frames (every 9 frames) by HiFiC at the corresponding quality ("lo", "mi" and "hi" models of HiFiC corresponds to the "lo", "mi" and "hi" models of PLVC, respectively). Save the compressed frames (e.g., f001.png, f010.png, f019.png, etc.) to "path_com" defined [here](https://github.com/RenYang-home/RLVC/blob/master/helper.py#L29), and the compressed bitstream (e.g., f001.bin, f010.bin, f019.bin, etc.) to "path_bin" defined [here](to "path_com" defined [here](https://github.com/RenYang-home/RLVC/blob/master/helper.py#L29)).
4. Change the [model path](https://github.com/RenYang-home/RLVC/blob/master/Recurrent_AutoEncoder.py#L114) to the downloaded PLVC models.
5. Comment (or delete) the codes that compress I-frames at [line 77 to 90 in helper.py](https://github.com/RenYang-home/RLVC/blob/master/helper.py#L77).
6. Then, run [RLVC.py](https://github.com/RenYang-home/RLVC/blob/master/RLVC.py) to test PLVC.

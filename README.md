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

Email: ren.yang@vision.ee.ethz.ch

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

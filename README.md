HART on PyTorch
===

A minimal (?) reproduction of [HART](https://github.com/akosiorek/hart).

### Usage

Download KTH Dataset first from [here](https://drive.google.com/a/nyu.edu/file/d/1KBQFWWaUg1ePPX2EXietBtGL2kOf-QIU/view?usp=sharing)
and unpack it to the repo directory.
You will have `KTHBoundingBoxInfo.txt` and a directory called `frames` in the
same place as the source code files.

I split the dataset into 1793, 300, 294 sequences for training, validation and test respectively.

Depending on whether you are enabling L2 regularization, you may need to apply a patch (see [below](#norm-patch)),
unless you are running a bleeding-edge version or release version later than 0.2.0-post3.
If you do, make sure that the patching succeeded.

You also need to run a patch for `visdom` (see [below](#visdom-patch))

Finally, run `main.py` with Python 3.

Setting environment variable `MPL_BACKEND=Agg` is recommended unless you plan to interactivvely display some
matplotlib figure.

If you want to run with CUDA, set the environment variable `USE_CUDA=1`.

#### Norm patch

Gradient of `torch.norm` on GPU doesn't work on the latest release (PyTorch 0.2.0-post3).  It is fixed in
the `master` branch.  I grabbed the [fix](https://github.com/pytorch/pytorch/pull/2775) as a patch file
(`norm.patch`).  To apply it, find the python
site package directory where you have PyTorch installed (in my case, it's `~/.local/lib/python3.6/site-packages`).
Then, run the following command:

```
$ patch -d <your-site-package-directory-path> -p1 <norm.patch
```

#### Visdom patch

`visdom` video visualization code messed up the video size, and they have yet to fix that (I left a comment in the
latest related commit instead of opening up an issue).

You'll have to locate the `visdom` package directory (`~/.local/lib/python3.6/site-packages/visdom` in my case),
and run

```
$ patch -d <visdom-package-directory> -p2 <visdom-video.patch
```

### Result so far

Seems that the training process is very unstable even with SGD without momentum.
It either got comparably good results very soon or got completely thrown off.

The best result on validation dataset differs when I evaluate on all complete sequences or all 30-frame
sequences, which I got 45% and 65% respectively.  The reason of the huge drop by complete sequences is
that the bounding boxes are given by a detector, which is noisy and often fail horribly on the first frame.

I indeed got 75% on training set though.

### Code organization

The key ingredients:

* `main.py`: Entry program
* `hart.py`: Recurrent tracker itself, without the attention cell, and its loss functions
* `attention.py`: Attention module
  * Note that I didn't use their fixed-scale attention; the scale parameter of the attention is predicted.

Some miscellaneous modules:

* `adaptive_loss.py`: Adaptive loss module, which automatically balances multi-task losses
* `alexnet.py`: AlexNet(-ish) feature extractor
* `dfn.py`: Dynamic Convolutional Filter and its weight generator module
* `kth.py`: KTH Dataset
* `viz.py`: Visdom helper class for drawing matplotlib figures & time series
* `zoneout.py`: Zoneout implementations, including a variation of Zoneout-LSTM
* `util.py`: Things I don't know where to put

### TODO

- [x] Gradient clipping
- [x] CUDA support
- [x] Validation and Test
- [x] Visualization (with Visdom, patch needed; see above)
- [x] L2 Regularization (patch needed; see above)
- [x] ImageNet VID challenge
- [ ] KITTI

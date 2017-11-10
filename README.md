HART on PyTorch
===

A minimal (?) reproduction of [HART](https://github.com/akosiorek/hart).

### Usage

Download KTH Dataset first from [here](https://drive.google.com/a/nyu.edu/file/d/1KBQFWWaUg1ePPX2EXietBtGL2kOf-QIU/view?usp=sharing)
and unpack it to the repo directory.
You will have `KTHBoundingBoxInfo.txt` and a directory called `frames` in the
same place as the source code files.

Depending on whether you are enabling L2 regularization, you may need to apply a patch (see [below](#norm-patch)),
unless you are running a bleeding-edge version or release version later than 0.2.0-post3.
If you do, make sure that the patching succeeded.

Finally, run `main.py` with Python 3.

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

### TODO

- [ ] Gradient clipping
- [x] CUDA support
- [x] Validation and Test
- [ ] Visualization (Tensorboard or Visdom; I would like to use Visdom but probably not until they have sliders)
- [x] L2 Regularization (patch needed; see above)

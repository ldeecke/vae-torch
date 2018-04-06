This repository contains an implementation for training a **variational autoencoder** *(Kingma et al., 2014)*, that makes (almost exclusive) use of pytorch.

Training is available for data from MNIST, CIFAR10, and both datasets may be conditioned on an individual digit or class (using `--training_digits`). To initialize training, simply go ahead and `python3 train.py`.

For scoring anomalies on the respective test set, evoke `python3 score_elbo.py` and make sure to point toward a trained instance with `--ckpt_path`.

Other available commands are listed by calling `python3 train.py -h`.

---

Kingma, D. P. & Welling, M. (2013). **Auto-encoding variational bayes**. *arXiv preprint arXiv:1312.6114*.

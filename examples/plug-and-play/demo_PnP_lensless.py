import deepinv as dinv
from pathlib import Path
import torch
from deepinv.models import DnCNN
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim.optimizers import optim_builder
from deepinv.utils.plotting import plot, plot_curves
from deepinv.utils.demo import load_url_image, get_image_url

# %%
# Setup paths for data loading and results.
# ----------------------------------------------------------------------------------------
#
BASE_DIR = Path(".")
RESULTS_DIR = BASE_DIR / "results"

# %%
# Load image and parameters
# ----------------------------------------------------------------------------------------

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
# load data
downsample = 2
repo_path = "bezzam/DiffuserCam-Lensless-Mirflickr-Dataset"
grayscale = False

url = get_image_url("psf.png", repo_path=repo_path)
psf = load_url_image(
    # PSF is 4x larger than the image
    url=url,
    downsample=downsample * 4,
    grayscale=grayscale,
    resize_mode="resize",
    device=device,
)
url = get_image_url("lensless_example.png", repo_path=repo_path)
y = load_url_image(
    url=url,
    downsample=downsample,
    grayscale=grayscale,
    resize_mode="resize",
    device=device,
)
url = get_image_url("lensed_example.png", repo_path=repo_path)
x = load_url_image(
    url=url,
    downsample=downsample,
    grayscale=grayscale,
    resize_mode="resize",
    device=device,
)
img_size = tuple(y.shape[-2:])

# %%
# Set the forward operator
# --------------------------------------------------------------------------------
# We use the :class:`deepinv.physics.Tomography`
# class from the physics module to generate a CT measurements.


noise_level_img = 0.03  # Gaussian Noise standard deviation for the degradation
n_channels = 1 if grayscale else 3
physics = dinv.physics.Lensless(
    psf=psf,
    img_size=img_size,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
)

# %%
# Set up the PnP algorithm to solve the inverse problem.
# --------------------------------------------------------------------------------
# We use the Proximal Gradient Descent optimization algorithm.
# The algorithm alternates between a denoising step and a gradient descent step.
# The denoising step is performed by a DNCNN pretrained denoiser :class:`deepinv.models.DnCNN`.
#
# Set up the PnP algorithm parameters : the ``stepsize``, ``g_param`` the noise level of the denoiser
# and ``lambda`` the regularization parameter. The following parameters have been chosen manually.

# Logging parameters
verbose = True
plot_metrics = True  # compute performance and convergence metrics along the algorithm, curved saved in RESULTS_DIR

params_algo = {"stepsize": 1.0, "g_param": noise_level_img, "lambda": 0.01}
max_iter = 100
early_stop = True

# Select the data fidelity term
data_fidelity = L2()

# Specify the denoising prior
denoiser = DnCNN(
    in_channels=n_channels,
    out_channels=n_channels,
    pretrained="download",  # automatically downloads the pretrained weights, set to a path to use custom weights.
    train=False,
    device=device,
)
prior = PnP(denoiser=denoiser)

# instantiate the algorithm class to solve the IP problem.
model = optim_builder(
    iteration="PGD",
    prior=prior,
    data_fidelity=data_fidelity,
    early_stop=early_stop,
    max_iter=max_iter,
    verbose=verbose,
    params_algo=params_algo,
)

# %%
# Evaluate the model on the problem and plot the results.
# --------------------------------------------------------------------
#
# The model returns the output and the metrics computed along the iterations.
# For computing PSNR, the ground truth image ``x_gt`` must be provided.

x_lin = physics.A_adjoint(y)  # linear reconstruction with the adjoint operator

# run the model on the problem.
x_model, metrics = model(
    y, physics, x_gt=x, compute_metrics=True
)  # reconstruction with PnP algorithm

# compute PSNR
print(f"Linear reconstruction PSNR: {dinv.utils.metric.cal_psnr(x, x_lin):.2f} dB")
print(f"PnP reconstruction PSNR: {dinv.utils.metric.cal_psnr(x, x_model):.2f} dB")

# plot images. Images are saved in RESULTS_DIR.
imgs = [y, x, x_lin, x_model]
plot(
    imgs,
    titles=["Input", "GT", "Linear", "Recons."],
    save_dir=RESULTS_DIR / "images",
    show=True,
)

# plot convergence curves. Metrics are saved in RESULTS_DIR.
if plot_metrics:
    plot_curves(metrics, save_dir=RESULTS_DIR / "curves", show=True)

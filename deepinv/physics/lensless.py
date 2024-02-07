import torch
import torch.fft as fft
from deepinv.physics.forward import Physics, LinearPhysics, DecomposablePhysics
from deepinv.physics.blur import filter_fft


class Lensless(LinearPhysics):

    def __init__(self, img_size, psf, device="cpu", **kwargs):
        super().__init__(**kwargs)
        # PSF and image have same shape
        self.img_size = psf.shape
        # precompute the fft of the psf
        self.psf_fft = filter_fft(psf, img_size, real_fft=True)
        self.psf_fft = torch.nn.Parameter(self.psf_fft, requires_grad=False).to(device)

    def A(self, x):
        x_fft = filter_fft(x, self.img_size, real_fft=True)
        # TODO remove padding?
        return fft.irfft2(x_fft * self.psf_fft)

    def A_adjoint(self, y):
        y_fft = filter_fft(y, self.img_size, real_fft=True)
        # TODO remove padding?
        return fft.irfft2(y_fft * torch.conj(self.psf_fft))

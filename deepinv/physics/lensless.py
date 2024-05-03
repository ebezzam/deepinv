import torch
import torch.fft as fft
import numpy as np
from scipy.fftpack import next_fast_len
from deepinv.physics.forward import LinearPhysics
from deepinv.physics.blur import filter_fft


class Lensless(LinearPhysics):

    def __init__(self, img_size, psf, norm="ortho", device="cpu", **kwargs):
        super().__init__(**kwargs)
        # PSF and image have same shape
        self.img_size = psf.shape

        # normalize
        psf /= np.linalg.norm(psf.ravel())

        # cropping / padding indexes
        self._psf_shape = np.array(psf.shape)
        self._padded_shape = 2 * self._psf_shape[-2:] - 1
        self._padded_shape = np.array([next_fast_len(i) for i in self._padded_shape])
        self._padded_shape = list(np.r_[self._psf_shape[:2], self._padded_shape])
        self._start_idx = (self._padded_shape[-2:] - self._psf_shape[-2:]) // 2
        self._end_idx = self._start_idx + self._psf_shape[-2:]

        # precompute the fft of the psf
        self._H = torch.fft.rfft2(
            self._pad(psf), norm=norm, dim=(-2, -1), s=self._padded_shape[-2:]
        )
        self._Hadj = torch.conj(self._H)
        self._padded_data = (
            None  # This must be reinitialized each time to preserve differentiability
        )

    def A(self, x):
        self._padded_data = self._pad(x)
        conv_output = torch.fft.ifftshift(
            torch.fft.irfft2(
                torch.fft.rfft2(self._padded_data, dim=(-2, -1)) * self._H,
                dim=(-2, -1),
                s=self._padded_shape[-2:],
            ),
            dim=(-2, -1),
        )
        return self._crop(conv_output)
    
        # x_fft = filter_fft(x, self.img_size, real_fft=True)
        # # TODO remove padding?
        # return fft.irfft2(x_fft * self.psf_fft)

    def A_adjoint(self, y):
        self._padded_data = self._pad(y)
        deconv_output = torch.fft.ifftshift(
            torch.fft.irfft2(
                torch.fft.rfft2(self._padded_data, dim=(-2, -1)) * self._Hadj,
                dim=(-2, -1),
                s=self._padded_shape[-2:],
            ),
            dim=(-2, -1),
        )
        return self._crop(deconv_output)

        # y_fft = filter_fft(y, self.img_size, real_fft=True)
        # # TODO remove padding?
        # return fft.irfft2(y_fft * torch.conj(self.psf_fft))
    
    def _crop(self, x):
        return x[
            ..., self._start_idx[0] : self._end_idx[0], self._start_idx[1] : self._end_idx[1]
        ]

    def _pad(self, v):
        vpad = torch.zeros(size=self._padded_shape, dtype=v.dtype, device=v.device)
        vpad[
            ..., self._start_idx[0] : self._end_idx[0], self._start_idx[1] : self._end_idx[1]
        ] = v
        return vpad

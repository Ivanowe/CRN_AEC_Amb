import torch
import torch.nn.functional as F

from utils.stft import STFT


class NetFeeder(object):
    def __init__(self, device, win_size=320, hop_size=160):
        self.eps = torch.finfo(torch.float32).eps
        # Return STFT object to device
        self.stft = STFT(win_size, hop_size).to(device)

    # Now supports multi-channel input and output
    def __call__(self, mix, sph):
    # Initialize lists to store the "features" (stft of mix) and "labels" (stft of sph)
        feat_list = []
        lbl_list = []

        # Iterate for each channel in the input mixture
        for i in range(mix.shape[1]):       
            real_mix, imag_mix = self.stft.stft(mix[:, i, :])
            # Stack the real and imaginary parts along a new dimension
            feat = torch.stack([real_mix, imag_mix], dim=1)
            feat_list.append(feat)

        # Iterate for each channel in the target speech
        for i in range(sph.shape[1]):
            real_sph, imag_sph = self.stft.stft(sph[:, i, :])
            # Stack the real and imaginary parts along a new dimension
            lbl = torch.stack([real_sph, imag_sph], dim=1)
            lbl_list.append(lbl)

        # Concatenate the features and labels along the channel dimension
        feat = torch.cat(feat_list, dim=1)
        lbl = torch.cat(lbl_list, dim=1)

        return feat, lbl


class Resynthesizer(object):
    def __init__(self, device, win_size=320, hop_size=160):
        # Return STFT object to device
        self.stft = STFT(win_size, hop_size).to(device)

    # Create audio samples from estimated spectrum
    # Multichannel support implemented for future-proofing
    def __call__(self, est, mix):
        sph_est_list = []
        for i in range(0, est.shape[1], 2):
            est_i = est[:, i:i+2, :, :]
            sph_est = self.stft.istft(est_i)
            sph_est = F.pad(sph_est, [0, mix.shape[2]-sph_est.shape[1]])
            sph_est_list.append(sph_est)

        sph_est = torch.stack(sph_est_list, dim=1)
        return sph_est

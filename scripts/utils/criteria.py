import torch
from utils.stft import STFT


# class LossFunction(object):
#     def __call__(self, est, lbl, loss_mask, n_frames):
#         est_t = est * loss_mask
#         lbl_t = lbl * loss_mask

#         n_feats = est.shape[-1]

#         loss = torch.sum((est_t - lbl_t)**2) / float(sum(n_frames) * n_feats)
        
#         return loss

class LossFunction(object):
    def __init__(self, device, win_size=320, hop_size=160):
        self.stft = STFT(win_size, hop_size).to(device)
    
    def _imae(self, est, lbl):
         
        est_real = est[:, ::2, :, :]
        est_imag = est[:, 1::2, :, :]
        lbl_imag = lbl[:, 1::2, :, :]
        lbl_real = lbl[:, ::2, :, :]
        
        real_loss = torch.abs(est_real - lbl_real).mean()

        # L1 loss of the imaginary parts
        imag_loss = torch.abs(est_imag - lbl_imag).mean()

        # L1 loss of the magnitudes
        est_mag = torch.sqrt(est_real**2 + est_imag**2 + 1e-8)
        lbl_mag = torch.sqrt(lbl_real**2 + lbl_imag**2 + 1e-8)
        mag_loss = torch.abs(est_mag - lbl_mag).mean()

        # Combine the losses
        imae_loss = real_loss + imag_loss + mag_loss

        return imae_loss
        
    
    def _sdr(self, est, lbl):
        sk = self.stft.istft(lbl)
        sk_hat = self.stft.istft(est)
        
        sdr_loss = 10 * torch.log10((torch.pow(torch.norm(sk, p=2), 2) + 1e-8) 
                            / (torch.pow(torch.norm((sk - sk_hat), p=2), 2) + 1e-8))
        return sdr_loss
    
    def __call__(self, est, lbl, loss_mask, n_frames, weight_fac=0.1):
        est_t = est * loss_mask
        lbl_t = lbl * loss_mask
        n_bins = est.shape[-1]
        imae_loss = self._imae(est_t, lbl_t)
        sdr_loss = self._sdr(est_t, lbl_t)

        loss = (imae_loss / (n_bins * sum(n_frames))) - (weight_fac * sdr_loss)
        
        return loss
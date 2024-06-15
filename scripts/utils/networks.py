import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    # Define class variables
    ch_in = 5
    ch_out = 1
    ch_hid = 64
    def __init__(self):
        super(Net, self).__init__()
        ch_in = Net.ch_in
        ch_out = Net.ch_out
        ch_hid = Net.ch_hid
        self.conv1 = nn.Conv2d(in_channels=2 * ch_in, out_channels=ch_hid, kernel_size=(5,1), stride=(1,1), padding=(1,0))
        self.conv2 = nn.Conv2d(in_channels=ch_hid, out_channels=ch_hid, kernel_size=(5,1), stride=(1,1), padding=(1,0))
        self.conv3 = nn.Conv2d(in_channels=ch_hid, out_channels=ch_hid, kernel_size=(5,1), stride=(1,1), padding=(1,0))
        self.conv4 = nn.Conv2d(in_channels=ch_hid, out_channels=ch_hid, kernel_size=(5,1), stride=(1,1), padding=(1,0))
        self.conv5 = nn.Conv2d(in_channels=ch_hid, out_channels=ch_hid, kernel_size=(5,1), stride=(1,1), padding=(1,0))
        self.conv6 = nn.Conv2d(in_channels=ch_hid, out_channels=ch_hid, kernel_size=(5,1), stride=(1,1), padding=(1,0))
        
        self.lstm = nn.LSTM(input_size=ch_hid, hidden_size= 2 * ch_hid, num_layers=2, batch_first=True)
        
        self.freq_linear = nn.Linear(2 * ch_hid, ch_hid)

        self.conv6_t_pha = nn.ConvTranspose2d(in_channels=2 * ch_hid, out_channels=ch_hid, kernel_size=(5,1), stride=(1,1), padding=(1,0))
        self.conv5_t_pha = nn.ConvTranspose2d(in_channels=2 * ch_hid, out_channels=ch_hid, kernel_size=(5,1), stride=(1,1), padding=(1,0))
        self.conv4_t_pha = nn.ConvTranspose2d(in_channels=2 * ch_hid, out_channels=ch_hid, kernel_size=(5,1), stride=(1,1), padding=(1,0))
        self.conv3_t_pha = nn.ConvTranspose2d(in_channels=2 * ch_hid, out_channels=ch_hid, kernel_size=(5,1), stride=(1,1), padding=(1,0))
        self.conv2_t_pha = nn.ConvTranspose2d(in_channels=2 * ch_hid, out_channels=ch_hid, kernel_size=(5,1), stride=(1,1), padding=(1,0))
        self.conv1_t_pha = nn.ConvTranspose2d(in_channels=2 * ch_hid, out_channels=2 * ch_out, kernel_size=(5,1), stride=(1,1), padding=(1,0))
        
        self.conv6_t_amp = nn.ConvTranspose2d(in_channels=2 * ch_hid, out_channels=ch_hid, kernel_size=(5,1), stride=(1,1), padding=(1,0))
        self.conv5_t_amp = nn.ConvTranspose2d(in_channels=2 * ch_hid, out_channels=ch_hid, kernel_size=(5,1), stride=(1,1), padding=(1,0))
        self.conv4_t_amp = nn.ConvTranspose2d(in_channels=2 * ch_hid, out_channels=ch_hid, kernel_size=(5,1), stride=(1,1), padding=(1,0))
        self.conv3_t_amp = nn.ConvTranspose2d(in_channels=2 * ch_hid, out_channels=ch_hid, kernel_size=(5,1), stride=(1,1), padding=(1,0))
        self.conv2_t_amp = nn.ConvTranspose2d(in_channels=2 * ch_hid, out_channels=ch_hid, kernel_size=(5,1), stride=(1,1), padding=(1,0))
        self.conv1_t_amp = nn.ConvTranspose2d(in_channels=2 * ch_hid, out_channels=2 * ch_out, kernel_size=(5,1), stride=(1,1), padding=(1,0))
        
        self.bn1 = nn.BatchNorm2d(ch_hid)
        self.bn2 = nn.BatchNorm2d(ch_hid)
        self.bn3 = nn.BatchNorm2d(ch_hid)
        self.bn4 = nn.BatchNorm2d(ch_hid)
        self.bn5 = nn.BatchNorm2d(ch_hid)

        self.bn6_t_pha = nn.BatchNorm2d(ch_hid)           
        self.bn5_t_pha = nn.BatchNorm2d(ch_hid)
        self.bn4_t_pha = nn.BatchNorm2d(ch_hid)
        self.bn3_t_pha = nn.BatchNorm2d(ch_hid)
        self.bn2_t_pha = nn.BatchNorm2d(ch_hid)

        self.bn6_t_amp = nn.BatchNorm2d(ch_hid)           
        self.bn5_t_amp = nn.BatchNorm2d(ch_hid)
        self.bn4_t_amp = nn.BatchNorm2d(ch_hid)
        self.bn3_t_amp = nn.BatchNorm2d(ch_hid)
        self.bn2_t_amp = nn.BatchNorm2d(ch_hid)

        
        self.out_pha_r = nn.Linear(161, 161)
        self.out_pha_i = nn.Linear(161, 161)
        
        self.S_map_abs_out = nn.Linear(161, 161)
        self.M_out = nn.Linear(161, 161)

        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        
        out = x.permute(0, 1, 3, 2)  # Swap bin and frame dimensions
        # encoder
        e1 = self.elu(self.bn1(self.conv1(out).contiguous()))
        e2 = self.elu(self.bn2(self.conv2(e1).contiguous()))
        e3 = self.elu(self.bn3(self.conv3(e2).contiguous()))
        e4 = self.elu(self.bn4(self.conv4(e3).contiguous()))
        e5 = self.elu(self.bn5(self.conv5(e4).contiguous()))
        e6 = self.conv6(e5).contiguous()
        
        out = e6.contiguous()
        # reshape
        batch_size = out.size(0)
        channels = out.size(1)
        bins = out.size(2)
        frames = out.size(3)
        out = out.contiguous().view(batch_size * bins, frames, channels)
        # lstm
        out, _ = self.lstm(out)
        # linear
        out = self.freq_linear(out)
        # reshape
        out = out.contiguous().view(batch_size, channels, bins, frames)


        out = torch.cat([out, e6], dim=1)
        
        # phase decoder
        d6_pha = self.elu(torch.cat([self.bn6_t_pha(self.conv6_t_pha(out).contiguous()), e5], dim=1))
        d5_pha = self.elu(torch.cat([self.bn5_t_pha(self.conv5_t_pha(d6_pha).contiguous()), e4], dim=1))
        d4_pha = self.elu(torch.cat([self.bn4_t_pha(self.conv4_t_pha(d5_pha).contiguous()), e3], dim=1))
        d3_pha = self.elu(torch.cat([self.bn3_t_pha(self.conv3_t_pha(d4_pha).contiguous()), e2], dim=1))
        d2_pha = self.elu(torch.cat([self.bn2_t_pha(self.conv2_t_pha(d3_pha).contiguous()), e1], dim=1))
        d1_pha = self.conv1_t_pha(d2_pha).contiguous()
        
        # reshape
        d1_pha_out = d1_pha.permute(0, 1, 3, 2)
        
        # Split the output of the phase decoder into real and imaginary parts
        d1_pha_r, d1_pha_i = torch.chunk(d1_pha_out, chunks=2 * Net.ch_out, dim=1)
        out_pha_r = self.out_pha_r(d1_pha_r)
        out_pha_i = self.out_pha_i(d1_pha_i)
        
        eps = np.finfo(float).eps # small number to avoid division by zero
        # Does not work with the current version of PyTorch
        psi = ((out_pha_r + 1j * out_pha_i) / (torch.sqrt(out_pha_r**2 + out_pha_i**2) + eps)).squeeze(dim=1)
        
        # amplitude decoder
        d6_amp = self.elu(torch.cat([self.bn6_t_amp(self.conv6_t_amp(out).contiguous()), e5], dim=1))
        d5_amp = self.elu(torch.cat([self.bn5_t_amp(self.conv5_t_amp(d6_amp).contiguous()), e4], dim=1))
        d4_amp = self.elu(torch.cat([self.bn4_t_amp(self.conv4_t_amp(d5_amp).contiguous()), e3], dim=1))
        d3_amp = self.elu(torch.cat([self.bn3_t_amp(self.conv3_t_amp(d4_amp).contiguous()), e2], dim=1))
        d2_amp = self.elu(torch.cat([self.bn2_t_amp(self.conv2_t_amp(d3_amp).contiguous()), e1], dim=1))
        d1_amp = self.conv1_t_amp(d2_amp).contiguous()
        
        # reshape
        d1_amp_out = d1_amp.permute(0, 1, 3, 2)
        
        d1_amp_S, d1_amp_M = torch.chunk(d1_amp_out, chunks=2 * Net.ch_out, dim=1)
        
        S_map_abs = self.S_map_abs_out(d1_amp_S).squeeze(dim=1)
        M = torch.sigmoid(self.M_out(d1_amp_M)).squeeze(dim=1)
        
        x_real = x[:, 0, :, :]
        x_imag = x[:, 1, :, :]

        # Calculate the amplitude of the spectrum
        Y_abs = torch.sqrt(torch.pow(x_real, 2) + torch.pow(x_imag, 2)) 
        
        S_hat_abs = Y_abs * M + S_map_abs
        
        # Does not work with the current version of PyTorch, see above
        S_hat = S_hat_abs * psi
        
        out1 = S_hat.real.unsqueeze(dim=1)
        out2 = S_hat.imag.unsqueeze(dim=1)
        
        
        out = torch.cat([out1, out2], dim=1)

        return out

        

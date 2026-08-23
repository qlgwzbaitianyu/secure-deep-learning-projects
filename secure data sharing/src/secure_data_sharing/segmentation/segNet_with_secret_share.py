import torch
from torch import compile as torch_compile
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import time
from random import randrange
from smc_function_number import smc_functions
import sys
import random
import math

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

class SegNet(nn.Module):
    def __init__(self, input_channels, output_channels, smc_config):
        super(SegNet, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.smc_config = smc_config

        self.num_channels = input_channels

                                                 

        self.encoder_conv_00 =  nn.Conv2d(in_channels=self.input_channels, out_channels=64, kernel_size=3,padding=1, bias=False)
        self.encoder_conv_01 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.encoder_conv_10 = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3, padding=1, bias=False)
        self.encoder_conv_11 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3, padding=1, bias=False)
        self.encoder_conv_20 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding=1, bias=False)
        self.encoder_conv_21 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, padding=1, bias=False)
        self.encoder_conv_22 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, padding=1, bias=False)
        self.encoder_conv_30 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1, bias=False)
        self.encoder_conv_31 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1, bias=False)
        self.encoder_conv_32 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1, bias=False)
        self.encoder_conv_40 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1, bias=False)
        self.encoder_conv_41 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1, bias=False)
        self.encoder_conv_42 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1, bias=False)

        self.decoder_convtr_42 = nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,padding=1, bias=False)
        self.decoder_convtr_41 = nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,padding=1, bias=False)
        self.decoder_convtr_40 = nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,padding=1, bias=False)
        self.decoder_convtr_32 = nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,padding=1, bias=False)
        self.decoder_convtr_31 = nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,padding=1, bias=False)
        self.decoder_convtr_30 = nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=3,padding=1, bias=False)
        self.decoder_convtr_22 = nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=3,padding=1, bias=False)
        self.decoder_convtr_21 = nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=3,padding=1, bias=False)
        self.decoder_convtr_20 = nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=3,padding=1, bias=False)
        self.decoder_convtr_11 = nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=3,padding=1, bias=False)
        self.decoder_convtr_10 = nn.ConvTranspose2d(in_channels=128,out_channels=64, kernel_size=3,padding=1, bias=False)
        self.decoder_convtr_01 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=3,padding=1, bias=False)
        self.decoder_convtr_00 = nn.ConvTranspose2d(in_channels=64,out_channels=self.output_channels,kernel_size=3,padding=1, bias=False)
        self.relu = nn.ReLU(True)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, dilation=1, return_indices=True)

    def pcnnReluA(self, bottom, top):
        smc = smc_functions(self.smc_config)
        bn = bottom.size()[0]
        bc = bottom.size()[1]
        bh = bottom.size()[2]
        bw = bottom.size()[3]

        for n in range(bn):
            for c in range(bc):
                for h in range(bh):
                    for w in range(bw):
                        u_1 = bottom[n, c, h, w].item()
                        u_2 = top[n, c, h, w].item()
                        bit_1, bit_2, MSB_1, MSB_2 = smc.SecBitExtra(u_1, u_2)
                        bit = MSB_1 ^ MSB_2
                        if bit == 1:
                            bottom[n, c, h, w] = 0
        out = bottom.detach().clone()
        return out

    def pcnnReluB(self, bottom, top):

        smc = smc_functions(self.smc_config)
        bn = bottom.size()[0]
        bc = bottom.size()[1]
        bh = bottom.size()[2]
        bw = bottom.size()[3]

        for n in range(bn):
            for c in range(bc):
                for h in range(bh):
                    for w in range(bw):
                        u_1 = bottom[n, c, h, w].item()
                        u_2 = top[n, c, h, w].item()
                        bit_1, bit_2, MSB_1, MSB_2 = smc.SecBitExtra(u_1, u_2)
                        bit = MSB_1 ^ MSB_2
                        if bit == 1:
                            top[n, c, h, w] = 0
        out = top.detach().clone()
        return out

    def sec_batchnom(self, share1, share2):
        total_element = torch.numel(share1)
        sum1 = torch.sum(share1).item()
        sum2 = torch.sum(share2).item()
        mean = (sum1 + sum2) / total_element

        temp2 = share2 - mean
        numpy_temp2 = temp2.detach().numpy()
        numpy_temp2 = np.square(numpy_temp2)
        temp2 = torch.from_numpy(numpy_temp2)
        var2 = torch.sum(temp2).item()  / total_element

        temp1 = share1 - mean
        numpy_temp1 = temp1.detach().numpy()
        numpy_temp1 = np.square(numpy_temp1)
        temp1 = torch.from_numpy(numpy_temp1)
        var1 = torch.sum(temp1).item() - total_element* (pow(mean,2))

        flatterned_share1 = share1.reshape(-1)
        flatterned_share2 = share2.reshape(-1)
        two_ab = self.get_2ab(flatterned_share1, flatterned_share2)
        var1 = (var1 + two_ab) / total_element

        var_combine = var1 + var2
        var_combine = math.sqrt(var_combine)
        batchnorm_share1 = (share1) / var_combine
        batchnorm_share2 = (share2 - mean) / var_combine

        share = share1 + share2
        temp = share - mean
        numpy_temp = temp.detach().numpy()
        numpy_temp = np.square(numpy_temp)
        temp = torch.from_numpy(numpy_temp)
        var = torch.sum(temp).item() / total_element
        var = math.sqrt(var)

        batchnorm_share = (share - mean) / var

        diff = batchnorm_share - (batchnorm_share1 + batchnorm_share2)

        return batchnorm_share1, batchnorm_share2, batchnorm_share

    def batch_norm(self, share):
        sum = torch.sum(share).item()
        total_element = torch.numel(share)
        mean = sum / total_element

        temp = share - mean
        numpy_temp = temp.detach().numpy()
        numpy_temp = np.square(numpy_temp)
        temp = torch.from_numpy(numpy_temp)
        var = torch.sum(temp).item() / total_element

        var = math.sqrt(var)
        batchnorm_share = (share - mean) / var
        return batchnorm_share

    def get_2ab(self, share1, share2):
        share1 = share1.detach().numpy()
        share2 = share2.detach().numpy()
        ratio_in_share1 = []
        exchange_data_from_share2 = []
        result = 0
        index = 0

        for i, k in zip(share1[0::2], share1[1::2]):
            ratio_in_share1.append(i / k)

        for i, k in zip(share2[0::2], share2[1::2]):
            ratio = ratio_in_share1[index]
            index = index + 1
            rand = random.uniform(0, 1)
            b1 = i + rand
            b2 = k - ratio * rand
            exchange_data_from_share2.append(b1)
            exchange_data_from_share2.append(b2)

        index = 0
        for i in share1:
            data_from_b = exchange_data_from_share2[index]
            index = index + 1
            result = result + 2*i*data_from_b

        return result

    def forward(self, input_img, share1, share2):
        m = nn.BatchNorm2d(64)
        dim_0 = input_img.size()
        x_00_cov = self.encoder_conv_00(input_img)
        x_00_cov_share1 = self.encoder_conv_00(share1)
        x_00_cov_share2 = self.encoder_conv_00(share2)
        diff1 = x_00_cov - (x_00_cov_share1 + x_00_cov_share2)

                                
        x_00_cov_norm = self.batch_norm(x_00_cov)

        start_time = time.time()
        x_00_cov_share1_norm, x_00_cov_share2_norm, x_00_cov_norm = self.sec_batchnom(x_00_cov_share1, x_00_cov_share2)
        print(x_00_cov_share1.shape)
        print("--- conv1(128*32*32) time %s seconds ---" % (time.time() - start_time))

        diff2 = x_00_cov_norm - (x_00_cov_share1_norm + x_00_cov_share2_norm)

        x_00_relu = self.relu(x_00_cov)
        x_00_relu_share1 = self.pcnnReluA(x_00_cov_share1, x_00_cov_share2)
        x_00_relu_share2 = self.pcnnReluB(x_00_cov_share1, x_00_cov_share2)
        diff3 = x_00_relu - (x_00_relu_share1 + x_00_relu_share2)

                                                           
        x_01 = self.encoder_conv_01(x_00_relu)
        x_01_share1 = self.encoder_conv_01(x_00_relu_share1)
        x_01_share2 = self.encoder_conv_01(x_00_relu_share2)

        x_01 = self.batch_norm(x_01)
        x_01_share1, x_01_share2, x_01 = self.sec_batchnom(x_01_share1, x_01_share2)
        diff4 = x_01 - (x_01_share1 + x_01_share2)

        x_01_relu = self.relu(x_01)
        x_01_relu_share1 = self.pcnnReluA(x_01_share1, x_01_share2)
        x_01_relu_share2 = self.pcnnReluB(x_01_share1, x_01_share2)
        diff = x_01_relu - (x_01_relu_share1 + x_01_relu_share2)
        x_0, indices_0 = self.maxPool(x_01_relu)
        x_0_share1 = self.MaxALayer(x_01_relu_share1, x_01_relu_share2).float()
        x_0_share2 = self.MaxBLayer(x_01_relu_share1, x_01_relu_share2).float()
        max1_1_error = x_0 - (x_0_share1 + x_0_share2)

                           
        dim_1 = x_0.size()
        x_10 = self.encoder_conv_10(x_0)
        x_10_share1 = self.encoder_conv_10(x_0_share1)
        x_10_share2 = self.encoder_conv_10(x_0_share2)
        x_10 = self.batch_norm(x_10)
        x_10_share1, x_10_share2, x_10 = self.sec_batchnom(x_10_share1, x_10_share2)

        x_10_relu = self.relu(x_10)
        x_10_relu_share1 = self.pcnnReluA(x_10_share1, x_10_share2)
        x_10_relu_share2 = self.pcnnReluB(x_10_share1, x_10_share2)
        diff = x_10_relu - (x_10_relu_share1 + x_10_relu_share2)
        x_11 = self.encoder_conv_11(x_10_relu)
        x_11_share1 = self.encoder_conv_11(x_10_relu_share1)
        x_11_share2 = self.encoder_conv_11(x_10_relu_share2)
        x_11 = self.batch_norm(x_11)
        x_11_share1, x_10_share2, x_11 = self.sec_batchnom(x_11_share1, x_11_share2)
        x_11_relu = self.relu(x_11)
        x_11_relu_share1 = self.pcnnReluA(x_11_share1, x_11_share2)
        x_11_relu_share2 = self.pcnnReluB(x_11_share1, x_11_share2)
        diff = x_11_relu - (x_11_relu_share1 + x_11_relu_share2)

        x_1, indices_1 = self.maxPool(x_11_relu)
        x_1_share1 = self.MaxALayer(x_11_relu_share1, x_11_relu_share2).float()
        x_1_share2 = self.MaxBLayer(x_11_relu_share1, x_11_relu_share2).float()
        diff = x_1 - (x_1_share1 + x_1_share2)

                           
        dim_2 = x_1.size()
        x_20 = self.encoder_conv_20(x_1)
        x_20_share1 = self.encoder_conv_20(x_1_share1)
        x_20_share2 = self.encoder_conv_20(x_1_share2)
        x_20 = self.batch_norm(x_20)
        x_20_share1, x_10_share2, x_20 = self.sec_batchnom(x_20_share1, x_20_share2)
        x_20_relu = self.relu(x_20)
        x_20_relu_share1 = self.pcnnReluA(x_20_share1, x_20_share2)
        x_20_relu_share2 = self.pcnnReluB(x_20_share1, x_20_share2)
        diff = x_20_relu - (x_20_relu_share1 + x_20_relu_share2)
        x_21 = self.encoder_conv_21(x_20_relu)
        x_21_share1 = self.encoder_conv_21(x_20_relu_share1)
        x_21_share2 = self.encoder_conv_21(x_20_relu_share2)
        x_21 = self.batch_norm(x_21)
        x_21_share1, x_21_share2, x_21 = self.sec_batchnom(x_21_share1, x_21_share2)
        x_21_relu = self.relu(x_21)
        x_21_relu_share1 = self.pcnnReluA(x_21_share1, x_21_share2)
        x_21_relu_share2 = self.pcnnReluB(x_21_share1, x_21_share2)
        diff = x_21_relu - (x_21_relu_share1 + x_21_relu_share2)
        x_22 = self.encoder_conv_22(x_21_relu)
        x_22_share1 = self.encoder_conv_22(x_21_relu_share1)
        x_22_share2 = self.encoder_conv_22(x_21_relu_share2)
        x_22 = self.batch_norm(x_22)
        x_22_share1, x_22_share2 , x_22= self.sec_batchnom(x_22_share1, x_22_share2)
        x_22_relu = self.relu(x_22)
        x_22_relu_share1 = self.pcnnReluA(x_22_share1, x_22_share2)
        x_22_relu_share2 = self.pcnnReluB(x_22_share1, x_22_share2)
        diff = x_22_relu - (x_22_relu_share1 + x_22_relu_share2)

        x_2, indices_2 = self.maxPool(x_22_relu)
        x_2_share1 = self.MaxALayer(x_22_relu_share1, x_22_relu_share2).float()
        x_2_share2 = self.MaxBLayer(x_22_relu_share1, x_22_relu_share2).float()
        diff = x_2 - (x_2_share1 + x_2_share2)

                           
        dim_3 = x_2.size()
        x_30 = self.encoder_conv_30(x_2)
        x_30_share1 = self.encoder_conv_30(x_2_share1)
        x_30_share2 = self.encoder_conv_30(x_2_share2)
        x_30 = self.batch_norm(x_30)
        x_30_share1, x_30_share2, x_30 = self.sec_batchnom(x_30_share1, x_30_share2)
        x_30_relu = self.relu(x_30)
        x_30_relu_share1 = self.pcnnReluA(x_30_share1, x_30_share2)
        x_30_relu_share2 = self.pcnnReluB(x_30_share1, x_30_share2)
        diff = x_30_relu - (x_30_relu_share1 + x_30_relu_share2)
        x_31 = self.encoder_conv_31(x_30_relu)
        x_31_share1 = self.encoder_conv_31(x_30_relu_share1)
        x_31_share2 = self.encoder_conv_31(x_30_relu_share2)
        x_31 = self.batch_norm(x_31)
        x_31_share1, x_31_share2, x_31 = self.sec_batchnom(x_31_share1, x_31_share2)
        x_31_relu = self.relu(x_31)
        x_31_relu_share1 = self.pcnnReluA(x_31_share1, x_31_share2)
        x_31_relu_share2 = self.pcnnReluB(x_31_share1, x_31_share2)
        diff = x_31_relu - (x_31_relu_share1 + x_31_relu_share2)
        x_32 = self.encoder_conv_32(x_31_relu)
        x_32_share1 = self.encoder_conv_32(x_31_relu_share1)
        x_32_share2 = self.encoder_conv_32(x_31_relu_share2)
        x_32 = self.batch_norm(x_32)
        x_32_share1, x_22_share2, x_32 = self.sec_batchnom(x_32_share1, x_32_share2)
        x_32_relu = self.relu(x_32)
        x_32_relu_share1 = self.pcnnReluA(x_32_share1, x_32_share2)
        x_32_relu_share2 = self.pcnnReluB(x_32_share1, x_32_share2)
        diff = x_32_relu - (x_32_relu_share1 + x_32_relu_share2)

        x_3, indices_3 = self.maxPool(x_32_relu)
        x_3_share1 = self.MaxALayer(x_32_relu_share1, x_32_relu_share2).float()
        x_3_share2 = self.MaxBLayer(x_32_relu_share1, x_32_relu_share2).float()
        diff = x_3 - (x_3_share1 + x_3_share2)

                                                            
        dim_4 = x_3.size()
        x_40 = self.encoder_conv_40(x_3)
        x_40_share1 = self.encoder_conv_40(x_3_share1)
        x_40_share2 = self.encoder_conv_40(x_3_share2)
        x_40 = self.batch_norm(x_40)
        x_40_share1, x_40_share2, x_40 = self.sec_batchnom(x_40_share1, x_40_share2)
        x_40_relu = self.relu(x_40)
        x_40_relu_share1 = self.pcnnReluA(x_40_share1, x_40_share2)
        x_40_relu_share2 = self.pcnnReluB(x_40_share1, x_40_share2)
        diff = x_40_relu - (x_40_relu_share1 + x_40_relu_share2)
        x_41 = self.encoder_conv_41(x_40_relu)
        x_41_share1 = self.encoder_conv_41(x_40_relu_share1)
        x_41_share2 = self.encoder_conv_41(x_40_relu_share2)
        x_41 = self.batch_norm(x_41)
        x_41_share1, x_41_share2, x_41 = self.sec_batchnom(x_41_share1, x_41_share2)
        x_41_relu = self.relu(x_41)
        x_41_relu_share1 = self.pcnnReluA(x_41_share1, x_41_share2)
        x_41_relu_share2 = self.pcnnReluB(x_41_share1, x_41_share2)
        diff = x_41_relu - (x_41_relu_share1 + x_41_relu_share2)
        x_42 = self.encoder_conv_42(x_41_relu)
        x_42_share1 = self.encoder_conv_42(x_41_relu_share1)
        x_42_share2 = self.encoder_conv_42(x_41_relu_share2)
        x_42 = self.batch_norm(x_42)
        x_42_share1, x_42_share2, x_42 = self.sec_batchnom(x_42_share1, x_42_share2)
        x_42_relu = self.relu(x_42)
        x_42_relu_share1 = self.pcnnReluA(x_42_share1, x_42_share2)
        x_42_relu_share2 = self.pcnnReluB(x_42_share1, x_42_share2)
        diff = x_42_relu - (x_42_relu_share1 + x_42_relu_share2)

        x_4, indices_4 = self.maxPool(x_42_relu)
        x_4_share1 = self.MaxALayer(x_42_relu_share1, x_42_relu_share2).float()
        x_4_share2 = self.MaxBLayer(x_42_relu_share1, x_42_relu_share2).float()
        diff = x_4 - (x_4_share1 + x_4_share2)

                                                            
        dim_d = x_4.size()
        x_4d = F.max_unpool2d(x_4, indices_4, kernel_size=2, stride=2, output_size=dim_4)
        x_4d_share1 = F.max_unpool2d(x_4_share1, indices_4, kernel_size=2, stride=2, output_size=dim_4)
        x_4d_share2 = F.max_unpool2d(x_4_share2, indices_4, kernel_size=2, stride=2, output_size=dim_4)

        x_42 = self.decoder_convtr_42(x_4d)
        x_42_share1 = self.decoder_convtr_42(x_4d_share1)
        x_42_share2 = self.decoder_convtr_42(x_4d_share2)

        x_42_share1, x_42_share2, x_42 = self.sec_batchnom(x_42_share1, x_42_share2)
        x_42_relu = self.relu(x_42)
        x_42_relu_share1 = self.pcnnReluA(x_42_share1, x_42_share2)
        x_42_relu_share2 = self.pcnnReluB(x_42_share1, x_42_share2)
        diff = x_42_relu - (x_42_relu_share1 + x_42_relu_share2)
        x_41 = self.decoder_convtr_41(x_42_relu)
        x_41_share1 = self.decoder_convtr_41(x_42_relu_share1)
        x_41_share2 = self.decoder_convtr_41(x_42_relu_share2)

        x_41_share1, x_41_share2, x_41 = self.sec_batchnom(x_41_share1, x_41_share2)
        x_41_relu = self.relu(x_41)
        x_41_relu_share1 = self.pcnnReluA(x_41_share1, x_41_share2)
        x_41_relu_share2 = self.pcnnReluB(x_41_share1, x_41_share2)
        diff = x_41_relu - (x_41_relu_share1 + x_41_relu_share2)
        x_40 = self.decoder_convtr_40(x_41_relu)
        x_40_share1 = self.decoder_convtr_40(x_41_relu_share1)
        x_40_share2 = self.decoder_convtr_40(x_41_relu_share2)

        x_40_share1, x_40_share2, x_40 = self.sec_batchnom(x_40_share1, x_40_share2)
        x_40_relu = self.relu(x_40)
        x_40_relu_share1 = self.pcnnReluA(x_40_share1, x_40_share2)
        x_40_relu_share2 = self.pcnnReluB(x_40_share1, x_40_share2)
        diff = x_40_relu - (x_40_relu_share1 + x_40_relu_share2)
        dim_4d = x_40_relu.size()

                           
        x_3d = F.max_unpool2d(x_40_relu, indices_3, kernel_size=2, stride=2, output_size=dim_3)
        x_3d_share1 = F.max_unpool2d(x_40_relu_share1, indices_3, kernel_size=2, stride=2, output_size=dim_3)
        x_3d_share2 = F.max_unpool2d(x_40_relu_share2, indices_3, kernel_size=2, stride=2, output_size=dim_3)

        x_32 = self.decoder_convtr_32(x_3d)
        x_32_share1 = self.decoder_convtr_32(x_3d_share1)
        x_32_share2 = self.decoder_convtr_32(x_3d_share2)

        x_32_share1, x_32_share2, x_32 = self.sec_batchnom(x_32_share1, x_32_share2)
        x_32_relu = self.relu(x_32)
        x_32_relu_share1 = self.pcnnReluA(x_32_share1, x_32_share2)
        x_32_relu_share2 = self.pcnnReluB(x_32_share1, x_32_share2)
        diff = x_32_relu - (x_32_relu_share1 + x_32_relu_share2)
        x_31 = self.decoder_convtr_31(x_32_relu)
        x_31_share1 = self.decoder_convtr_31(x_32_relu_share1)
        x_31_share2 = self.decoder_convtr_31(x_32_relu_share2)

        x_31_share1, x_31_share2, x_31 = self.sec_batchnom(x_31_share1, x_31_share2)
        x_31_relu = self.relu(x_31)
        x_31_relu_share1 = self.pcnnReluA(x_31_share1, x_31_share2)
        x_31_relu_share2 = self.pcnnReluB(x_31_share1, x_31_share2)
        diff = x_31_relu - (x_31_relu_share1 + x_31_relu_share2)
        x_30 = self.decoder_convtr_30(x_31_relu)
        x_30_share1 = self.decoder_convtr_30(x_31_relu_share1)
        x_30_share2 = self.decoder_convtr_30(x_31_relu_share2)

        x_30_share1, x_30_share2, x_30 = self.sec_batchnom(x_30_share1, x_30_share2)
        x_30_relu = self.relu(x_30)
        x_30_relu_share1 = self.pcnnReluA(x_30_share1, x_30_share2)
        x_30_relu_share2 = self.pcnnReluB(x_30_share1, x_30_share2)
        diff = x_30_relu - (x_30_relu_share1 + x_30_relu_share2)
        dim_4d = x_30_relu.size()

                           
        x_2d = F.max_unpool2d(x_30_relu, indices_2, kernel_size=2, stride=2, output_size=dim_2)
        x_2d_share1 = F.max_unpool2d(x_30_relu_share1, indices_2, kernel_size=2, stride=2, output_size=dim_2)
        x_2d_share2 = F.max_unpool2d(x_30_relu_share2, indices_2, kernel_size=2, stride=2, output_size=dim_2)

        x_22 = self.decoder_convtr_22(x_2d)
        x_22_share1 = self.decoder_convtr_22(x_2d_share1)
        x_22_share2 = self.decoder_convtr_22(x_2d_share2)

        x_22_share1, x_22_share2, x_22 = self.sec_batchnom(x_22_share1, x_22_share2)
        x_22_relu = self.relu(x_22)
        x_22_relu_share1 = self.pcnnReluA(x_22_share1, x_22_share2)
        x_22_relu_share2 = self.pcnnReluB(x_22_share1, x_22_share2)
        diff = x_22_relu - (x_22_relu_share1 + x_22_relu_share2)
        x_21 = self.decoder_convtr_21(x_22_relu)
        x_21_share1 = self.decoder_convtr_21(x_22_relu_share1)
        x_21_share2 = self.decoder_convtr_21(x_22_relu_share2)

        x_21_share1, x_21_share2, x_21 = self.sec_batchnom(x_21_share1, x_21_share2)
        x_21_relu = self.relu(x_21)
        x_21_relu_share1 = self.pcnnReluA(x_21_share1, x_21_share2)
        x_21_relu_share2 = self.pcnnReluB(x_21_share1, x_21_share2)
        diff = x_21_relu - (x_21_relu_share1 + x_21_relu_share2)
        x_20 = self.decoder_convtr_20(x_21_relu)
        x_20_share1 = self.decoder_convtr_20(x_21_relu_share1)
        x_20_share2 = self.decoder_convtr_20(x_21_relu_share2)

        x_20_share1, x_20_share2, x_20 = self.sec_batchnom(x_20_share1, x_20_share2)
        x_20_relu = self.relu(x_20)
        x_20_relu_share1 = self.pcnnReluA(x_20_share1, x_20_share2)
        x_20_relu_share2 = self.pcnnReluB(x_20_share1, x_20_share2)
        diff = x_20_relu - (x_20_relu_share1 + x_20_relu_share2)
        dim_2d = x_20_relu.size()

                                                            

                           
        x_1d = F.max_unpool2d(x_20_relu, indices_1, kernel_size=2, stride=2, output_size=dim_1)
        x_1d_share1 = F.max_unpool2d(x_20_relu_share1, indices_1, kernel_size=2, stride=2, output_size=dim_1)
        x_1d_share2 = F.max_unpool2d(x_20_relu_share2, indices_1, kernel_size=2, stride=2, output_size=dim_1)

        x_11 = self.decoder_convtr_11(x_1d)
        x_11_share1 = self.decoder_convtr_11(x_1d_share1)
        x_11_share2 = self.decoder_convtr_11(x_1d_share2)

        x_11_share1, x_11_share2, x_11 = self.sec_batchnom(x_11_share1, x_11_share2)
        x_11_relu = self.relu(x_11)
        x_11_relu_share1 = self.pcnnReluA(x_11_share1, x_11_share2)
        x_11_relu_share2 = self.pcnnReluB(x_11_share1, x_11_share2)
        diff = x_11_relu - (x_11_relu_share1 + x_11_relu_share2)

        x_10 = self.decoder_convtr_10(x_11_relu)
        x_10_share1 = self.decoder_convtr_10(x_11_relu_share1)
        x_10_share2 = self.decoder_convtr_10(x_11_relu_share2)

        x_10_share1, x_10_share2, x_10 = self.sec_batchnom(x_10_share1, x_10_share2)
        x_10_relu = self.relu(x_10)
        x_10_relu_share1 = self.pcnnReluA(x_10_share1, x_10_share2)
        x_10_relu_share2 = self.pcnnReluB(x_10_share1, x_10_share2)
        diff = x_10_relu - (x_10_relu_share1 + x_10_relu_share2)
        dim_1d = x_10_relu.size()

                                                            
        x_0d = F.max_unpool2d(x_10_relu, indices_0, kernel_size=2, stride=2, output_size=dim_0)
        x_0d_share1 = F.max_unpool2d(x_10_relu_share1, indices_0, kernel_size=2, stride=2, output_size=dim_0)
        x_0d_share2 = F.max_unpool2d(x_10_relu_share2, indices_0, kernel_size=2, stride=2, output_size=dim_0)

        x_01 = self.decoder_convtr_01(x_0d)
        x_01_share1 = self.decoder_convtr_01(x_0d_share1)
        x_01_share2 = self.decoder_convtr_01(x_0d_share2)

        x_01_share1, x_01_share2, x_01 = self.sec_batchnom(x_01_share1, x_01_share2)
        x_01_relu = self.relu(x_01)
        x_01_relu_share1 = self.pcnnReluA(x_01_share1, x_01_share2)
        x_01_relu_share2 = self.pcnnReluB(x_01_share1, x_01_share2)
        diff = x_01_relu - (x_01_relu_share1 + x_01_relu_share2)

        x_00 = self.decoder_convtr_00(x_01_relu)
        x_00_share1 = self.decoder_convtr_00(x_01_relu_share1)
        x_00_share2 = self.decoder_convtr_00(x_01_relu_share2)

        x_00_share1, x_00_share2, x_00 = self.sec_batchnom(x_00_share1, x_00_share2)
        x_00_relu = self.relu(x_00)
        x_00_relu_share1 = self.pcnnReluA(x_00_share1, x_00_share2)
        x_00_relu_share2 = self.pcnnReluB(x_00_share1, x_00_share2)
        diff = x_00_relu - (x_00_relu_share1 + x_00_relu_share2)
        dim_1d = x_00_relu.size()

        dim_0d = x_00_relu.size()

        x_softmax = F.softmax(x_00_relu, dim=1)
        x_softmax_share1 = F.softmax(x_00_relu_share1, dim=1)
        x_softmax_share2 = F.softmax(x_00_relu_share2, dim=1)

        diff = x_softmax - (x_softmax_share1 + x_softmax_share2)

        return x_00_relu, x_softmax

    def MaxALayer(self, bottom, top):
        bn = bottom.size()[0]
        bc = bottom.size()[1]
        bh = bottom.size()[2]
        bw = bottom.size()[3]
        tn = bn
        tc = bc
        th = int(bh / 2)
        tw = int(bw / 2)

        smc = smc_functions(self.smc_config)
        out = torch.zeros(tn, tc, th, tw, dtype=torch.double)
        for n in range(tn):
            for c in range(tc):

                                                
                temp_b1 = bottom[n, c]
                temp_b2 = top[n, c]
                temp_b = temp_b1 + temp_b2

                temp_t1 = torch.zeros(th, tw)
                for h in range(th):
                    for w in range(tw):
                        x, y = smc.SecMaxIndex(temp_b1[h * 2:h * 2 + 2, w * 2:w * 2 + 2],
                                               temp_b2[h * 2:h * 2 + 2, w * 2:w * 2 + 2])
                        self.counter = self.counter + 1
                        temp_t1[h, w] = temp_b1[h * 2 + x, w * 2 + y]

                out[n, c] = temp_t1.detach().clone()

        return out

    def MaxBLayer(self, bottom, top):
        bn = bottom.size()[0]
        bc = bottom.size()[1]
        bh = bottom.size()[2]
        bw = bottom.size()[3]
        tn = bn
        tc = bc
        th = int(bh / 2)
        tw = int(bw / 2)

        smc = smc_functions(self.smc_config)
        out = torch.zeros(tn, tc, th, tw, dtype=torch.double)
        for n in range(tn):
            for c in range(tc):

                                                
                temp_b1 = bottom[n, c]
                temp_b2 = top[n, c]
                temp_b = temp_b1 + temp_b2

                temp_t2 = torch.zeros(th, tw)
                for h in range(th):
                    for w in range(tw):
                        x, y = smc.SecMaxIndex(temp_b1[h * 2:h * 2 + 2, w * 2:w * 2 + 2],
                                               temp_b2[h * 2:h * 2 + 2, w * 2:w * 2 + 2])
                        temp_t2[h, w] = temp_b2[h * 2 + x, w * 2 + y]

                out[n, c] = temp_t2.detach().clone()
        return out

    def imageGen(self, image):
        for element in image:
            ele = element / randrange(2, 9)
        return image


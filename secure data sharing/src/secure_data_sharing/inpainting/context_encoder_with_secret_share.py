import torch
from torch import compile as torch_compile
import torch.nn as nn
from smc_function_number import smc_functions
import numpy as np
import sys
import random
import math

class _netG(nn.Module):
    def __init__(self, opt, smc_config):
        super(_netG, self).__init__()
        self.ngpu = opt.ngpu
        self.smc_config = smc_config

        self.CONV1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4,padding=1, bias=False)
        self.CONV2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, padding=1, bias=False)
        self.CONV3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=1, bias=False)
        self.CONV4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=1, bias=False)
        self.CONV5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, padding=1, bias=False)
        self.CONV6 = nn.Conv2d(in_channels=512, out_channels=4000, kernel_size=4, bias=False)
        self.CONV5_TRANSPOSE= nn.ConvTranspose2d(in_channels=4000, out_channels=64*8, kernel_size=4, padding=1,bias=False)
        self.CONV4_TRANSPOSE =nn.ConvTranspose2d(in_channels=64*8, out_channels=64 * 4, kernel_size=4, padding=1,bias=False)
        self.CONV3_TRANSPOSE = nn.ConvTranspose2d(in_channels=64 * 4, out_channels=64 * 2, kernel_size=4, padding=1,bias=False)
        self.CONV2_TRANSPOSE = nn.ConvTranspose2d(in_channels=64 * 2, out_channels=64, kernel_size=4, padding=1,bias=False)
        self.CONV1_TRANSPOSE = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, padding=1, bias=False)
       
        self.relu = nn.ReLU(True)

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, dilation=1, return_indices=True)
        self.main = nn.Sequential(
            nn.Conv2d(3,64,4,1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(3,64,4,2,1, bias=False),
            nn.BatchNorm2d(opt.nef),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(3,64*2,4,2,1, bias=False),
            nn.BatchNorm2d(opt.nef*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(3*2,64*4,4,2,1, bias=False),
            nn.BatchNorm2d(opt.nef*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(3*4,64*8,4,2,1, bias=False),
            nn.BatchNorm2d(opt.nef*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(3*8,opt.nBottleneck,4, bias=False),
            nn.BatchNorm2d(opt.nBottleneck),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(4000, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(opt.ngf, opt.nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

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

        if(flatterned_share1.shape[0] == 2523):
            flatterned_share1 = torch.cat((flatterned_share1, torch.tensor([0.001])), 0)
            flatterned_share2 = torch.cat((flatterned_share1, torch.tensor([0.001])), 0)
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

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

    def secure_forward(self, input, share1, share2):
        cov1 = self.CONV1(input)
        cov1_share1 = self.CONV1(share1)
        cov1_share2 = self.CONV1(share2)
        diff1 = cov1 - (cov1_share1 + cov1_share2)
        
        relu1 = self.relu(cov1)
        relu1_share1 = self.relu(cov1_share1)
        relu1_share2 = self.relu(cov1_share2)
        diff1_relu = relu1 - (relu1_share1 + relu1_share2)

        cov2 = self.CONV2(relu1)
        cov2_share1 = self.CONV2(relu1_share1)
        cov2_share2 = self.CONV2(relu1_share2)
        diff2 = cov2 - (cov2_share1 + cov2_share2)
        
        norm1 = self.batch_norm(cov2)
        norm1_share1, norm1_share2, norm1 = self.sec_batchnom(cov2_share1, cov2_share2)
        diff2_norm = norm1 - (norm1_share1 + norm1_share2)

        relu2 = self.relu(norm1)
        relu2_share1 = self.relu(norm1_share1)
        relu2_share2 = self.relu(norm1_share2)
        diff2_relu = relu1 - (relu1_share1 + relu1_share2)

                           
        cov3 = self.CONV3(relu2)
        cov3_share1 = self.CONV3(relu2_share1)
        cov3_share2 = self.CONV3(relu2_share2)
        diff3 = cov3 - (cov3_share1 + cov3_share2)

        norm3 = self.batch_norm(cov3)
        norm3_share1, norm3_share2, norm3 = self.sec_batchnom(cov3_share1, cov3_share2)
        diff3_norm = norm3 - (norm3_share1 + norm3_share2)

        relu3 = self.relu(norm3)
        relu3_share1 = self.relu(norm3_share1)
        relu3_share2 = self.relu(norm3_share2)
        diff3_relu = relu3 - (relu3_share1 + relu3_share2)

                           
        cov4 = self.CONV4(relu3)
        cov4_share1 = self.CONV4(relu3_share1)
        cov4_share2 = self.CONV4(relu3_share2)
        diff4 = cov4 - (cov4_share1 + cov4_share2)

        norm4 = self.batch_norm(cov4)
        norm4_share1, norm4_share2, norm4 = self.sec_batchnom(cov4_share1, cov4_share2)
        diff4_norm = norm4 - (norm4_share1 + norm4_share2)

        relu4 = self.relu(norm4)
        relu4_share1 = self.relu(norm4)
        relu4_share2 = self.relu(norm4)
        diff4_relu = relu4 - (relu4_share1 + relu4_share2)

                           
        cov5 = self.CONV5(relu4)
        cov5_share1 = self.CONV5(relu4_share1)
        cov5_share2 = self.CONV5(relu4_share2)
        diff5 = cov5 - (cov5_share1 + cov5_share2)

        norm5 = self.batch_norm(cov5)
        norm5_share1, norm5_share2, norm5 = self.sec_batchnom(cov5_share1, cov5_share2)
        diff5_norm = norm5 - (norm5_share1 + norm5_share2)

        relu5 = self.relu(norm5)
        relu5_share1 = self.relu(norm5_share1)
        relu5_share2 = self.relu(norm5_share2)
        diff5_relu = relu5 - (relu5_share1 + relu5_share2)

                           
        cov6 = self.CONV6(relu5)
        cov6_share1 = self.CONV6(relu5_share1)
        cov6_share2 = self.CONV6(relu5_share2)
        diff6 = cov6 - (cov6_share1 + cov6_share2)

        norm6 = self.batch_norm(cov6)
        norm6_share1, norm6_share2, norm6 = self.sec_batchnom(cov6_share1, cov6_share2)
        diff6_norm = norm6 - (norm6_share1 + norm6_share2)

        relu6 = self.relu(norm6)
        relu6_share1 = self.relu(norm6_share1)
        relu6_share2 = self.relu(norm6_share2)
        diff6_relu = relu6 - (relu6_share1 + relu6_share2)
        
                           
        cov_tran_5 = self.CONV5_TRANSPOSE(relu6)
        cov_tran_5_share1 = self.CONV5_TRANSPOSE(relu6_share1)
        cov_tran_5_share2 = self.CONV5_TRANSPOSE(relu6_share2)
        diff5 = cov_tran_5 - (cov_tran_5_share1 + cov_tran_5_share2)

        norm_tran_5 = self.batch_norm(cov_tran_5)
        norm_tran_5_share1, norm_tran_5_share2, norm_tran_5 = self.sec_batchnom(cov_tran_5_share1, cov_tran_5_share2)
        diff5_norm_tran = norm_tran_5 - (norm_tran_5_share1 + norm_tran_5_share2)

        relu5 = self.relu(norm_tran_5)
        relu5_share1 = self.relu(norm_tran_5_share1)
        relu5_share2 = self.relu(norm_tran_5_share2)
        diff5_relu = relu5 - (relu5_share1 + relu5_share2)

                           
        cov_tran_4 = self.CONV4_TRANSPOSE(relu5)
        cov_tran_4_share1 = self.CONV4_TRANSPOSE(relu5_share1)
        cov_tran_4_share2 = self.CONV4_TRANSPOSE(relu5_share2)
        diff5 = cov_tran_4 - (cov_tran_4_share1 + cov_tran_4_share2)

        norm_tran_4 = self.batch_norm(cov_tran_4)
        norm_tran_4_share1, norm_tran_4_share2, norm_tran_4 = self.sec_batchnom(cov_tran_4_share1, cov_tran_4_share2)
        diff4_norm_tran = norm_tran_4 - (norm_tran_4_share1 + norm_tran_4_share2)

        relu4 = self.relu(norm_tran_4)
        relu4_share1 = self.relu(norm_tran_4_share1)
        relu4_share2 = self.relu(norm_tran_4_share2)
        diff4_relu = relu4 - (relu4_share1 + relu4_share2)

                           
        cov_tran_3 = self.CONV3_TRANSPOSE(relu4)
        cov_tran_3_share1 = self.CONV3_TRANSPOSE(relu4_share1)
        cov_tran_3_share2 = self.CONV3_TRANSPOSE(relu4_share2)
        diff_tran_3 = cov_tran_3 - (cov_tran_3_share1 + cov_tran_3_share2)

        norm_tran_3 = self.batch_norm(cov_tran_3)
        norm_tran_3_share1, norm_tran_3_share2, norm_tran_3 = self.sec_batchnom(cov_tran_3_share1, cov_tran_3_share2)
        diff3_norm_tran = norm_tran_3 - (norm_tran_3_share1 + norm_tran_3_share2)

        relu3 = self.relu(norm_tran_3)
        relu3_share1 = self.relu(norm_tran_3_share1)
        relu3_share2 = self.relu(norm_tran_3_share2)
        diff3_relu_tran = relu3 - (relu3_share1 + relu3_share2)

                           
        cov_tran_2 = self.CONV2_TRANSPOSE(relu3)
        cov_tran_2_share1 = self.CONV2_TRANSPOSE(relu3_share1)
        cov_tran_2_share2 = self.CONV2_TRANSPOSE(relu3_share2)
        diff_tran_2 = cov_tran_2 - (cov_tran_2_share1 + cov_tran_2_share2)

        norm_tran_2 = self.batch_norm(cov_tran_2)
        norm_tran_2_share1, norm_tran_2_share2, norm_tran_2 = self.sec_batchnom(cov_tran_2_share1, cov_tran_2_share2)
        diff2_norm_tran = norm_tran_2 - (norm_tran_2_share1 + norm_tran_2_share2)

        relu2 = self.relu(norm_tran_2)
        relu2_share1 = self.relu(norm_tran_2_share1)
        relu2_share2 = self.relu(norm_tran_2_share2)
        diff2_relu_tran = relu2 - (relu2_share1 + relu2_share2)

                           
        cov_tran_1 = self.CONV1_TRANSPOSE(relu2)
        cov_tran_1_share1 = self.CONV1_TRANSPOSE(relu2_share1)
        cov_tran_1_share2 = self.CONV1_TRANSPOSE(relu2_share2)
        diff_tran_1 = cov_tran_1 - (cov_tran_1_share1 + cov_tran_1_share2)

        norm_tran_1 = self.batch_norm(cov_tran_1)
        norm_tran_1_share1, norm_tran_1_share2, norm_tran_1 = self.sec_batchnom(cov_tran_1_share1, cov_tran_1_share2)
        diff2_norm_tran = norm_tran_1 - (norm_tran_1_share1 + norm_tran_1_share2)

        relu1 = self.relu(norm_tran_1)
        relu1_share1 = self.relu(norm_tran_1_share1)
        relu1_share2 = self.relu(norm_tran_1_share2)
        diff1_relu_tran = relu1 - (relu1_share1 + relu1_share2)

        return relu1_share1, relu1_share2


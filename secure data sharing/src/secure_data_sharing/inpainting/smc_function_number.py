import numpy as np
import random as random

class smc_functions(object):
    def __init__(self, config):
        self.bit_width = int(config["bit_width"])
        self.modulus_bits = int(config["modulus_bits"])
        self.share_min = int(config["share_min"])
        self.share_max = int(config["share_max"])
        self.modulus = (1 << self.modulus_bits) - 1
        self.signed_threshold = 1 << (self.modulus_bits - 1)
        self.pooling_kernel = int(config["pooling_kernel"])


    def SecBitMul(self, x_1, x_2, y_1, y_2):                   
        l = self.bit_width

        a_1 = np.random.randint(self.share_min, self.share_max + 1)
        a = np.random.randint(self.share_min, self.share_max + 1)
        b_1 = np.random.randint(self.share_min, self.share_max + 1)
        b = np.random.randint(self.share_min, self.share_max + 1)
        a_2 = a_1 ^ a
        b_2 = b_1 ^ b

        c = a & b                              
        c_1 = np.random.randint(self.share_min, self.share_max + 1)
        c_2 = c ^ c_1

        alpha_1 = x_1 ^ a_1          
        alpha_2 = x_2 ^ a_2
        alpha = alpha_1 ^ alpha_2
        beta_1 = y_1 ^ b_1
        beta_2 = y_2 ^ b_2
        beta = beta_1 ^ beta_2

        f_1 = c_1 ^ (b_1 & alpha) ^ (a_1 & beta)
        f_2 = c_2 ^ (b_2 & alpha) ^ (a_2 & beta) ^ (alpha & beta)

        return f_1, f_2                 

    def SecBitAdd(self, x_1, x_2, y_1, y_2):           
        u_1 = x_1 ^ y_1                                 
        u_2 = x_2 ^ y_2
        c_1, c_2 = self.SecBitMul(x_1, x_2, y_1, y_2)                                 
        c_1 = c_1 << 1
        c_2 = c_2 << 1

        while (c_1 ^ c_2) != 0:
            t_1 = u_1 ^ c_1
            t_2 = u_2 ^ c_2
            c_1, c_2 = self.SecBitMul(u_1, u_2, c_1, c_2)
            c_1 = c_1 << 1 & self.modulus
            c_2 = c_2 << 1 & self.modulus
            u_1 = t_1 & self.modulus
            u_2 = t_2 & self.modulus

        if u_1 > self.signed_threshold - 1:
            u_1 = ~(u_1 ^ self.modulus)
        if u_2 > self.signed_threshold - 1:
            u_2 = ~(u_2 ^ self.modulus)

        return u_1, u_2

    def SecBitExtra(self, u_1, u_2):
        l = self.bit_width          

        r_1 = np.random.randint(self.share_min, self.share_max + 1)
        r_2 = np.random.randint(self.share_min, self.share_max + 1)
        r = r_1 ^ r_2      

                 

        s_1 = np.random.randint(self.share_min, self.share_max + 1)
        s_2 = r - s_1             

        v = (u_1 - s_1) + (u_2 - s_2)      

        if v < 0:      
            v = round(v - 0.5)
        v = round(v)

        v_1 = np.random.randint(self.share_min, self.share_max + 1)
        v_2 = v ^ v_1

                                              

        uu_1, uu_2 = self.SecBitAdd(v_1, v_2, r_1, r_2)

        MSB_1 = 0
        MSB_2 = 0

        if uu_1 < 0:
            MSB_1 = 1
        if uu_2 < 0:
            MSB_2 = 1

        return uu_1, uu_2, MSB_1, MSB_2

                              
    def SecMaxIndex(self, X, Y):                            
        x, y = 0, 0
        if ((X[x, y].item() - X[0, 1].item()) + (Y[x, y].item() - Y[0, 1].item())) < 0:
            x, y = 0, 1
        if ((X[x, y].item() - X[1, 0].item()) + (Y[x, y].item() - Y[1, 0].item())) < 0:
            x, y = 1, 0
        if ((X[x, y].item() - X[1, 1].item()) + (Y[x, y].item() - Y[1, 1].item())) < 0:
            x, y = 1, 1
        return x, y


                                           

u_1, u_2 = -2.01, 2.1
uu_1, uu_2, MSB_1, MSB_2 = smc.SecBitExtra(u_1, u_2)
print(MSB_1 ^ MSB_2)
u_test = (uu_1 ^ uu_2)
u_real = (u_1 + u_2)

a = np.array([[-1.78, 0.28], [3.27, -4.15]])
b = np.array([[0.7839, 1.3234], [-2.123, -4.345654]])
x, y = smc.SecMaxIndex(a, b)

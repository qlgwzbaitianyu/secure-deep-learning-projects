import numpy as np
import random as random


class smc_functions(object):

    def SecBitExtra(self, U_1, U_2):  # 比较(u_1+u_2)与0的大小，输入是十进制整数原码,{-2^(l-1),2^(l-1)-1)
        l = 8  # 二进制位长度
        sh = U_1.shape  # 矩阵大小

        R_1 = np.random.randint(0, 2 ** (l - 1), sh)
        R_2 = np.random.randint(0, 2 ** (l - 1), sh)
        R = R_1 ^ R_2  # 原码=补码，补码参与计算

        S_1 = np.random.randint(0, 2 ** (l - 1), sh)
        S_2 = R - S_1

        V = (U_1 - S_1) + (U_2 - S_2)  # 补码
        V = np.rint(V - 0.5)  # 取整
        V = V.astype(int)
        V_1 = np.random.randint(0, 2 ** (l - 1), sh)
        V_2 = V ^ V_1

        uu_1, uu_2 = self.SecBitAdd(V_1, V_2, R_1, R_2)  # 得到补码结果，符号位与原码相同
        MSB_1 = uu_1.copy()
        MSB_1[MSB_1 > 0] = 0  # 大于0置0，小于0置1
        MSB_1[MSB_1 < 0] = 1
        MSB_2 = uu_2.copy()
        MSB_2[MSB_2 > 0] = 0
        MSB_2[MSB_2 < 0] = 1

        return MSB_1, MSB_2

    def SecBitAdd(self, x_1, x_2, y_1, y_2):  # 二进制进位加法
        u_1 = x_1 ^ y_1  # 异或u_1^u_2=(x_1^x_2)+(y_1^y_2)
        u_2 = x_2 ^ y_2
        c_1, c_2 = self.SecBitMul(x_1, x_2, y_1, y_2)  # 进位c_1^c_2=(x_1^x_2)&(y_1^y_2)
        c_1 = c_1 << 1
        c_2 = c_2 << 1

        while (c_1 ^ c_2).any() != 0:  # 以矩阵为整体，共同进退，只要有一个c不为0，全部进行移位运算。
            t_1 = u_1 ^ c_1
            t_2 = u_2 ^ c_2
            c_1, c_2 = SecBitMul(u_1, u_2, c_1, c_2)
            c_1 = c_1 << 1 & 0xFFFF
            c_2 = c_2 << 1 & 0xFFFF
            u_1 = t_1 & 0xFFFF
            u_2 = t_2 & 0xFFFF

        u_1 = np.where(u_1 > 0x7FFF, ~(u_1 ^ 0xFFFF), u_1)
        u_2 = np.where(u_2 > 0x7FFF, ~(u_2 ^ 0xFFFF), u_2)

        return u_1, u_2

    def SecBitMul(self, x_1, x_2, y_1, y_2):  # 乘法（二进制） 输入需要为数组
        l = 8
        sh = x_1.shape  # 矩阵大小

        a_1 = np.random.randint(0, 2 ** (l - 1), sh)
        a = np.random.randint(0, 2 ** (l - 1), sh)
        b_1 = np.random.randint(0, 2 ** (l - 1), sh)
        b = np.random.randint(0, 2 ** (l - 1), sh)
        a_2 = a_1 ^ a
        b_2 = b_1 ^ b
        c = a & b  # c=a*b   ，c=c_1+c_2,  c_i>0
        c_1 = np.random.randint(0, 2 ** (l - 1), sh)
        c_2 = c ^ c_1

        alpha_1 = x_1 ^ a_1  # 补码参与运算
        alpha_2 = x_2 ^ a_2
        alpha = alpha_1 ^ alpha_2
        beta_1 = y_1 ^ b_1
        beta_2 = y_2 ^ b_2
        beta = beta_1 ^ beta_2

        f_1 = c_1 ^ (b_1 & alpha) ^ (a_1 & beta)
        f_2 = c_2 ^ (b_2 & alpha) ^ (a_2 & beta) ^ (alpha & beta)

        return f_1, f_2

    def SecMaxIndex(self, X, Y):  # M代表子图像1，N代表子图像2，池化区域为2×2
        x, y = 0, 0
        if ((X[x, y] - X[0, 1]) + (Y[x, y] - Y[0, 1])) < 0:
            x, y = 0, 1
        if ((X[x, y] - X[1, 0]) + (Y[x, y] - Y[1, 0])) < 0:
            x, y = 1, 0
        if ((X[x, y] - X[1, 1]) + (Y[x, y] - Y[1, 1])) < 0:
            x, y = 1, 1
        return x, y
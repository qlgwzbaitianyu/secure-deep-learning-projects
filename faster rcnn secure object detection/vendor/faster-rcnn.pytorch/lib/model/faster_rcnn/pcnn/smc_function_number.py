import numpy as np
import random as random


class smc_functions(object):

    def SecBitMul(self, x_1, x_2, y_1, y_2):  # 乘法（二进制） 输入需要为数组
        l = 8

        a_1 = np.random.randint(0, 2 ** (l - 1))
        a = np.random.randint(0, 2 ** (l - 1))
        b_1 = np.random.randint(0, 2 ** (l - 1))
        b = np.random.randint(0, 2 ** (l - 1))
        a_2 = a_1 ^ a
        b_2 = b_1 ^ b

        c = a & b  # c=a*b   ，c=c_1+c_2,  c_i>0
        c_1 = np.random.randint(0, 2 ** (l - 1))
        c_2 = c ^ c_1

        alpha_1 = x_1 ^ a_1  # 补码参与运算
        alpha_2 = x_2 ^ a_2
        alpha = alpha_1 ^ alpha_2
        beta_1 = y_1 ^ b_1
        beta_2 = y_2 ^ b_2
        beta = beta_1 ^ beta_2

        f_1 = c_1 ^ (b_1 & alpha) ^ (a_1 & beta)
        f_2 = c_2 ^ (b_2 & alpha) ^ (a_2 & beta) ^ (alpha & beta)

        return f_1, f_2  # f_1异或f_2,得到原码

    def SecBitAdd(self, x_1, x_2, y_1, y_2):  # 二进制进位加法
        u_1 = x_1 ^ y_1  # 异或u_1^u_2=(x_1^x_2)+(y_1^y_2)
        u_2 = x_2 ^ y_2
        c_1, c_2 = self.SecBitMul(x_1, x_2, y_1, y_2)  # 进位c_1^c_2=(x_1^x_2)&(y_1^y_2)
        c_1 = c_1 << 1
        c_2 = c_2 << 1

        while (c_1 ^ c_2) != 0:
            t_1 = u_1 ^ c_1
            t_2 = u_2 ^ c_2
            c_1, c_2 = self.SecBitMul(u_1, u_2, c_1, c_2)
            c_1 = c_1 << 1 & 0xFFFF
            c_2 = c_2 << 1 & 0xFFFF
            u_1 = t_1 & 0xFFFF
            u_2 = t_2 & 0xFFFF

        if u_1 > 0x7FFF:
            u_1 = ~(u_1 ^ 0xFFFF)
        if u_2 > 0x7FFF:
            u_2 = ~(u_2 ^ 0xFFFF)

        return u_1, u_2

    def SecBitExtra(self, u_1, u_2):  # 比较(u_1+u_2)与0的大小，输入是十进制整数原码,{-2^(l-1),2^(l-1)-1)
        l = 8  # 二进制位长度

        r_1 = np.random.randint(0, 2 ** (l - 1))
        r_2 = np.random.randint(0, 2 ** (l - 1))
        r = r_1 ^ r_2  # 原码

        # if r<0:
        #    r=2**16+r  #补码

        s_1 = np.random.randint(0, 2 ** (l - 1))
        s_2 = r - s_1  # r=s_1+s_2

        v = (u_1 - s_1) + (u_2 - s_2)  # 补码

        if v < 0:  # 近似
            v = round(v - 0.5)
        v = round(v)

        v_1 = np.random.randint(0, 2 ** (l - 1))
        v_2 = v ^ v_1

        # uu=(v_1 ^ v_2) + (r_1 ^ r_2)   #real

        uu_1, uu_2 = self.SecBitAdd(v_1, v_2, r_1, r_2)

        MSB_1 = 0
        MSB_2 = 0

        if uu_1 < 0:
            MSB_1 = 1
        if uu_2 < 0:
            MSB_2 = 1

        return uu_1, uu_2, MSB_1, MSB_2

    # 返回一个池化区域内最大值的索引（映射至原始图像）
    def SecMaxIndex(self, X, Y):  # M代表子图像1，N代表子图像2，池化区域为2×2
        x, y = 0, 0
        if ((X[x, y] - X[0, 1]) + (Y[x, y] - Y[0, 1])) < 0:
            x, y = 0, 1
        if ((X[x, y] - X[1, 0]) + (Y[x, y] - Y[1, 0])) < 0:
            x, y = 1, 0
        if ((X[x, y] - X[1, 1]) + (Y[x, y] - Y[1, 1])) < 0:
            x, y = 1, 1
        return x, y


smc = smc_functions()
# x_1, x_2, y_1, y_2=79,-123,2,52   #u_real=0
# x_1, x_2, y_1, y_2=102,-88,64,115   #u_real=1
# x_1, x_2, y_1, y_2=83,-74,108,113   #u_real=2
# x_1, x_2, y_1, y_2=int(83),int(-74),int(108),int(113)   #u_real=2
# x_1, x_2, y_1, y_2=11,6,31,17

# u_1,u_2=smc.SecBitAdd(x_1, x_2, y_1, y_2)
# xx=(x_1^ x_2)
# yy=(y_1^ y_2)
# u_real=xx+yy
# u_test=u_1^u_2

u_1, u_2 = -2, 1.8
uu_1, uu_2, MSB_1, MSB_2 = smc.SecBitExtra(u_1, u_2)
print(MSB_1 ^ MSB_2)
u_test = (uu_1 ^ uu_2)
u_real = (u_1 + u_2)
print(u_test)
print(u_real)
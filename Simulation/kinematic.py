import numpy as np
import matplotlib.pyplot as plt
import math


class ode_RK_4:
    def __init__(self, func, T_span, Y0, Nt):
        """
        初始化函数
        :param func:需要计算的导函数即f(x,y)
        :param T_span: x的范围
        :param Y0: Y的初始值
        :param Nt: 需要分割的个数
        """
        self.func = func
        self.T = T_span
        self.Y0 = np.array(Y0)
        self.Nt = Nt
        self.t_array = np.linspace(T_span[0], T_span[1], Nt)
        self.y_array = np.zeros((len(self.Y0), Nt))
        self.y_array[:, 0] = Y0
        self.h = (T_span[1] - T_span[0]) / (Nt - 1)

    def main(self):
        """
        龙格库塔计算函数
        :return: 返回t的列表与y的列表
        """
        for i in range(0, self.Nt - 1):
            # NOTE:不能调用类中的函数，会出bug，调用类外的函数
            K1 = self.func(self.t_array[i], self.y_array[:, i])
            K2 = self.func(self.t_array[i] + self.h / 2,
                           self.y_array[:, i] + self.h / 2 * K1)
            K3 = self.func(self.t_array[i] + self.h / 2,
                           self.y_array[:, i] + self.h / 2 * K2)
            K4 = self.func(self.t_array[i] + self.h,
                           self.y_array[:, i] + self.h * K3)
            self.y_array[:, i + 1] = self.y_array[:, i] + \
                self.h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
        return self.t_array, self.y_array


if __name__ == '__main__':
    def func(t, Y, Nx=0, Nz=-20, phi=0, g=9.8):
        '''
        Nx 切向过载
        Nz 法向过载
        phi 滚转角
        '''
        v = Y[0]
        psi = Y[1]
        theta = Y[2]
        x = Y[3]
        y = Y[4]
        z = Y[5]
        print(math.sin(phi))
        velocity_func = g*(Nx-math.sin(theta))
        psi_func = (g*Nz*math.sin(phi))/(v*math.cos(theta))
        theta_func = g*(Nz*math.cos(phi)-math.cos(theta))/v
        position_x_func = v*math.cos(theta)*math.cos(psi)
        position_y_func = v*math.cos(theta)*math.sin(psi)
        position_z_func = -v*math.sin(theta)
        return np.array([velocity_func, psi_func, theta_func,
                         position_x_func, position_y_func, position_z_func])

    next_state = ode_RK_4(func, [0, 0.01], [1, 0, math.pi/3, 40, 50, 50], 100)
    t, y = next_state.main()
    fig = plt.figure()
    print('start point: ', y[3, 0], y[4, 0], y[5, 0])
    ax = fig.gca(projection='3d')
    ax.scatter(y[3, 0], y[4, 0], y[5, 0])
    ax.plot(y[3, :], y[4, :], y[5, :])

    # plt.plot(t, y[0,:],lw=2, label="v")
    # plt.plot(t, y[1,:], '-.', label="psi")
    # plt.plot(t, y[2,:], '.', label="theta")
    plt.show()

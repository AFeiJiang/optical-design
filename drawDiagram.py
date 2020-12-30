import numpy as np
import matplotlib.pyplot as plt

class drawDiagram():
    def __init__(self, lensSys):
        self.lensSys = lensSys
        self.L_OEP = 0
        for i in range(self.lensSys.pupilPosition):
            self.L_OEP = self.L_OEP + self.lensSys.Lens[i]['t']

    # 功能：绘制二维透镜面(私有函数)
    def __drawSurf(self, yLast):
        lenH = self.lensSys.pupilRadius
        num_lens = len(self.lensSys.Lens)
        zRange = []
        yRange = []
        z0 = 0
        for nl in range(num_lens):
            C = self.lensSys.Lens[nl]['C']    # 曲率
            # 计算镜面中心在光轴上的位置
            if nl > 0:
                z0 = z0 + self.lensSys.Lens[nl - 1]['t']
            # 计算透镜圆心
            if C == 0:
                surf_y = np.array([-lenH, lenH])
                surf_z =  np.array([1,1]) * z0
            else:
                r = 1 / C    # 计算半径
                (a, b) = (z0 + r, 0)    # 透镜中心
                c = np.arcsin(abs((lenH - b) / r))  # 计算透镜最大的角度
                theta = np.arange(-c, c, 0.01)    # 计算角度范围
                thetas = np.append(theta, c)
                surf_z = a - r * np.cos(thetas)    # 计算透镜面z范围
                surf_y = b + abs(r) * np.sin(thetas)    # 计算透镜面y范围
            # 存储透镜面的坐标
            zRange.append(surf_z)
            yRange.append(surf_y)
            if nl > 0:
                if nl == num_lens - 1 and yLast > lenH / self.lensSys.pupilRadius:
                    plt.plot(np.array(surf_z) / self.lensSys.pupilRadius, np.array([-yLast, yLast]), 'k')
                else:
                    plt.plot(np.array(surf_z) / self.lensSys.pupilRadius, np.array(surf_y) / self.lensSys.pupilRadius,
                             'k')
            # 绘制透镜上下表面
            if nl > 0 and nl % 2 == 0:
                plt.plot([zRange[nl-1][-1] / self.lensSys.pupilRadius, surf_z[-1] / self.lensSys.pupilRadius],
                             [lenH / self.lensSys.pupilRadius, lenH / self.lensSys.pupilRadius], 'k')
                plt.plot([zRange[nl-1][-1] / self.lensSys.pupilRadius, surf_z[-1] / self.lensSys.pupilRadius],
                             [-lenH / self.lensSys.pupilRadius, -lenH / self.lensSys.pupilRadius], 'k')

    # 绘制光线示意图
    # 输入：theta:    光束角度
    #      num_rays:    光线个数
    # 输出：光线追迹示意图
    def drawRayTrace(self, theta, num_rays, color):
        num_lens = self.lensSys.num_lens    # 透镜面的个数
        theta = np.array(theta) * np.pi / 180       # 角度转弧度
        # 光线采样
        ray_x = np.zeros(shape = num_rays)
        ray_y = np.linspace(-self.lensSys.pupilRadius / 2 - self.L_OEP * np.tan(theta),
                            self.lensSys.pupilRadius / 2 - self.L_OEP * np.tan(theta), num_rays)
        ray_z = np.zeros(shape = num_rays)
        ray_X = np.zeros(shape = num_rays)
        ray_Y = np.ones(shape = num_rays) * np.sin(theta)
        ray_Z = np.ones(shape = num_rays) * np.cos(theta)
        ray = {'x': ray_x, 'y': ray_y, 'z': ray_z, 'X': ray_X, 'Y': ray_Y, 'Z': ray_Z}
        rays = self.lensSys.SkewRayTrace(ray)
        # 1. 生成坐标数组
        # 1) 将光线在每个透镜面上的坐标提取出来
        y = np.zeros(shape = (num_lens, num_rays))
        z = np.zeros(shape = (num_lens, num_rays))
        for i in range(len(self.lensSys.Lens)):
            for j in range(num_rays):
                y[i,j] = rays[i]['y'][j]
                z[i,j] = rays[i]['z'][j]
        # 2) 将其修改为在光轴坐标系中的坐标
        t = np.array([self.lensSys.Lens[i]['t'] for i in range(num_lens)])
        for i in range(num_lens):
            if i > 0:
                t[i] = t[i] + t[i-1]
        S = np.zeros(shape = num_lens)
        S[1:num_lens] = t[:num_lens-1]
        z = z + np.array([S] * num_rays).T
        # 归一化
        y = y / self.lensSys.pupilRadius
        z = z / self.lensSys.pupilRadius
        # 2. 绘制光线
        for i in range(num_rays):
            plt.plot(z[1:num_lens, i], y[1:num_lens, i], color)
        # 3. 绘制透镜
        yLast = y.max()
        self.__drawSurf(yLast)
import math
from Material import *
import numpy as np

class LensSys():
    def __init__(self, OriginalLens, pupilRadius, pupiltheta, pupilPosition, wavelength):
        self.OriginalLens = OriginalLens  # 透镜参数
        self.pupilRadius = pupilRadius  # 入瞳半径
        self.pupiltheta = pupiltheta  # 最大视场角
        self.pupilPosition = pupilPosition  # 入瞳位置
        self.num_lens = len(self.OriginalLens)

        global L_OEP
        L_OEP = 0
        for i in range(self.pupilPosition):
            L_OEP = L_OEP + self.OriginalLens[i]['t']

    # 所要实现的功能代码
    # 功能: 计算光学系统焦点位置，确定光学系统理想成像面的位置
    def calBfl(self):
        ray = {'u': 0, 'y': self.pupilRadius}    # 输入光线在第一个透镜面上的入射角u，以及入射高度y
        for i in range(len(self.OriginalLens)-1):
            
        # 计算光学系统的后截距
        bfl = -(axial_rays[self.num_lens - 3]['y'] / axial_rays[self.num_lens - 2]['u'])
        self.Lens[self.num_lens - 2]['t'] = bfl

    # 功能: 计算入瞳中心光线在成像面上的坐标
    # 输入: theta: 视场角
    # 输出: 主光线在像面上的坐标
    def centralLocationInImage(self, thetas):
        thetas = thetas * math.pi / 180    # 角度转为弧度
        # 进行光线追迹
        ray = {'x': 0, 'y': -L_OEP * np.tan(thetas), 'z': 0, 'X': 0, 'Y': np.sin(thetas), 'Z': np.cos(thetas)}
        rays = self.SkewRayTrace(ray)
        return rays[self.num_lens - 1]['x'], rays[self.num_lens - 1]['y']    # 返回入瞳中心光线在成像面上的坐标

    # 计算一个视场角下光线的横向像差
    # 输入: theta    视场角
    #      apertureRays    光束孔径
    # 输出: x    横坐标
    #      y    纵坐标
    def lateralAberration(self, thetas, apertureRays):
        # 1. 初始化
        # 1) 光束信息, 包括：角度thetas，方向矢量[X, Y, Z]
        num_thetas = len(thetas)
        thetas = thetas * math.pi / 180    # 角度转弧度
        X = np.zeros(np.shape(thetas));    Y = np.sin(thetas);    Z = np.cos(thetas)    # 光线的方向矢量
        # 2)极坐标下的采样规则
        num_phi = 8    # pi的采样间隔数
        num_h = 21     # 孔径直径采样
        # 2. 对光线进行采样
        phi = np.arange(0, math.pi, math.pi/num_phi)
        thetas, phi = np.meshgrid(thetas, phi)
        min_y = -apertureRays * np.sin(phi) - L_OEP * np.tan(thetas)
        max_y = apertureRays * np.sin(phi) - L_OEP * np.tan(thetas)
        min_x = -apertureRays * np.cos(phi)
        max_x = apertureRays * np.cos(phi)
        # 3. 计算光线的初始信息
        ray_x = np.zeros(shape=(num_h, num_phi, num_thetas))
        for i in range(num_phi):
            for j in range(num_thetas):
                ray_x[:, i, j] = np.linspace(min_x[i,j], max_x[i,j], num_h)
        ray_y = np.zeros(shape=(num_h, num_phi, num_thetas))
        for i in range(num_phi):
            for j in range(num_thetas):
                ray_y[:, i, j] = np.linspace(min_y[i, j], max_y[i, j], num_h)
        ray_z = np.zeros(np.shape(ray_x))
        ray_X = np.ones(np.shape(ray_x))
        ray_Y = np.ones(np.shape(ray_x))
        ray_Z = np.ones(np.shape(ray_x))
        for i in range(len(X)):
            ray_X[:, :, i] = ray_X[:, :, i] * X[i]
            ray_Y[:, :, i] = ray_Y[:, :, i] * Y[i]
            ray_Z[:, :, i] = ray_Z[:, :, i] * Z[i]
        # 4. 对光线进行追迹
        ray = {'x': ray_x, 'y': ray_y, 'z': ray_z, 'X': ray_X, 'Y': ray_Y, 'Z': ray_Z}
        rays = self.SkewRayTrace(ray)
        # 5. 保存光线在像面上的成像点坐标
        x = rays[self.num_lens - 1]['x']    # 保存光线在像面上的横坐标
        y = rays[self.num_lens - 1]['y']    # 保存光线在像面上的纵坐标
        return x, y

    # 功能: 计算一个视场角下的径向像差
    # 输入: thetas   视场角
    #      aperturRays    孔径大小
    # 输出: z_yc_error    x固定，y变化时的径向误差
    #      z_xc_error    y固定，x变化时的径向误差
    def longitudalAberration(self, thetas, apertureRays):
        # 1. 初始化
        thetas = thetas * math.pi / 180    # 角度转弧度
        # 光束的方向矢量
        X = np.zeros(np.shape(thetas))
        Y = np.sin(thetas)
        Z = np.cos(thetas)
        #
        errorRange = 2    # 误差范围
        interval = 0.01    # 采样间隔
        num_imgs_half = int(errorRange / interval)    # 一半的采样个数
        # 计算成像面的位置
        imgs = np.array(range(-num_imgs_half, num_imgs_half)) * interval + self.Lens[self.num_lens - 2]['t']
        num_imgs = len(imgs)    # 计算成像面的个数

        # 2. 计算主光线在不同像面上的成像点
        # 1) 计算主光线在最后一个透镜面上的点坐标，以及方向矢量
        # 主光线的初始信息
        ray_x = np.zeros(np.shape(thetas))
        ray_y = -L_OEP * np.tan(thetas)
        ray_z = np.zeros(np.shape(thetas))
        # 对主光线进行追迹，追迹到最后一个透镜面
        ray = {'x': ray_x, 'y': ray_y, 'z': ray_z, 'X': X, 'Y': Y, 'Z': Z}
        rays = self.SkewRayTrace(ray, 0)
        # 2) 计算入瞳中心光线在像面上的点坐标21个角度，400个像面
        lastLen_x = np.array([rays[len(rays) - 1]['x']] * num_imgs).T
        lastLen_y = np.array([rays[len(rays) - 1]['y']] * num_imgs).T
        lastLen_z = np.array([rays[len(rays) - 1]['z']] * num_imgs).T
        lastLen_X = np.array([rays[len(rays) - 1]['X']] * num_imgs).T
        lastLen_Y = np.array([rays[len(rays) - 1]['Y']] * num_imgs).T
        lastLen_Z = np.array([rays[len(rays) - 1]['Z']] * num_imgs).T
        lastLen_t = np.array([imgs]*len(thetas))
        x0_imgs, y0_imgs, z3, X3, Y3, Z3 = self.skew_raytrace(lastLen_x, lastLen_y, lastLen_z, lastLen_X,
                                                              lastLen_Y, lastLen_Z, 1.0, 1.0, lastLen_t, 0)

        # 3. 计算两组光线的径向像差
        # 1) 对两组光线进行采样
        num_h = 11  # 每组采样的光线个数
        # 计算两组光线的取值范围
        min_y = -apertureRays - L_OEP * np.tan(thetas)
        max_y = apertureRays - L_OEP * np.tan(thetas)
        min_x = - np.ones(np.shape(thetas)) * apertureRays
        max_x = np.ones(np.shape(thetas)) * apertureRays
        # 2) 计算光线在所有成像面上的成像点坐标
        # 对光线进行追迹，追迹到最后一个透镜面
        rays_xc = np.zeros(shape=(len(min_x), num_h))
        rays_yc = np.zeros(shape=(len(min_x), num_h))
        for i in range(len(min_x)):
            rays_xc[i, :] = np.linspace(min_x[i], max_x[i], num_h)
            rays_yc[i, :] = np.linspace(min_y[i], max_y[i], num_h)
        ray_x = np.vstack((np.zeros(shape=(len(thetas), num_h)), rays_xc))    # 42*11
        ray_y = np.vstack((rays_yc, -np.array([L_OEP * np.tan(thetas)] * num_h).T))
        ray_z = np.zeros(np.shape(ray_x))
        ray_X = np.hstack((np.array([X] * num_h), np.array([X] * num_h)))    # 11*42
        ray_Y = np.hstack((np.array([Y] * num_h), np.array([Y] * num_h)))
        ray_Z = np.hstack((np.array([Z] * num_h), np.array([Z] * num_h)))
        ray = {'x': ray_x, 'y': ray_y, 'z': ray_z, 'X': ray_X.T, 'Y': ray_Y.T, 'Z': ray_Z.T}
        rays = self.SkewRayTrace(ray, 0)
        # 计算光线在像面上的坐标
        lastLen_x = np.array([rays[len(rays) - 1]['x']] * num_imgs)
        lastLen_y = np.array([rays[len(rays) - 1]['y']] * num_imgs)
        lastLen_z = np.array([rays[len(rays) - 1]['z']] * num_imgs)
        lastLen_X = np.array([rays[len(rays) - 1]['X']] * num_imgs)
        lastLen_Y = np.array([rays[len(rays) - 1]['Y']] * num_imgs)
        lastLen_Z = np.array([rays[len(rays) - 1]['Z']] * num_imgs)
        lastLen_t = np.array([np.array([imgs] * len(thetas) * 2)] * num_h).T
        x3, y3, z3, X3, Y3, Z3 = self.skew_raytrace(lastLen_x, lastLen_y, lastLen_z, lastLen_X,
                                                    lastLen_Y, lastLen_Z, 1.0, 1.0, lastLen_t, 0)

        # 4. 计算径向像差
        # 1) 对入瞳中心像点进行扩展
        x0_imgs = np.array([np.vstack((x0_imgs, x0_imgs))] * num_h).T
        y0_imgs = np.array([np.vstack((y0_imgs, y0_imgs))] * num_h).T
        # 2) 计算每一个面的横向像差
        error = ((x3 - x0_imgs) ** 2 + (y3 - y0_imgs) ** 2).sum(axis=2)
        yc_error = error[:, 0:len(thetas)]
        xc_error = error[:, len(thetas):2*len(thetas)]
        index_min = np.where(yc_error == yc_error.min(axis=0))  # 最小值的索引
        z_yc_error = (index_min[0][np.argsort(index_min[1])] - num_imgs_half) * interval
        index_min = np.where(xc_error == xc_error.min(axis=0))  # 最小值的索引
        z_xc_error = (index_min[0][np.argsort(index_min[1])] - num_imgs_half) * interval

        return z_yc_error, z_xc_error

    # 功能：计算畸变曲线
    # 输入: thetas   视场角
    # 输出: 畸变误差
    def distortion(self, thetas):
        thetas = thetas * np.pi / 180
        # 近轴公式追迹光线
        axial_ray = {'y': -L_OEP * np.tan(thetas), 'u': np.tan(thetas)}
        axial_rays = self.ParaxialRayTrace(axial_ray)
        # 任意斜光线追迹公式追迹光线
        skew_ray = {'x': 0, 'y': -L_OEP * np.tan(thetas), 'z': 0, 'X': 0, 'Y': np.sin(thetas), 'Z': np.cos(thetas)}
        skew_rays = self.SkewRayTrace(skew_ray)
        # 获取光线使用不同追迹公式在像面上的高度
        y1 = skew_rays[self.num_lens - 1]['y']
        y2 = axial_rays[self.num_lens - 1]['y']
        # 计算畸变误差
        error = np.zeros(len(thetas))
        error[y2 != 0] = 100 * (y1[y2 != 0] - y2[y2 != 0]) / y2[y2 != 0]
        return error  # 返回畸变误差

    # 光学计算的基础算法
    # 近轴光线追迹
    # 输入: u1: 入射角
    #      y1: 光线在透镜面上的高度
    #      n1->n2: 折射率n1到折射率n2
    #      C: 透镜面曲率
    #      t: 到下一透镜面的距离
    # 输出: u2: 出射角
    #      y2: 光线在下一透镜面上的高度
    def paraxial_raytrace(self, u1, y1, n1, n2, C, t):
        u2 = (n1 * u1 - y1 * (n2 - n1) * C) / n2
        y2 = y1 + t * u2
        return u2, y2

    # 对任意斜光线进行追迹
    # 输入: [x1, y1, z1]: 光线在前一个面上的坐标
    #      [X1, Y1, Z1]: 光线的入射矢量
    #      n1->n2: 折射率n1->n2
    #      c: 曲率
    # 输出: [x1, y1, z1]: 光线在当前面上的坐标
    #      [X1, Y1, Z1]: 光线在当前面上的出射矢量
    def skew_raytrace(self, x1, y1, z1, X1, Y1, Z1, n1, n2, t, c):
        e = t * Z1 - (x1 * X1 + y1 * Y1 + z1 * Z1)
        M1z = z1 + e * Z1 - t
        M12 = x1 * x1 + y1 * y1 + z1 * z1 - e * e + t * t - 2 * t * z1
        E1 = np.sqrt(Z1 * Z1 - c * (c * M12 - 2 * M1z))
        L = e + (c * M12 - 2 * M1z) / (Z1 + E1)
        z2 = z1 + L * Z1 - t
        y2 = y1 + L * Y1
        x2 = x1 + L * X1
        E2 = np.sqrt(1 - ((n1 / n2) ** 2) * (1 - E1 ** 2))
        g1 = E2 - (n1 / n2) * E1
        Z2 = (n1 / n2) * Z1 - g1 * c * z2 + g1
        Y2 = (n1 / n2) * Y1 - g1 * c * y2
        X2 = (n1 / n2) * X1 - g1 * c * x2
        return x2, y2, z2, X2, Y2, Z2

    # 功能：使用近轴追迹公式对光线进行完整地追迹
    # 输入：ray: 光线
    # 输出：axial_rays: 光线在每一个面上的交点以及出射角
    def ParaxialRayTrace(self, ray):
        axial_rays = []
        axial_rays.append(ray)
        for i in range(1, self.num_lens):
            u2, y2 = self.paraxial_raytrace(axial_rays[i - 1]['u'], axial_rays[i - 1]['y'], self.Lens[i - 1]['n'],
                                            self.Lens[i]['n'], self.Lens[i]['C'], self.Lens[i]['t'])
            axial_rays.append({'u': u2, 'y': y2})
        return axial_rays

    # 功能：使用任意斜光线追迹公式对光线进行完整地追迹
    # 输入：ray: 光线
    #     Oimg: == 1时，说明将光线追迹到像面；== 0时，说明将光线追迹到最后一个透镜面
    # 输出：rays: 光线在每一个面上的交点坐标，以及出射方向矢量
    def SkewRayTrace(self, ray, Oimg=1):
        rays = []
        rays.append(ray)
        if Oimg == 1:
            for i in range(1, self.num_lens):
                x2, y2, z2, X2, Y2, Z2 = self.skew_raytrace(rays[i - 1]['x'], rays[i - 1]['y'], rays[i - 1]['z'],
                                                            rays[i - 1]['X'], rays[i - 1]['Y'], rays[i - 1]['Z'],
                                                            self.Lens[i - 1]['n'], self.Lens[i]['n'],
                                                            self.Lens[i - 1]['t'], self.Lens[i]['C'])
                rays.append({'x': x2, 'y': y2, 'z': z2, 'X': X2, 'Y': Y2, 'Z': Z2})
        else:
            for i in range(1, self.num_lens - 1):
                x2, y2, z2, X2, Y2, Z2 = self.skew_raytrace(rays[i - 1]['x'], rays[i - 1]['y'], rays[i - 1]['z'],
                                                            rays[i - 1]['X'], rays[i - 1]['Y'], rays[i - 1]['Z'],
                                                            self.Lens[i - 1]['n'], self.Lens[i]['n'],
                                                            self.Lens[i - 1]['t'], self.Lens[i]['C'])
                rays.append({'x': x2, 'y': y2, 'z': z2, 'X': X2, 'Y': Y2, 'Z': Z2})
        return rays
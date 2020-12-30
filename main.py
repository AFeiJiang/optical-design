import numpy as np
import opticalSystem
import drawDiagram
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 1. 输入
    # 1) 光学系统透镜信息(曲率半径，厚度，材料)
    num_Lens = 3
    OriginalLens = []
    # 目前是给定的，所以直接写出来，该部分'm'如果为空格，表示透镜的第二个面
    OriginalLens.append({'C': 1.0 / 40.94, 't': 8.74, 'm': 'SSK4A'})    # 将第一个透镜面添加进去
    OriginalLens.append({'C': 0.0, 't': 11.05, 'm': ' '})
    OriginalLens.append({'C': -1.0 / 55.65, 't': 2.78, 'm': 'SF12'})
    OriginalLens.append({'C': 1.0 / 39.75, 't': 7.63, 'm': ' '})
    OriginalLens.append({'C': 1.0 / 107.56, 't': 9.54, 'm': 'SSK4A'})
    OriginalLens.append({'C': -1.0 / 43.33, 't': 0.0, 'm': ' '})
    # 实际中，我们需要根据透镜的个数添加透镜面的信息
    '''
    for nL in range(num_Lens):
        surf1 = {'C': 1.0 / 40.94, 't': 8.74, 'm': 'SSK4A'}    # 根据需求添加
        surf2 = {'C': 0.0, 't': 11.05, 'm': ' '}
    '''
    # 2) 光学系统的基本参数
    pupilRadius = 18.5    # 入瞳孔径大小
    pupiltheta = 20       # 最大视场角
    pupilPosition = 4     # 入瞳位置
    # 3) 光线波长
    wavelength = []              # 建立wavelength列表用来存储波长
    wavelength.append(0.4861)    # 将所需第一个波长的光添加进去
    wavelength.append(0.5876)
    wavelength.append(0.6563)

    # 2. 实例化光学系统
    lensSys = opticalSystem.LensSys(OriginalLens, pupilRadius, pupiltheta, pupilPosition, wavelength)
    lensSys.calBfl()    # 确定光学系统理想成像面的位置
    #

    # 5) 产生不同波长的光学系统
    Lens = []
    for nw in range(len(wavelength)):
        Lens.append({'C': 0.0, 't': 100.0, 'n': 1.0})
        for nO in range(len(OriginalLens)):
            # 找到透镜所对应的材料
            indeM = materials['m'].index(OriginalLens[nO]['m'])    # 寻找所对应的材料的索引
            # 计算透镜折射率

    # 2. 绘制横向像差点列图
    thetasDiagram = np.array([0, 8, 14, 20])    # 定义视场角
    Lateral_apertureRays = pupilRadius    # 光束孔径设置为18.5
    x0, y0 = lensSys.centralLocationInImage(thetasDiagram)    # 每个视场角主光线与像面的交点
    X, Y = lensSys.lateralAberration(thetasDiagram, Lateral_apertureRays)    #
    # Airiy斑
    # 绘制横向像差点列图
    fig = plt.figure()
    for nt in range(len(thetasDiagram)):
        plt.subplot(2, 2, nt + 1)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        Xb = X[:, :, nt].flatten()
        Yb = Y[:, :, nt].flatten()
        plt.scatter(Xb - x0[nt], Yb - y0[nt], marker='+')
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        plt.grid()
        plt.xlabel("z/mm")
        plt.ylabel("y/mm")
    plt.suptitle('Spot diagram')
    plt.show()

    # 3. 绘制径向像差曲线
    Longitude_apertureRays = 2    # 光束孔径设置为2，1/10
    thetasRange = np.array(range(pupiltheta+1))    # 角度范围为0-20
    z_yc, z_xc= lensSys.longitudalAberration(thetasRange, Longitude_apertureRays)
    # 绘制径向像差曲线
    plt.figure()
    plt.xlim(-Longitude_apertureRays, Longitude_apertureRays)    # 范围有问题
    plt.ylim(0, pupiltheta)
    plt.plot(z_yc, thetasRange)
    plt.plot(z_xc, thetasRange, linestyle='--')
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.4))    # 0.4自动化
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))    # 2自动化
    plt.grid()
    plt.title('Longitudal Aberration')
    plt.xlabel('range/mm')
    plt.ylabel('angle/degree')
    plt.show()

    # 4. 绘制畸变曲线
    error = lensSys.distortion(thetasRange)
    # 绘制畸变曲线
    plt.figure()
    plt.xlim(-0.5, 0.5)
    plt.ylim(0, pupiltheta)
    plt.plot(error, thetasRange)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    plt.grid()
    plt.title('Distortion')
    plt.xlabel('percent/%')
    plt.ylabel('angle/degree')
    plt.show()

    # 5. 绘制光线追迹示意图
    colors = ['b', 'lime', 'r', 'y']
    num_rays = 3
    for i in range(len(thetasDiagram)):
        drawRays = drawDiagram.drawDiagram(lensSys)
        drawRays.drawRayTrace(thetasDiagram[i], num_rays, colors[i])
    plt.show()
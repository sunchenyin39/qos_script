import numpy as np
import copy
import matplotlib.pyplot as plt
import fit


def main():
    # test_exp_Fit_1D()
    # test_exp_FitX_2D()
    # test_exp_FitY_2D()
    # test_linear_Fit_1D()
    # test_linear_FitX_2D()
    # test_linear_FitY_2D()
    test_sin_Fit_1D()
    print(1)


def test_exp_Fit_1D():
    # 拟合指数曲线解析式
    def target_func(x, a, b, c):
        return a * np.exp(b*x)+c
    # 模拟生成一组实验数据(加噪声)
    x = np.arange(0, 100, 0.2)
    y = 2.0*np.exp(-x / 51.3)+3.0
    noise = np.random.uniform(0, 0.1, len(x))
    y += noise
    fig, ax = plt.subplots()
    ax.plot(x, y, 'b--')
    # 调用exp_Fit_1D拟合实验数据
    parameters, err = fit.exp_Fit_1D(x, y)
    y_fit = [target_func(a, *parameters) for a in x]
    # 画图
    ax.plot(x, y_fit, 'g')
    plt.show()


def test_exp_FitX_2D():
    # 拟合指数曲线解析式
    def target_func(x, a, b, c):
        return a * np.exp(b*x)+c
    # 模拟生成4组实验数据(加噪声)
    x = np.arange(0, 100, 0.2)
    y0 = 1.0*np.exp(-0.02*x)+2.0
    noise = np.random.uniform(0, 0.1, len(x))
    y0 += noise

    y1 = -2*np.exp(-0.05*x)-1.0
    noise = np.random.uniform(0, 0.1, len(x))
    y1 += noise

    y2 = -3*np.exp(+0.02*x)-1.0
    noise = np.random.uniform(0, 0.1, len(x))
    y2 += noise

    y3 = np.exp(+0.02*x)+3.0
    noise = np.random.uniform(0, 0.1, len(x))
    y3 += noise
    # 调用exp_FitX_2D拟合实验数据
    parameters, err = fit.exp_FitX_2D([x, x, x, x], [y0, y1, y2, y3])
    y0_fit = [target_func(a, *parameters[0]) for a in x]
    y1_fit = [target_func(a, *parameters[1]) for a in x]
    y2_fit = [target_func(a, *parameters[2]) for a in x]
    y3_fit = [target_func(a, *parameters[3]) for a in x]
    # 画图
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(x, y0, 'b--')
    plt.plot(x, y0_fit, 'g')
    plt.subplot(2, 2, 2)
    plt.plot(x, y1, 'b--')
    plt.plot(x, y1_fit, 'g')
    plt.subplot(2, 2, 3)
    plt.plot(x, y2, 'b--')
    plt.plot(x, y2_fit, 'g')
    plt.subplot(2, 2, 4)
    plt.plot(x, y3, 'b--')
    plt.plot(x, y3_fit, 'g')
    plt.show()


def test_exp_FitY_2D():
    # 拟合指数曲线解析式
    def target_func(x, a, b, c):
        return a * np.exp(b*x)+c
    # 模拟生成4组实验数据(加噪声)
    x = np.arange(0, 100, 0.2)
    y0 = 1.0*np.exp(-0.02*x)+2.0
    noise = np.random.uniform(0, 0.1, len(x))
    y0 += noise

    y1 = -2*np.exp(-0.05*x)-1.0
    noise = np.random.uniform(0, 0.1, len(x))
    y1 += noise

    y2 = -3*np.exp(+0.02*x)-1.0
    noise = np.random.uniform(0, 0.1, len(x))
    y2 += noise

    y3 = np.exp(+0.02*x)+3.0
    noise = np.random.uniform(0, 0.1, len(x))
    y3 += noise
    # 调用exp_FitY_2D拟合实验数据
    y_data = np.transpose(np.array([x, x, x, x])).tolist()
    data = np.transpose(np.array([y0, y1, y2, y3])).tolist()
    parameters, err = fit.exp_FitY_2D(y_data, data)
    y0_fit = [target_func(a, *parameters[0]) for a in x]
    y1_fit = [target_func(a, *parameters[1]) for a in x]
    y2_fit = [target_func(a, *parameters[2]) for a in x]
    y3_fit = [target_func(a, *parameters[3]) for a in x]
    # 画图
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(x, y0, 'b--')
    plt.plot(x, y0_fit, 'g')
    plt.subplot(2, 2, 2)
    plt.plot(x, y1, 'b--')
    plt.plot(x, y1_fit, 'g')
    plt.subplot(2, 2, 3)
    plt.plot(x, y2, 'b--')
    plt.plot(x, y2_fit, 'g')
    plt.subplot(2, 2, 4)
    plt.plot(x, y3, 'b--')
    plt.plot(x, y3_fit, 'g')
    plt.show()


def test_linear_Fit_1D():
    # 拟合直线曲线解析式
    def target_func(x, a, b):
        return a * x+b
    # 模拟生成一组实验数据(加噪声)
    x = np.arange(0, 100, 0.2)
    y = 2.0*x+3
    noise = np.random.uniform(0, 0.1, len(x))
    y += noise
    fig, ax = plt.subplots()
    ax.plot(x, y, 'b--')
    # 调用linear_Fit_1D拟合实验数据
    parameters, err = fit.linear_Fit_1D(x, y)
    y_fit = [target_func(a, *parameters) for a in x]
    # 画图
    ax.plot(x, y_fit, 'g')
    plt.show()


def test_linear_FitX_2D():
    # 拟合直线曲线解析式
    def target_func(x, a, b):
        return a * x+b
    # 模拟生成4组实验数据(加噪声)
    x = np.arange(0, 100, 0.2)
    y0 = 1.0*x+2.0
    noise = np.random.uniform(0, 0.1, len(x))
    y0 += noise

    y1 = x-1.0
    noise = np.random.uniform(0, 0.1, len(x))
    y1 += noise

    y2 = -3*x-1.0
    noise = np.random.uniform(0, 0.1, len(x))
    y2 += noise

    y3 = -2*x+3.0
    noise = np.random.uniform(0, 0.1, len(x))
    y3 += noise
    # 调用linear_FitX_2D拟合实验数据
    parameters, err = fit.linear_FitX_2D([x, x, x, x], [y0, y1, y2, y3])
    y0_fit = [target_func(a, *parameters[0]) for a in x]
    y1_fit = [target_func(a, *parameters[1]) for a in x]
    y2_fit = [target_func(a, *parameters[2]) for a in x]
    y3_fit = [target_func(a, *parameters[3]) for a in x]
    # 画图
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(x, y0, 'b--')
    plt.plot(x, y0_fit, 'g')
    plt.subplot(2, 2, 2)
    plt.plot(x, y1, 'b--')
    plt.plot(x, y1_fit, 'g')
    plt.subplot(2, 2, 3)
    plt.plot(x, y2, 'b--')
    plt.plot(x, y2_fit, 'g')
    plt.subplot(2, 2, 4)
    plt.plot(x, y3, 'b--')
    plt.plot(x, y3_fit, 'g')
    plt.show()


def test_linear_FitY_2D():
    # 拟合直线曲线解析式
    def target_func(x, a, b):
        return a * x+b
    # 模拟生成4组实验数据(加噪声)
    x = np.arange(0, 100, 0.2)
    y0 = 1.0*x+2.0
    noise = np.random.uniform(0, 0.1, len(x))
    y0 += noise

    y1 = x-1.0
    noise = np.random.uniform(0, 0.1, len(x))
    y1 += noise

    y2 = -3*x-1.0
    noise = np.random.uniform(0, 0.1, len(x))
    y2 += noise

    y3 = -2*x+3.0
    noise = np.random.uniform(0, 0.1, len(x))
    y3 += noise
    # 调用linear_FitY_2D拟合实验数据
    y_data = np.transpose(np.array([x, x, x, x])).tolist()
    data = np.transpose(np.array([y0, y1, y2, y3])).tolist()
    parameters, err = fit.linear_FitY_2D(y_data, data)
    y0_fit = [target_func(a, *parameters[0]) for a in x]
    y1_fit = [target_func(a, *parameters[1]) for a in x]
    y2_fit = [target_func(a, *parameters[2]) for a in x]
    y3_fit = [target_func(a, *parameters[3]) for a in x]
    # 画图
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(x, y0, 'b--')
    plt.plot(x, y0_fit, 'g')
    plt.subplot(2, 2, 2)
    plt.plot(x, y1, 'b--')
    plt.plot(x, y1_fit, 'g')
    plt.subplot(2, 2, 3)
    plt.plot(x, y2, 'b--')
    plt.plot(x, y2_fit, 'g')
    plt.subplot(2, 2, 4)
    plt.plot(x, y3, 'b--')
    plt.plot(x, y3_fit, 'g')
    plt.show()


def test_sin_Fit_1D():
    # 拟合sin曲线解析式
    def target(t, a0, a1, a2, a3):
        return a0 * np.sin(t / a1 + a2) + a3
    # 模拟生成一组实验数据(加噪声)
    x = np.arange(0, 10, 0.01)
    y = 10.0 * np.sin((2*np.pi*1.0)*x - np.pi/2) + 3.0
    noise = 5*np.random.uniform(0, 0.5, len(x))
    y += noise
    fig, ax = plt.subplots()
    ax.plot(x, y, 'b-')
    # 调用sin_Fit_1D拟合实验数据
    parameters, err = fit.fit_sin(x, y)
    y_fit = [target(a, *parameters) for a in x]
    print(parameters)
    # 画图
    ax.plot(x, y_fit, 'g')
    plt.show()


if __name__ == "__main__":
    main()

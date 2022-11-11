import numpy as np
import copy
import matplotlib.pyplot as plt
import fit


def main():
    test_exp_fit_1d()
    test_exp_fitX_2d()
    test_exp_fitY_2d()
    test_linear_fit_1d()
    test_linear_fitX_2d()
    test_linear_fitY_2d()
    test_sin_fit_1d()
    test_sin_fitX_2d()
    test_sin_fitY_2d()
    test_sin_decay_fit_1d()
    test_sin_decay_fitX_2d()
    test_sin_decay_fitY_2d()
    test_quadratic_fit_1d()
    test_quadratic_fitX_2d()
    test_quadratic_fitY_2d()


def test_exp_fit_1d():
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
    # 调用exp_fit_1d拟合实验数据
    parameters, err = fit.exp_fit_1d(x, y)
    y_fit = [target_func(a, *parameters) for a in x]
    # 画图
    ax.plot(x, y_fit, 'g')
    plt.title("exp 1d")
    plt.show()


def test_exp_fitX_2d():
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
    # 调用exp_fitX_2d拟合实验数据
    parameters, err = fit.exp_fit_2d(
        [x, x, x, x], [y0, y1, y2, y3], mode="row")
    y0_fit = [target_func(a, *parameters[0]) for a in x]
    y1_fit = [target_func(a, *parameters[1]) for a in x]
    y2_fit = [target_func(a, *parameters[2]) for a in x]
    y3_fit = [target_func(a, *parameters[3]) for a in x]
    # 画图
    fig=plt.figure()
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
    fig.suptitle("exp 2d row")
    plt.show()


def test_exp_fitY_2d():
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
    # 调用exp_fitY_2d拟合实验数据
    y_data = np.transpose(np.array([x, x, x, x])).tolist()
    data = np.transpose(np.array([y0, y1, y2, y3])).tolist()
    parameters, err = fit.exp_fit_2d(y_data, data, mode="col")
    y0_fit = [target_func(a, *parameters[0]) for a in x]
    y1_fit = [target_func(a, *parameters[1]) for a in x]
    y2_fit = [target_func(a, *parameters[2]) for a in x]
    y3_fit = [target_func(a, *parameters[3]) for a in x]
    # 画图
    fig=plt.figure()
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
    fig.suptitle("exp 2d col")
    plt.show()


def test_linear_fit_1d():
    # 拟合直线曲线解析式
    def target_func(x, a, b):
        return a * x+b
    # 模拟生成一组实验数据(加噪声)
    x = np.arange(0, 100, 0.2)
    y = 2.0*x+3.0
    noise = np.random.uniform(0, 0.1, len(x))
    y += noise
    fig, ax = plt.subplots()
    ax.plot(x, y, 'b--')
    # 调用linear_fit_1d拟合实验数据
    parameters, err = fit.linear_fit_1d(x, y)
    y_fit = [target_func(a, *parameters) for a in x]
    # 画图
    ax.plot(x, y_fit, 'g')
    plt.title("linear 1d")
    plt.show()


def test_linear_fitX_2d():
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
    # 调用linear_fitX_2d拟合实验数据
    parameters, err = fit.linear_fit_2d(
        [x, x, x, x], [y0, y1, y2, y3], mode="row")
    y0_fit = [target_func(a, *parameters[0]) for a in x]
    y1_fit = [target_func(a, *parameters[1]) for a in x]
    y2_fit = [target_func(a, *parameters[2]) for a in x]
    y3_fit = [target_func(a, *parameters[3]) for a in x]
    # 画图
    fig=plt.figure()
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
    fig.suptitle("linear 2d row")
    plt.show()


def test_linear_fitY_2d():
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
    # 调用linear_fitY_2d拟合实验数据
    y_data = np.transpose(np.array([x, x, x, x])).tolist()
    data = np.transpose(np.array([y0, y1, y2, y3])).tolist()
    parameters, err = fit.linear_fit_2d(y_data, data, mode="col")
    y0_fit = [target_func(a, *parameters[0]) for a in x]
    y1_fit = [target_func(a, *parameters[1]) for a in x]
    y2_fit = [target_func(a, *parameters[2]) for a in x]
    y3_fit = [target_func(a, *parameters[3]) for a in x]
    # 画图
    fig=plt.figure()
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
    fig.suptitle("linear 2d col")
    plt.show()


def test_sin_fit_1d():
    # 拟合sin曲线解析式
    def target_func(x, a, b, c, phi):
        return a * np.sin(2*np.pi*b*x + phi) + c
    # 模拟生成一组实验数据(加噪声)
    x = np.arange(0, 1e-8, 1e-12)
    y = 10.0 * np.sin((2*np.pi*1.0e9)*x + np.pi/4) + 3.0
    noise = 1*np.random.uniform(0, 0.5, len(x))
    y += noise
    fig, ax = plt.subplots()
    ax.plot(x, y, 'b-')
    # 调用sin_fit_1d拟合实验数据
    parameters, err = fit.sin_fit_1d(x, y, 1e0)
    y_fit = [target_func(a, *parameters) for a in x]
    # 画图
    ax.plot(x, y_fit, 'g')
    plt.title("sin 1d")
    plt.show()


def test_sin_fitX_2d():
    # 拟合sin曲线解析式
    def target_func(x, a, b, c, phi):
        return a * np.sin(2*np.pi*b*x + phi) + c
    # 模拟生成4组实验数据(加噪声)
    x0 = np.arange(0, 1e-8, 1e-12)
    y0 = 10.0 * np.sin((2*np.pi*1.0e9)*x0 + np.pi/4) + 3.0
    noise = 1*np.random.uniform(0, 0.5, len(x0))
    y0 += noise

    x1 = np.arange(0, 10, 0.01)
    y1 = 10.0 * np.sin((2*np.pi*1.0)*x1 - np.pi/2) + 3.0
    noise = 5*np.random.uniform(0, 0.5, len(x1))
    y1 += noise

    x2 = np.arange(0, 1e-6, 1e-12)
    y2 = 10.0 * np.sin((2*np.pi*1.0e9)*x2 + np.pi/4) + 3.0
    noise = 1*np.random.uniform(0, 0.5, len(x2))
    y2 += noise

    x3 = np.arange(0, 1e-8, 1e-12)
    y3 = 10.0 * np.sin((2*np.pi*1.0e9)*x3 - np.pi/2) + 3.0
    noise = 1*np.random.uniform(0, 0.5, len(x3))
    y3 += noise

    # 调用sin_fitX_2d拟合实验数据
    parameters, err = fit.sin_fit_2d(
        [x0, x1, x2, x3], [y0, y1, y2, y3], [1e9, 1, 1e9, 1e9], mode="row")
    y0_fit = [target_func(a, *parameters[0]) for a in x0]
    y1_fit = [target_func(a, *parameters[1]) for a in x1]
    y2_fit = [target_func(a, *parameters[2]) for a in x2]
    y3_fit = [target_func(a, *parameters[3]) for a in x3]
    # 画图
    fig=plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(x0, y0, 'b--')
    plt.plot(x0, y0_fit, 'g')
    plt.subplot(2, 2, 2)
    plt.plot(x1, y1, 'b--')
    plt.plot(x1, y1_fit, 'g')
    plt.subplot(2, 2, 3)
    plt.plot(x2, y2, 'b--')
    plt.plot(x2, y2_fit, 'g')
    plt.subplot(2, 2, 4)
    plt.plot(x3, y3, 'b--')
    plt.plot(x3, y3_fit, 'g')
    fig.suptitle("sin 2d row")
    plt.show()


def test_sin_fitY_2d():
    # 拟合sin曲线解析式
    def target_func(x, a, b, c, phi):
        return a * np.sin(2*np.pi*b*x + phi) + c
    # 模拟生成4组实验数据(加噪声)
    x0 = np.arange(0, 1e-8, 1e-12)
    y0 = 10.0 * np.sin((2*np.pi*1.0e9)*x0 + np.pi/4) + 3.0
    noise = 1*np.random.uniform(0, 0.5, len(x0))
    y0 += noise

    x1 = np.arange(0, 10, 0.001)
    y1 = 10.0 * np.sin((2*np.pi*1.0)*x1 - np.pi/2) + 3.0
    noise = 5*np.random.uniform(0, 0.5, len(x1))
    y1 += noise

    x2 = np.arange(0, 1e-8, 1e-12)
    y2 = 10.0 * np.sin((2*np.pi*1.0e10)*x2 + np.pi/4) + 3.0
    noise = 1*np.random.uniform(0, 0.5, len(x2))
    y2 += noise

    x3 = np.arange(0, 1e-8, 1e-12)
    y3 = 10.0 * np.sin((2*np.pi*1.0e9)*x3 - np.pi/2) + 3.0
    noise = 1*np.random.uniform(0, 0.5, len(x3))
    y3 += noise
    # 调用sin_fitY_2d拟合实验数据
    y_data = np.transpose(np.array([x0, x1, x2, x3])).tolist()
    data = np.transpose(np.array([y0, y1, y2, y3])).tolist()
    parameters, err = fit.sin_fit_2d(
        y_data, data, [1e9, 1, 1e10, 1e9], mode="col")
    y0_fit = [target_func(a, *parameters[0]) for a in x0]
    y1_fit = [target_func(a, *parameters[1]) for a in x1]
    y2_fit = [target_func(a, *parameters[2]) for a in x2]
    y3_fit = [target_func(a, *parameters[3]) for a in x3]
    # 画图
    fig=plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(x0, y0, 'b--')
    plt.plot(x0, y0_fit, 'g')
    plt.subplot(2, 2, 2)
    plt.plot(x1, y1, 'b--')
    plt.plot(x1, y1_fit, 'g')
    plt.subplot(2, 2, 3)
    plt.plot(x2, y2, 'b--')
    plt.plot(x2, y2_fit, 'g')
    plt.subplot(2, 2, 4)
    plt.plot(x3, y3, 'b--')
    plt.plot(x3, y3_fit, 'g')
    fig.suptitle("sin 2d col")
    plt.show()


def test_sin_decay_fit_1d():
    # 拟合sin_decay曲线解析式
    def target_func(x, a, b, c, d, phi):
        return a * np.sin(2*np.pi*b*x + phi)*np.exp(c*x) + d
    # 模拟生成一组实验数据(加噪声)
    x = np.arange(0, 1e-8, 1e-12)
    y = 10 * np.sin(2*np.pi*1e9*x + np.pi/2)*np.exp(-1e8*x) + 1
    noise = 10*np.random.uniform(0, 0.5, len(x))
    y += noise
    fig, ax = plt.subplots()
    ax.plot(x, y, 'b-')
    # 调用sin_decay_fit_1d拟合实验数据
    parameters, err = fit.sin_decay_fit_1d(x, y, 1e9)
    y_fit = [target_func(a, *parameters) for a in x]
    # 画图
    ax.plot(x, y_fit, 'g')
    plt.title("sin_decay 1d")
    plt.show()


def test_sin_decay_fitX_2d():
    # 拟合sin_decay曲线解析式
    def target_func(x, a, b, c, d, phi):
        return a * np.sin(2*np.pi*b*x + phi)*np.exp(c*x) + d
    # 模拟生成4组实验数据(加噪声)
    x0 = np.arange(0, 1e-8, 1e-12)
    y0 = 10 * np.sin(2*np.pi*1e9*x0 + np.pi/2)*np.exp(-1e8*x0) + 1
    noise = 10*np.random.uniform(0, 0.5, len(x0))
    y0 += noise

    x1 = np.arange(0, 10, 0.001)
    y1 = 10 * np.sin(2*np.pi*1*x1 + np.pi/2)*np.exp(-1*x1) + 1
    noise = 2*np.random.uniform(0, 0.5, len(x0))
    y1 += noise

    x2 = np.arange(0, 1e-6, 1e-12)
    y2 = 10 * np.sin(2*np.pi*1e9*x2 + np.pi/2)*np.exp(-1e8*x2) + 1
    noise = 2*np.random.uniform(0, 0.5, len(x2))
    y2 += noise

    x3 = np.arange(0, 10, 0.001)
    y3 = 10 * np.sin(2*np.pi*1*x3 + np.pi/2)*np.exp(-1*x3) + 1
    noise = 10*np.random.uniform(0, 0.5, len(x3))
    y3 += noise
    # 调用sin_decay_fitX_2d拟合实验数据
    parameters, err = fit.sin_decay_fit_2d(
        [x0, x1, x2, x3], [y0, y1, y2, y3], [1e9, 1, 1e9, 1], mode="row")
    y0_fit = [target_func(a, *parameters[0]) for a in x0]
    y1_fit = [target_func(a, *parameters[1]) for a in x1]
    y2_fit = [target_func(a, *parameters[2]) for a in x2]
    y3_fit = [target_func(a, *parameters[3]) for a in x3]
    # 画图
    fig=plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(x0, y0, 'b--')
    plt.plot(x0, y0_fit, 'g')
    plt.subplot(2, 2, 2)
    plt.plot(x1, y1, 'b--')
    plt.plot(x1, y1_fit, 'g')
    plt.subplot(2, 2, 3)
    plt.plot(x2, y2, 'b--')
    plt.plot(x2, y2_fit, 'g')
    plt.subplot(2, 2, 4)
    plt.plot(x3, y3, 'b--')
    plt.plot(x3, y3_fit, 'g')
    fig.suptitle("sin_decay 2d row")
    plt.show()


def test_sin_decay_fitY_2d():
    # 拟合sin_decay曲线解析式
    def target_func(x, a, b, c, d, phi):
        return a * np.sin(2*np.pi*b*x + phi)*np.exp(c*x) + d
    # 模拟生成4组实验数据(加噪声)
    x0 = np.arange(0, 1e-8, 1e-12)
    y0 = 10 * np.sin(2*np.pi*1e9*x0 + np.pi/2)*np.exp(-1e8*x0) + 1
    noise = 10*np.random.uniform(0, 0.5, len(x0))
    y0 += noise

    x1 = np.arange(0, 10, 0.001)
    y1 = 10 * np.sin(2*np.pi*1*x1 + np.pi/2)*np.exp(-1*x1) + 1
    noise = 2*np.random.uniform(0, 0.5, len(x0))
    y1 += noise

    x2 = np.arange(0, 1e-6, 1e-10)
    y2 = 10 * np.sin(2*np.pi*1e9*x2 + np.pi/2)*np.exp(-1e8*x2) + 1
    noise = 2*np.random.uniform(0, 0.5, len(x2))
    y2 += noise

    x3 = np.arange(0, 10, 0.001)
    y3 = 10 * np.sin(2*np.pi*1*x3 + np.pi/2)*np.exp(-1*x3) + 1
    noise = 10*np.random.uniform(0, 0.5, len(x3))
    y3 += noise
    # 调用sin_decay_fitY_2d拟合实验数据
    y_data = np.transpose(np.array([x0, x1, x2, x3])).tolist()
    data = np.transpose(np.array([y0, y1, y2, y3])).tolist()
    parameters, err = fit.sin_decay_fit_2d(
        y_data, data, [1e9, 1, 1e9, 1], mode="col")
    y0_fit = [target_func(a, *parameters[0]) for a in x0]
    y1_fit = [target_func(a, *parameters[1]) for a in x1]
    y2_fit = [target_func(a, *parameters[2]) for a in x2]
    y3_fit = [target_func(a, *parameters[3]) for a in x3]
    # 画图
    fig=plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(x0, y0, 'b--')
    plt.plot(x0, y0_fit, 'g')
    plt.subplot(2, 2, 2)
    plt.plot(x1, y1, 'b--')
    plt.plot(x1, y1_fit, 'g')
    plt.subplot(2, 2, 3)
    plt.plot(x2, y2, 'b--')
    plt.plot(x2, y2_fit, 'g')
    plt.subplot(2, 2, 4)
    plt.plot(x3, y3, 'b--')
    plt.plot(x3, y3_fit, 'g')
    fig.suptitle("sin_decay 2d col")
    plt.show()


def test_quadratic_fit_1d():
    # 拟合二次曲线解析式
    def target_func(x, a, b, c):
        return a * x * x + b * x + c
    # 模拟生成一组实验数据(加噪声)
    x = np.arange(0, 10, 0.01)
    y = 0.2 * x * x - 1.5 * x + 1
    noise = 10*np.random.uniform(0, 0.1, len(x))
    y += noise
    fig, ax = plt.subplots()
    ax.plot(x, y, 'b--')
    # 调用quadratic_fit_1d拟合实验数据
    parameters, err = fit.quadratic_fit_1d(x, y)
    y_fit = [target_func(a, *parameters) for a in x]
    # 画图
    ax.plot(x, y_fit, 'g')
    plt.title("quadratic 1d")
    plt.show()


def test_quadratic_fitX_2d():
    # 拟合二次曲线解析式
    def target_func(x, a, b, c):
        return a * x * x + b * x + c
    # 模拟生成4组实验数据(加噪声)
    x = np.arange(0, 10, 0.01)
    y0 = 0.2 * x * x - 1.5 * x + 1
    noise = 10*np.random.uniform(0, 0.1, len(x))
    y0 += noise

    y1 = -0.2 * x * x + 1.5 * x + 1
    noise = 10*np.random.uniform(0, 0.1, len(x))
    y1 += noise

    y2 = 0.2 * x * x - 1.5 * x + 1
    noise = 10*np.random.uniform(0, 0.1, len(x))
    y2 += noise

    y3 = -0.2 * x * x + 1.5 * x + 1
    noise = 10*np.random.uniform(0, 0.1, len(x))
    y3 += noise
    # 调用quadratic_fitX_2d拟合实验数据
    parameters, err = fit.quadratic_fit_2d(
        [x, x, x, x], [y0, y1, y2, y3], mode="row")
    y0_fit = [target_func(a, *parameters[0]) for a in x]
    y1_fit = [target_func(a, *parameters[1]) for a in x]
    y2_fit = [target_func(a, *parameters[2]) for a in x]
    y3_fit = [target_func(a, *parameters[3]) for a in x]
    # 画图
    fig=plt.figure()
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
    fig.suptitle("quadratic 2d row")
    plt.show()


def test_quadratic_fitY_2d():
    # 拟合二次曲线解析式
    def target_func(x, a, b, c):
        return a * x * x + b * x + c
    # 模拟生成4组实验数据(加噪声)
    x = np.arange(0, 10, 0.01)
    y0 = 0.2 * x * x - 1.5 * x + 1
    noise = 10*np.random.uniform(0, 0.1, len(x))
    y0 += noise

    y1 = -0.2 * x * x + 1.5 * x + 1
    noise = 10*np.random.uniform(0, 0.1, len(x))
    y1 += noise

    y2 = 0.2 * x * x - 1.5 * x + 1
    noise = 10*np.random.uniform(0, 0.1, len(x))
    y2 += noise

    y3 = -0.2 * x * x + 1.5 * x + 1
    noise = 10*np.random.uniform(0, 0.1, len(x))
    y3 += noise
    # 调用quadratic_fitY_2d拟合实验数据
    y_data = np.transpose(np.array([x, x, x, x])).tolist()
    data = np.transpose(np.array([y0, y1, y2, y3])).tolist()
    parameters, err = fit.quadratic_fit_2d(y_data, data, mode="col")
    y0_fit = [target_func(a, *parameters[0]) for a in x]
    y1_fit = [target_func(a, *parameters[1]) for a in x]
    y2_fit = [target_func(a, *parameters[2]) for a in x]
    y3_fit = [target_func(a, *parameters[3]) for a in x]
    # 画图
    fig=plt.figure()
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
    fig.suptitle("quadratic 2d col")
    plt.show()


if __name__ == "__main__":
    main()

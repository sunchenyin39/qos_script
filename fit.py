import numpy as np
from scipy import optimize


def exp_Fit_1D(x, data, fitRangeMin=None, fitRangeMax=None, fitInitial=None):
    def target_func(x, a, b, c):
        return a * np.exp(b*x)+c

    # make a guess if fitInitial parameters are not available
    if fitInitial is None:
        if (data[-1] < data[0]) and (data[-1] + data[0])/2 > data[round(len(data) / 2)]:
            fitInitial = [max(data)-min(data), -1.0 /
                          x[round(len(x) / 2)], min(data)]
        if (data[-1] > data[0]) and (data[-1] + data[0])/2 > data[round(len(data) / 2)]:
            fitInitial = [max(data)-min(data), 1.0 /
                          x[round(len(x) / 2)], min(data)]
        if (data[-1] < data[0]) and (data[-1] + data[0])/2 < data[round(len(data) / 2)]:
            fitInitial = [min(data)-max(data), +1.0 /
                          x[round(len(x) / 2)], max(data)]
        if (data[-1] > data[0]) and (data[-1] + data[0])/2 < data[round(len(data) / 2)]:
            fitInitial = [min(data)-max(data), -1.0 /
                          x[round(len(x) / 2)], max(data)]

    parameters, covariance = optimize.curve_fit(
        target_func, x, data, p0=fitInitial, bounds=(fitRangeMin, fitRangeMax), maxfev=500000)  # maxfev is the maximul times of iteration regression

    err = np.sqrt(np.diag(covariance)) * 1.96  # 95% confidence area

    return parameters, err


def exp_FitX_2D(x, data, fitRangeMin=None, fitRangeMax=None, fitInitial=None):
    parameters = []
    err = []
    for i in range(len(x)):
        parameters_temp, err_temp = exp_Fit_1D(
            x[i], data[i], fitRangeMin, fitRangeMax, fitInitial)
        parameters.append(parameters_temp)
        err.append(err_temp)
    return parameters, err


def exp_FitY_2D(y, data, fitRangeMin=None, fitRangeMax=None, fitInitial=None):
    y = np.transpose(np.array(y)).tolist()
    data = np.transpose(np.array(data)).tolist()
    parameters, err = exp_FitX_2D(
        y, data, fitRangeMin=None, fitRangeMax=None, fitInitial=None)
    return parameters, err


def linear_Fit_1D(x, data, fitRangeMin=None, fitRangeMax=None, fitInitial=None):
    def target_func(x, a, b):
        return a * x+b
    if fitInitial is None:
        fitInitial = [(data[-1]-data[0])/(x[-1]-x[0]), data[0] -
                      (data[-1]-data[0])/(x[-1]-x[0])*x[0]]
    parameters, covariance = optimize.curve_fit(
        target_func, x, data, p0=fitInitial, bounds=(fitRangeMin, fitRangeMax), maxfev=500000)
    err = np.sqrt(np.diag(covariance)) * 1.96  # 95% confidence area
    return parameters, err


def linear_FitX_2D(x, data, fitRangeMin=None, fitRangeMax=None, fitInitial=None):
    parameters = []
    err = []
    for i in range(len(x)):
        parameters_temp, err_temp = linear_Fit_1D(
            x[i], data[i], fitRangeMin, fitRangeMax, fitInitial)
        parameters.append(parameters_temp)
        err.append(err_temp)
    return parameters, err


def linear_FitY_2D(y, data, fitRangeMin=None, fitRangeMax=None, fitInitial=None):
    y = np.transpose(np.array(y)).tolist()
    data = np.transpose(np.array(data)).tolist()
    parameters, err = linear_FitX_2D(
        y, data, fitRangeMin=None, fitRangeMax=None, fitInitial=None)
    return parameters, err


def guess_freq(x, y):
    '''
        Guess the frequency of input data
        by finding sideband position in DFT of input data
    '''
    x = np.array(x)
    y = np.array(y)
    freq = np.fft.fftfreq(len(x), x[1] - x[0])  # uniform spacing
    Y = abs(np.fft.fft(y))
    # find peak (exclude zero frequency)
    guess = abs(freq[np.argmax(Y[1:]) + 1])
    return guess


# def sin_Fit_1D(x, data, fitRangeMin=None, fitRangeMax=None, fitInitial=None):
#     def target_fun(x, a, b, c, phi):
#         return a * np.sin(2*np.pi*b*x + phi) + c

#     # make a guess if initial parameters are not available
#     if fitInitial is None:
#         if np.argmax(data) == len(data) - 1:
#             freq = 1/np.pi/x[-1]
#         else:
#             freq = guess_freq(x, data)
#         amp = (np.max(data)-np.min(data))/2
#         offset = np.mean(data)
#         fitInitial = [amp, freq, offset, -np.pi/2]

#     parameters, covariance = optimize.curve_fit(
#         target_fun, x, data, p0=fitInitial, maxfev=500000)
#     err = np.sqrt(np.diag(covariance)) * 1.96  # 95% confidence area

#     if err[1] > abs(parameters[1])*0.1:
#         errlist = []
#         parmlist = []
#         variance = []
#         for ii in range(5, len(data)):
#             fitInitial[1] = 1.0/(2*np.pi*x[ii])
#             parameters, covariance = optimize.curve_fit(
#                 target_fun, x, data, p0=fitInitial, maxfev=500000)
#             err = np.sqrt(np.diag(covariance)) * 1.96  # 95% confidence area
#             errlist.append(err)
#             parmlist.append(parameters)

#             y2 = [target_fun(i, parameters[0], parameters[1],
#                              parameters[2], parameters[3]) for i in x]
#             if np.var([data, y2]) == 'nan':
#                 variance.append(1e9)
#             else:
#                 variance.append(np.var([data, y2]))
#         idx = np.argmin(variance)
#         parameters = parmlist[idx]
#         err = errlist[idx]
#     return parameters, err

def fit_sin(x, y, p0=None):
    def target(t, a0, a1, a2, a3):
        return a0 * np.sin(t / a1 + a2) + a3

    # make a guess if initial parameters are not available
    if p0 is None:
        if np.argmax(y) == len(y) - 1:
            freq = 1/np.pi/x[-1]
        else:
            freq = guess_freq(x, y)
        # amp = np.std(y) * 2.0 ** 0.5
        amp = (np.max(y)-np.min(y))/2
        offset = np.mean(y)
        p0 = [amp, 1 / (2 * np.pi * freq), -np.pi/2, offset]

    parameters, covariance = optimize.curve_fit(target, x, y, p0=p0, maxfev=500000)  # maxfev修改拟合次数上限，默认为1000
    err = np.sqrt(np.diag(covariance)) * 1.96  # 95% confidence area
    if err[1]>abs(parameters[1])*0.1:
        errlist=[]
        parmlist=[]
        variance=[]
        for ii in range(5, len(y)):
            p0[1] = x[ii]
            parameters, covariance = optimize.curve_fit(target, x, y, p0=p0, maxfev=500000)  # maxfev修改拟合次数上限，默认为1000
            err = np.sqrt(np.diag(covariance)) * 1.96  # 95% confidence area
            errlist.append(err)
            parmlist.append(parameters)
            y2=[target(i, parameters[0], parameters[1], parameters[2], parameters[3]) for i in x]
            if np.var([y,y2])=='nan':
                variance.append(1e9)
            else:
                variance.append(np.var([y,y2]))
        idx=np.argmin(variance)
        parameters=parmlist[idx]
    return parameters, err
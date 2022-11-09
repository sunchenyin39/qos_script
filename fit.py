import numpy as np
from scipy import optimize


def exp_fit_1D(x, data, fitRangeMin=None, fitRangeMax=None, fitInitial=None):
    """Exponential function fitting.

    Args:
        x (list[float]): Independent variable list.
        data (list[float]): Dependent variable list.
        fitRangeMin (list[float], optional): The minimum values of the parameters to be fitted. Defaults to None.
        fitRangeMax (list[float], optional): The maximum values of the parameters to be fitted. Defaults to None.
        fitInitial (list[float], optional): The initial values of the parameters to be fitted. Defaults to None.

    Returns:
        tuple[list[float],list[float]]: The fist list is fitted parameters, the second list is the error list.

    """    
    def target_func(x, a, b, c):
        return a * np.exp(b*x)+c

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
        target_func, x, data, p0=fitInitial, bounds=(fitRangeMin, fitRangeMax), maxfev=500000)

    err = np.sqrt(np.diag(covariance)) * 1.96

    return parameters, err


def exp_fit_2D(x, data, fitRangeMin=None, fitRangeMax=None, fitInitial=None, mode="row"):
    """Exponential function fitting. According the fist 

    Args:
        x (list(list[float])): Independent variable list.
        data (list(list[float])): Dependent variable list.
        fitRangeMin (list(list[float]), optional): The minimum values of the parameters to be fitted. Defaults to None.
        fitRangeMax (list(list[float]), optional): The maximum values of the parameters to be fitted. Defaults to None.
        fitInitial (list(list[float]), optional): The initial values of the parameters to be fitted. Defaults to None.
        mode (str, optional): _description_. Defaults to "row".

    Returns:
        tuple[list[float],list[float]]: The fist list is fitted parameters, the second list is the error list.

    """    
    def exp_fitX_2D(x, data, fitRangeMin=None, fitRangeMax=None, fitInitial=None):
        parameters = []
        err = []
        for i in range(len(x)):
            parameters_temp, err_temp = exp_fit_1D(
                x[i], data[i], fitRangeMin, fitRangeMax, fitInitial)
            parameters.append(parameters_temp)
            err.append(err_temp)
        return parameters, err
    if mode[0] == 'r':
        parameters, err = exp_fitX_2D(
            x, data, fitRangeMin, fitRangeMax, fitInitial)
        return parameters, err
    else:
        x = np.transpose(np.array(x)).tolist()
        data = np.transpose(np.array(data)).tolist()
        parameters, err = exp_fitX_2D(
            x, data, fitRangeMin, fitRangeMax, fitInitial)
        return parameters, err


def linear_fit_1D(x, data, fitRangeMin=None, fitRangeMax=None, fitInitial=None):
    """_summary_

    Args:
        x (_type_): _description_
        data (_type_): _description_
        fitRangeMin (_type_, optional): _description_. Defaults to None.
        fitRangeMax (_type_, optional): _description_. Defaults to None.
        fitInitial (_type_, optional): _description_. Defaults to None.
    """    
    def target_func(x, a, b):
        return a * x+b
    if fitInitial is None:
        fitInitial = [(data[-1]-data[0])/(x[-1]-x[0]), data[0] -
                      (data[-1]-data[0])/(x[-1]-x[0])*x[0]]
    parameters, covariance = optimize.curve_fit(
        target_func, x, data, p0=fitInitial, bounds=(fitRangeMin, fitRangeMax), maxfev=500000)
    err = np.sqrt(np.diag(covariance)) * 1.96  
    return parameters, err


def linear_fit_2D(x, data, fitRangeMin=None, fitRangeMax=None, fitInitial=None, mode="row"):
    """_summary_

    Args:
        x (_type_): _description_
        data (_type_): _description_
        fitRangeMin (_type_, optional): _description_. Defaults to None.
        fitRangeMax (_type_, optional): _description_. Defaults to None.
        fitInitial (_type_, optional): _description_. Defaults to None.
        mode (str, optional): _description_. Defaults to "row".
    """    
    def linear_fitX_2D(x, data, fitRangeMin=None, fitRangeMax=None, fitInitial=None):
        parameters = []
        err = []
        for i in range(len(x)):
            parameters_temp, err_temp = linear_fit_1D(
                x[i], data[i], fitRangeMin, fitRangeMax, fitInitial)
            parameters.append(parameters_temp)
            err.append(err_temp)
        return parameters, err
    if mode[0] == 'r':
        parameters, err = linear_fitX_2D(
            x, data, fitRangeMin, fitRangeMax, fitInitial)
        return parameters, err
    else:
        x = np.transpose(np.array(x)).tolist()
        data = np.transpose(np.array(data)).tolist()
        parameters, err = linear_fitX_2D(
            x, data, fitRangeMin, fitRangeMax, fitInitial)
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


def sin_fit_1D(x, data, freqmagnitude=None, fitRangeMin=None, fitRangeMax=None, fitInitial=None):
    def target_func(x, a, b, c, phi):
        return a * np.sin(2*np.pi*b*x + phi) + c
    if freqmagnitude is None:
        freqmagnitude = 1e6

    x = np.array(x)
    x = x*freqmagnitude
    
    if fitInitial is None:
        if np.argmax(data) == len(data) - 1:
            freq = 1/np.pi/x[-1]
        else:
            freq = guess_freq(x, data)
        amp = (np.max(data)-np.min(data))/2
        offset = np.mean(data)
        fitInitial = [amp, freq, offset, -np.pi/2]

    parameters, covariance = optimize.curve_fit(
        target_func, x, data, p0=fitInitial, maxfev=500000)
    err = np.sqrt(np.diag(covariance)) * 1.96  

    if err[1] > abs(parameters[1])*0.1:
        errlist = []
        parmlist = []
        variance = []
        for ii in range(5, len(data)):
            fitInitial[1] = 1.0/(2*np.pi*x[ii])
            parameters, covariance = optimize.curve_fit(
                target_func, x, data, p0=fitInitial, maxfev=500000)
            err = np.sqrt(np.diag(covariance)) * 1.96  
            errlist.append(err)
            parmlist.append(parameters)

            y2 = [target_func(i, parameters[0], parameters[1],
                              parameters[2], parameters[3]) for i in x]
            if np.var([data, y2]) == 'nan':
                variance.append(1e9)
            else:
                variance.append(np.var([data, y2]))
        idx = np.argmin(variance)
        parameters = parmlist[idx]
        err = errlist[idx]
    parameters[1] = parameters[1]*freqmagnitude
    return parameters, err


def sin_fit_2D(x, data, freqmagnitude=None, fitRangeMin=None, fitRangeMax=None, fitInitial=None, mode="row"):
    def sin_fitX_2D(x, data, freqmagnitude=None, fitRangeMin=None, fitRangeMax=None, fitInitial=None):
        parameters = []
        err = []
        for i in range(len(x)):
            if freqmagnitude is None:
                parameters_temp, err_temp = sin_fit_1D(
                    x[i], data[i], 1e6, fitRangeMin, fitRangeMax, fitInitial)
            else:
                parameters_temp, err_temp = sin_fit_1D(
                    x[i], data[i], freqmagnitude[i], fitRangeMin, fitRangeMax, fitInitial)
            parameters.append(parameters_temp)
            err.append(err_temp)
        return parameters, err
    if mode[0] == 'r':
        parameters, err = sin_fitX_2D(
            x, data, freqmagnitude, fitRangeMin, fitRangeMax, fitInitial)
        return parameters, err
    else:
        x = np.transpose(np.array(x)).tolist()
        data = np.transpose(np.array(data)).tolist()
        parameters, err = sin_fitX_2D(
            x, data, freqmagnitude, fitRangeMin, fitRangeMax, fitInitial)
        return parameters, err


def sin_decay_fit_1D(x, data, freqmagnitude=None, fitRangeMin=None, fitRangeMax=None, fitInitial=None):
    def target_func(x, a, b, c, d, phi):
        return a * np.sin(2*np.pi*b*x + phi)*np.exp(c*x) + d
    if freqmagnitude is None:
        freqmagnitude = 1e6

    x = np.array(x)
    x = x*freqmagnitude
    
    if fitInitial is None:
        if np.argmax(data) == len(data) - 1:
            freq = 1/np.pi/x[-1]
        else:
            freq = guess_freq(x, data)
        
        amp = (np.max(data)-np.min(data))/2
        offset = np.mean(data)
        fitInitial = [amp, freq, -1.0/x[-1], offset, -np.pi/2]

    parameters, covariance = optimize.curve_fit(
        target_func, x, data, p0=fitInitial, maxfev=500000)
    err = np.sqrt(np.diag(covariance)) * 1.96  
    if err[1] > abs(parameters[1])*0.1 and len(data) > 5:
        errlist = []
        parmlist = []
        variance = []
        for ii in range(5, len(data)):
            fitInitial[1] = x[ii]
            parameters, covariance = optimize.curve_fit(
                target_func, x, data, p0=fitInitial, maxfev=500000)
            err = np.sqrt(np.diag(covariance)) * 1.96  
            errlist.append(err)
            parmlist.append(parameters)
            y2 = [target_func(i, parameters[0], parameters[1],
                              parameters[2], parameters[3], parameters[4]) for i in x]
            if np.var([data, y2]) == 'nan':
                variance.append(1e9)
            else:
                variance.append(np.var([data, y2]))
        idx = np.argmin(variance)
        parameters = parmlist[idx]
        err = errlist[idx]
    parameters[1] = parameters[1]*freqmagnitude
    parameters[2] = parameters[2]*freqmagnitude
    return parameters, err


def sin_decay_fit_2D(x, data, freqmagnitude=None, fitRangeMin=None, fitRangeMax=None, fitInitial=None, mode="row"):
    def sin_decay_fitX_2D(x, data, freqmagnitude=None, fitRangeMin=None, fitRangeMax=None, fitInitial=None):
        parameters = []
        err = []
        for i in range(len(x)):
            if freqmagnitude is None:
                parameters_temp, err_temp = sin_decay_fit_1D(
                    x[i], data[i], 1e6, fitRangeMin, fitRangeMax, fitInitial)
            else:
                parameters_temp, err_temp = sin_decay_fit_1D(
                    x[i], data[i], freqmagnitude[i], fitRangeMin, fitRangeMax, fitInitial)
            parameters.append(parameters_temp)
            err.append(err_temp)
        return parameters, err
    if mode[0] == 'r':
        parameters, err = sin_decay_fitX_2D(
            x, data, freqmagnitude, fitRangeMin, fitRangeMax, fitInitial)
        return parameters, err
    else:
        x = np.transpose(np.array(x)).tolist()
        data = np.transpose(np.array(data)).tolist()
        parameters, err = sin_decay_fitX_2D(
            x, data, freqmagnitude, fitRangeMin, fitRangeMax, fitInitial)
        return parameters, err


def quadratic_fit_1D(x, data, fitRangeMin=None, fitRangeMax=None, fitInitial=None):
    def target_func(x, a, b, c):
        return a * x * x + b * x + c
    parameters, covariance = optimize.curve_fit(
        target_func, x, data, p0=fitInitial, bounds=(fitRangeMin, fitRangeMax), maxfev=500000)
    err = np.sqrt(np.diag(covariance)) * 1.96  
    return parameters, err


def quadratic_fit_2D(x, data, fitRangeMin=None, fitRangeMax=None, fitInitial=None, mode="row"):
    def quadratic_fitX_2D(x, data, fitRangeMin=None, fitRangeMax=None, fitInitial=None):
        parameters = []
        err = []
        for i in range(len(x)):
            parameters_temp, err_temp = quadratic_fit_1D(
                x[i], data[i], fitRangeMin, fitRangeMax, fitInitial)
            parameters.append(parameters_temp)
            err.append(err_temp)
        return parameters, err
    if mode[0] == 'r':
        parameters, err = quadratic_fitX_2D(
            x, data, fitRangeMin, fitRangeMax, fitInitial)
        return parameters, err
    else:
        x = np.transpose(np.array(x)).tolist()
        data = np.transpose(np.array(data)).tolist()
        parameters, err = quadratic_fitX_2D(
            x, data, fitRangeMin, fitRangeMax, fitInitial)
        return parameters, err

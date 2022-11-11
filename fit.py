import numpy as np
from scipy import optimize


def exp_fit_1D(x, data, fit_range_min=None, fit_range_max=None, fit_initial=None):
    """Exponential function fitting. data = a * exp( b * x ) + c


    Args:
        x (list[float]): Independent variable list.
        data (list[float]): Dependent variable list.
        fit_range_min (list[float], optional): The minimum values of the parameters to be fitted which has the form of [a_min , b_min , c_min]. Defaults to None.
        fit_range_max (list[float], optional): The maximum values of the parameters to be fitted which has the form of [a_max , b_max , c_max]. Defaults to None.
        fit_initial (list[float], optional): The initial values of the parameters to be fitted which has the form of [a_init , b_init , c_init]. Defaults to None.

    Returns:
        tuple[list[float],list[float]]: The fist list is fitted parameters [a_fitted , b_fitted , c_fitted], the second list is the error list.

    """
    def target_func(x, a, b, c):
        return a * np.exp(b*x)+c

    if fit_initial is None:
        if (data[-1] < data[0]) and (data[-1] + data[0])/2 > data[round(len(data) / 2)]:
            fit_initial = [max(data)-min(data), -1.0 /
                           x[round(len(x) / 2)], min(data)]
        if (data[-1] > data[0]) and (data[-1] + data[0])/2 > data[round(len(data) / 2)]:
            fit_initial = [max(data)-min(data), 1.0 /
                           x[round(len(x) / 2)], min(data)]
        if (data[-1] < data[0]) and (data[-1] + data[0])/2 < data[round(len(data) / 2)]:
            fit_initial = [min(data)-max(data), +1.0 /
                           x[round(len(x) / 2)], max(data)]
        if (data[-1] > data[0]) and (data[-1] + data[0])/2 < data[round(len(data) / 2)]:
            fit_initial = [min(data)-max(data), -1.0 /
                           x[round(len(x) / 2)], max(data)]

    parameters, covariance = optimize.curve_fit(
        target_func, x, data, p0=fit_initial, bounds=(fit_range_min, fit_range_max), maxfev=500000)

    err = np.sqrt(np.diag(covariance)) * 1.96

    return parameters, err


def exp_fit_2D(x, data, fit_range_min=None, fit_range_max=None, fit_initial=None, mode="row"):
    """Exponential function fitting. data = a * exp( b * x ) + c
    If the first letter of mode was 'r', each row of data would be fitted once, 
    else each column of data would be fitted once.

    Args:
        x (list(list[float])): Independent variable list.
        data (list(list[float])): Dependent variable list.
        fit_range_min (list(list[float]), optional): The minimum values of the parameters to be fitted. Defaults to None.
        fit_range_max (list(list[float]), optional): The maximum values of the parameters to be fitted. Defaults to None.
        fit_initial (list(list[float]), optional): The initial values of the parameters to be fitted. Defaults to None.
        mode (str, optional): If the first letter of mode was 'r', each row of data would be fitted once, 
            else each column of data would be fitted once. Defaults to "row".

    Returns:
        tuple[list[list[float]],list[list[float]]]: The fist list is fitted parameters, the second list is the error list.

    """
    def exp_fitX_2D(x, data, fit_range_min=None, fit_range_max=None, fit_initial=None):
        parameters = []
        err = []
        for i in range(len(x)):
            parameters_temp, err_temp = exp_fit_1D(
                x[i], data[i], fit_range_min, fit_range_max, fit_initial)
            parameters.append(parameters_temp)
            err.append(err_temp)
        return parameters, err
    if mode[0] == 'r':
        parameters, err = exp_fitX_2D(
            x, data, fit_range_min, fit_range_max, fit_initial)
        return parameters, err
    else:
        x = np.transpose(np.array(x)).tolist()
        data = np.transpose(np.array(data)).tolist()
        parameters, err = exp_fitX_2D(
            x, data, fit_range_min, fit_range_max, fit_initial)
        return parameters, err


def linear_fit_1D(x, data, fit_range_min=None, fit_range_max=None, fit_initial=None):
    """Linear function fitting. data = a * x + b


    Args:
        x (list[float]): Independent variable list.
        data (list[float]): Dependent variable list.
        fit_range_min (list[float], optional): The minimum values of the parameters to be fitted which has the form of [a_min , b_min]. Defaults to None.
        fit_range_max (list[float], optional): The maximum values of the parameters to be fitted which has the form of [a_max , b_max]. Defaults to None.
        fit_initial (list[float], optional): The initial values of the parameters to be fitted which has the form of [a_init , b_init]. Defaults to None.

    Returns:
        tuple[list[float],list[float]]: The fist list is fitted parameters [a_fitted , b_fitted], the second list is the error list.

    """
    def target_func(x, a, b):
        return a * x+b
    if fit_initial is None:
        fit_initial = [(data[-1]-data[0])/(x[-1]-x[0]), data[0] -
                       (data[-1]-data[0])/(x[-1]-x[0])*x[0]]
    parameters, covariance = optimize.curve_fit(
        target_func, x, data, p0=fit_initial, bounds=(fit_range_min, fit_range_max), maxfev=500000)
    err = np.sqrt(np.diag(covariance)) * 1.96
    return parameters, err


def linear_fit_2D(x, data, fit_range_min=None, fit_range_max=None, fit_initial=None, mode="row"):
    """Linear function fitting. data = a * x + b
    If the first letter of mode was 'r', each row of data would be fitted once, 
    else each column of data would be fitted once.

    Args:
        x (list(list[float])): Independent variable list.
        data (list(list[float])): Dependent variable list.
        fit_range_min (list(list[float]), optional): The minimum values of the parameters to be fitted. Defaults to None.
        fit_range_max (list(list[float]), optional): The maximum values of the parameters to be fitted. Defaults to None.
        fit_initial (list(list[float]), optional): The initial values of the parameters to be fitted. Defaults to None.
        mode (str, optional): If the first letter of mode was 'r', each row of data would be fitted once, 
            else each column of data would be fitted once. Defaults to "row".

    Returns:
        tuple[list[list[float]],list[list[float]]]: The fist list is fitted parameters, the second list is the error list.

    """
    def linear_fitX_2D(x, data, fit_range_min=None, fit_range_max=None, fit_initial=None):
        parameters = []
        err = []
        for i in range(len(x)):
            parameters_temp, err_temp = linear_fit_1D(
                x[i], data[i], fit_range_min, fit_range_max, fit_initial)
            parameters.append(parameters_temp)
            err.append(err_temp)
        return parameters, err
    if mode[0] == 'r':
        parameters, err = linear_fitX_2D(
            x, data, fit_range_min, fit_range_max, fit_initial)
        return parameters, err
    else:
        x = np.transpose(np.array(x)).tolist()
        data = np.transpose(np.array(data)).tolist()
        parameters, err = linear_fitX_2D(
            x, data, fit_range_min, fit_range_max, fit_initial)
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


def sin_fit_1D(x, data, freqmagnitude=None, fit_range_min=None, fit_range_max=None, fit_initial=None):
    """Sinusoidal function fitting. data = a * sin( 2 * pi * b * x + phi) + c


    Args:
        x (list[float]): Independent variable list.
        data (list[float]): Dependent variable list.
        fit_range_min (list[float], optional): The minimum values of the parameters to be fitted which has the form of [a_min , b_min , c_min , phi_min]. Defaults to None.
        fit_range_max (list[float], optional): The maximum values of the parameters to be fitted which has the form of [a_max , b_max , c_max , phi_max]. Defaults to None.
        fit_initial (list[float], optional): The initial values of the parameters to be fitted which has the form of [a_init , b_init , c_init , phi_init]. Defaults to None.

    Returns:
        tuple[list[float],list[float]]: The fist list is fitted parameters [a_fitted , b_fitted , c_fitted , phi_fitted], the second list is the error list.

    """
    def target_func(x, a, b, c, phi):
        return a * np.sin(2*np.pi*b*x + phi) + c
    if freqmagnitude is None:
        freqmagnitude = 1e6

    x = np.array(x)
    x = x*freqmagnitude

    if fit_initial is None:
        if np.argmax(data) == len(data) - 1:
            freq = 1/np.pi/x[-1]
        else:
            freq = guess_freq(x, data)
        amp = (np.max(data)-np.min(data))/2
        offset = np.mean(data)
        fit_initial = [amp, freq, offset, -np.pi/2]

    parameters, covariance = optimize.curve_fit(
        target_func, x, data, p0=fit_initial, maxfev=500000)
    err = np.sqrt(np.diag(covariance)) * 1.96

    if err[1] > abs(parameters[1])*0.1:
        errlist = []
        parmlist = []
        variance = []
        for ii in range(5, len(data)):
            fit_initial[1] = 1.0/(2*np.pi*x[ii])
            parameters, covariance = optimize.curve_fit(
                target_func, x, data, p0=fit_initial, maxfev=500000)
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


def sin_fit_2D(x, data, freqmagnitude=None, fit_range_min=None, fit_range_max=None, fit_initial=None, mode="row"):
    """Sinusoidal function fitting. data = a * sin( 2 * pi * b * x + phi) + c
    If the first letter of mode was 'r', each row of data would be fitted once, 
    else each column of data would be fitted once.

    Args:
        x (list(list[float])): Independent variable list.
        data (list(list[float])): Dependent variable list.
        fit_range_min (list(list[float]), optional): The minimum values of the parameters to be fitted. Defaults to None.
        fit_range_max (list(list[float]), optional): The maximum values of the parameters to be fitted. Defaults to None.
        fit_initial (list(list[float]), optional): The initial values of the parameters to be fitted. Defaults to None.
        mode (str, optional): If the first letter of mode was 'r', each row of data would be fitted once, 
            else each column of data would be fitted once. Defaults to "row".

    Returns:
        tuple[list[list[float]],list[list[float]]]: The fist list is fitted parameters, the second list is the error list.

    """
    def sin_fitX_2D(x, data, freqmagnitude=None, fit_range_min=None, fit_range_max=None, fit_initial=None):
        parameters = []
        err = []
        for i in range(len(x)):
            if freqmagnitude is None:
                parameters_temp, err_temp = sin_fit_1D(
                    x[i], data[i], 1e6, fit_range_min, fit_range_max, fit_initial)
            else:
                parameters_temp, err_temp = sin_fit_1D(
                    x[i], data[i], freqmagnitude[i], fit_range_min, fit_range_max, fit_initial)
            parameters.append(parameters_temp)
            err.append(err_temp)
        return parameters, err
    if mode[0] == 'r':
        parameters, err = sin_fitX_2D(
            x, data, freqmagnitude, fit_range_min, fit_range_max, fit_initial)
        return parameters, err
    else:
        x = np.transpose(np.array(x)).tolist()
        data = np.transpose(np.array(data)).tolist()
        parameters, err = sin_fitX_2D(
            x, data, freqmagnitude, fit_range_min, fit_range_max, fit_initial)
        return parameters, err


def sin_decay_fit_1D(x, data, freqmagnitude=None, fit_range_min=None, fit_range_max=None, fit_initial=None):
    """Sinusoidal exponential decay function fitting. data = a * sin(2 * pi * b * x + phi) * exp(c * x) + d


    Args:
        x (list[float]): Independent variable list.
        data (list[float]): Dependent variable list.
        fit_range_min (list[float], optional): The minimum values of the parameters to be fitted which has the form of [a_min , b_min , c_min , d_min , phi_min]. Defaults to None.
        fit_range_max (list[float], optional): The maximum values of the parameters to be fitted which has the form of [a_max , b_max , c_max , d_max , phi_max]. Defaults to None.
        fit_initial (list[float], optional): The initial values of the parameters to be fitted which has the form of [a_init , b_init , c_init , d_init , phi_init]. Defaults to None.

    Returns:
        tuple[list[float],list[float]]: The fist list is fitted parameters [a_fitted , b_fitted , c_fitted , d_fitted , phi_fitted], the second list is the error list.

    """
    def target_func(x, a, b, c, d, phi):
        return a * np.sin(2*np.pi*b*x + phi)*np.exp(c*x) + d
    if freqmagnitude is None:
        freqmagnitude = 1e6

    x = np.array(x)
    x = x*freqmagnitude

    if fit_initial is None:
        if np.argmax(data) == len(data) - 1:
            freq = 1/np.pi/x[-1]
        else:
            freq = guess_freq(x, data)

        amp = (np.max(data)-np.min(data))/2
        offset = np.mean(data)
        fit_initial = [amp, freq, -1.0/x[-1], offset, -np.pi/2]

    parameters, covariance = optimize.curve_fit(
        target_func, x, data, p0=fit_initial, maxfev=500000)
    err = np.sqrt(np.diag(covariance)) * 1.96
    if err[1] > abs(parameters[1])*0.1 and len(data) > 5:
        errlist = []
        parmlist = []
        variance = []
        for ii in range(5, len(data)):
            fit_initial[1] = x[ii]
            parameters, covariance = optimize.curve_fit(
                target_func, x, data, p0=fit_initial, maxfev=500000)
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


def sin_decay_fit_2D(x, data, freqmagnitude=None, fit_range_min=None, fit_range_max=None, fit_initial=None, mode="row"):
    """Sinusoidal exponential decay function fitting. data = a * sin(2 * pi * b * x + phi) * exp(c * x) + d
    If the first letter of mode was 'r', each row of data would be fitted once, 
    else each column of data would be fitted once.

    Args:
        x (list(list[float])): Independent variable list.
        data (list(list[float])): Dependent variable list.
        fit_range_min (list(list[float]), optional): The minimum values of the parameters to be fitted. Defaults to None.
        fit_range_max (list(list[float]), optional): The maximum values of the parameters to be fitted. Defaults to None.
        fit_initial (list(list[float]), optional): The initial values of the parameters to be fitted. Defaults to None.
        mode (str, optional): If the first letter of mode was 'r', each row of data would be fitted once, 
            else each column of data would be fitted once. Defaults to "row".

    Returns:
        tuple[list[list[float]],list[list[float]]]: The fist list is fitted parameters, the second list is the error list.

    """
    def sin_decay_fitX_2D(x, data, freqmagnitude=None, fit_range_min=None, fit_range_max=None, fit_initial=None):
        parameters = []
        err = []
        for i in range(len(x)):
            if freqmagnitude is None:
                parameters_temp, err_temp = sin_decay_fit_1D(
                    x[i], data[i], 1e6, fit_range_min, fit_range_max, fit_initial)
            else:
                parameters_temp, err_temp = sin_decay_fit_1D(
                    x[i], data[i], freqmagnitude[i], fit_range_min, fit_range_max, fit_initial)
            parameters.append(parameters_temp)
            err.append(err_temp)
        return parameters, err
    if mode[0] == 'r':
        parameters, err = sin_decay_fitX_2D(
            x, data, freqmagnitude, fit_range_min, fit_range_max, fit_initial)
        return parameters, err
    else:
        x = np.transpose(np.array(x)).tolist()
        data = np.transpose(np.array(data)).tolist()
        parameters, err = sin_decay_fitX_2D(
            x, data, freqmagnitude, fit_range_min, fit_range_max, fit_initial)
        return parameters, err


def quadratic_fit_1D(x, data, fit_range_min=None, fit_range_max=None, fit_initial=None):
    """Quadratic function fitting. data = a * x * x + b * x + c


    Args:
        x (list[float]): Independent variable list.
        data (list[float]): Dependent variable list.
        fit_range_min (list[float], optional): The minimum values of the parameters to be fitted which has the form of [a_min , b_min , c_min]. Defaults to None.
        fit_range_max (list[float], optional): The maximum values of the parameters to be fitted which has the form of [a_max , b_max , c_max]. Defaults to None.
        fit_initial (list[float], optional): The initial values of the parameters to be fitted which has the form of [a_init , b_init , c_init]. Defaults to None.

    Returns:
        tuple[list[float],list[float]]: The fist list is fitted parameters [a_fitted , b_fitted , c_fitted], the second list is the error list.

    """
    def target_func(x, a, b, c):
        return a * x * x + b * x + c
    parameters, covariance = optimize.curve_fit(
        target_func, x, data, p0=fit_initial, bounds=(fit_range_min, fit_range_max), maxfev=500000)
    err = np.sqrt(np.diag(covariance)) * 1.96
    return parameters, err


def quadratic_fit_2D(x, data, fit_range_min=None, fit_range_max=None, fit_initial=None, mode="row"):
    """Quadratic function fitting. data = a * x * x + b * x + c
    If the first letter of mode was 'r', each row of data would be fitted once, 
    else each column of data would be fitted once.

    Args:
        x (list(list[float])): Independent variable list.
        data (list(list[float])): Dependent variable list.
        fit_range_min (list(list[float]), optional): The minimum values of the parameters to be fitted. Defaults to None.
        fit_range_max (list(list[float]), optional): The maximum values of the parameters to be fitted. Defaults to None.
        fit_initial (list(list[float]), optional): The initial values of the parameters to be fitted. Defaults to None.
        mode (str, optional): If the first letter of mode was 'r', each row of data would be fitted once, 
            else each column of data would be fitted once. Defaults to "row".

    Returns:
        tuple[list[list[float]],list[list[float]]]: The fist list is fitted parameters, the second list is the error list.

    """
    def quadratic_fitX_2D(x, data, fit_range_min=None, fit_range_max=None, fit_initial=None):
        parameters = []
        err = []
        for i in range(len(x)):
            parameters_temp, err_temp = quadratic_fit_1D(
                x[i], data[i], fit_range_min, fit_range_max, fit_initial)
            parameters.append(parameters_temp)
            err.append(err_temp)
        return parameters, err
    if mode[0] == 'r':
        parameters, err = quadratic_fitX_2D(
            x, data, fit_range_min, fit_range_max, fit_initial)
        return parameters, err
    else:
        x = np.transpose(np.array(x)).tolist()
        data = np.transpose(np.array(data)).tolist()
        parameters, err = quadratic_fitX_2D(
            x, data, fit_range_min, fit_range_max, fit_initial)
        return parameters, err

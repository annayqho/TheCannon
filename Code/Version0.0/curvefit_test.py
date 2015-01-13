import numpy as np
from scipy.optimize import curve_fit

# There are two possible solutions to this scipy.optimize.curve_fit problem

# Solution 1: call curve_fit using initial guesses

def func(x, *p): 
    return p[0]*np.exp(-p[1]*x)+p[2]

xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
ydata = y + 0.2 * np.random.normal(size=len(xdata))
popt, pcov = curve_fit(func, xdata, ydata, p0=(2, 1, 1))
a,b,c = popt[0], popt[1], popt[2]
fitted_y = func(xdata, a, b, c)

# Solution 2: dynamically define a function

func_string = """def func(x, *p): return p[0]+np.exp(-p[1]*x)+p[2]"""
exec(func_string)

xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
ydata = y + 0.2 * np.random.normal(size=len(xdata))
popt, pcov = curve_fit(func, xdata, ydata)
a,b,c = popt[0], popt[1], popt[2]
fitted_y = func(xdata, a, b, c)

import numpy as np
import scipy.stats
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib as mat
import pandas as pd
from matplotlib.widgets import SpanSelector

# LECTURE ON SPECTRAL PEAKS - GAUSSIAN, LORENTZIAN, ETC

def plot_curves(fig, ax, x, real_ydata, fit_ydata, *args):
    ''' Function created for plotting experimental curve data and fitted curve data.
    It needs the following command to be called before it:
    #fig, ax = plt.subplots()
    #It needs the following command to be called after it:
    #plt.show()
    
    ARGUMENTS:
    ax: your figure axis
    x: x data array, COMMON for all curves
    real_y_data: your experimental intensity data, it is the one fitted
    fit_ydata: the global fit function
    *args: as many y curve values as you want, they need to have the same shape
    
    RETURNS:
    All the curves plotted in one graph, small deconvoluted curves filled bellow them.
    '''
    fig.set_size_inches(14, 7)
    ax.scatter(x, real_ydata, s=30, label="Input Curve Data", c="r")
    ax.plot(x, fit_ydata, label="Output Sum Curve Fit", c="k", ls="--")
    count = 1
    for curve in args:
        ax.plot(x, curve, label="Curve {}".format(count))
        ax.fill_between(x, 0, curve, alpha=0.5)
        count = count + 1
    ax.legend()

def gaussian_n(x, *args):
    intensity = np.zeros(x.shape[0])
    count = 0
    for p in args:
        temp_intensity = p[0]*(1/(p[1]*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-p[2])/p[1])**2)))
        intensity = intensity + temp_intensity
        count = count + 1
    print("Constructed Gaussian as a sum of {} gaussians".format(count))
    return intensity

def lor_n(x, *args):
    intensity = np.zeros(x.shape[0])
    count = 0
    for p in args:
        temp_intensity = (p[0]) / (p[0]**2 + 4 * np.pi**2 * (x-p[1])**2)
        intensity = intensity + temp_intensity
        count = count + 1
    print("Constructed Lorentzian as a sum of {} lorentzians".format(count))
    return intensity
        
class slice_fun():
    '''Slice Function Class.
    Opens up a two subfigures plot in an interactive matplotlib backend. 
    It allows to select in the top subfigure the x-datapoints of interest, plotting the output datapoints in bottom subfigure. 
    If your backend is not interactive, the Class won't work. 
    You can change backend with a line of code before calling this class:
    %matplotlib qt
    %matplotlib notebook
    Remember to call:
    %matplotlib inline
    After using this function.
    
    ARGUMENTS:
    X: x-data array
    Y: y-data array
    
    RETURNS:
    class.newx: selected x-array data points
    class.newy: selected y-array data points
    '''
    def __init__(self, X, Y):
        def onselect_x(xmin, xmax):
            indmin, indmax = np.searchsorted(self.X, (xmin, xmax))
            indmax = min(len(self.X) - 1, indmax)

            self.newx = self.X[indmin:indmax]
            self.newy = self.Y[indmin:indmax]
            self.line2.set_data(self.newx, self.newy)
            self.ax2.set_xlim(self.newx[0], self.newx[-1])
            self.ax2.set_ylim(self.newy.min()-self.newy.max()*0.05, self.newy.max()+self.newy.max()*0.05)
            self.fig.canvas.draw()
            print("You have the following X limits:")
            print("X min: {}".format(self.newx.min()))
            print("X max: {}".format(self.newx.max()))
        
        self.X = X
        self.Y = Y
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(7, 7))
        self.ax1.plot(self.X, self.Y)
        self.line2, = self.ax2.plot(self.X, self.Y)
        self.ax1.set_xlabel("Frequency / Hz")
        self.ax1.set_ylabel("Normalized Intensity")
        self.slide_x = np.array([])
        self.slide_y = np.array([])
        self.span = SpanSelector(self.ax1, onselect=onselect_x,
                            direction="horizontal", minspan=20, useblit=True, span_stays=True, button=1,
                            rectprops={"facecolor":"red", "alpha":0.3})
        plt.show()

        
    
    
# LECTURE ON LINEAR REGRESSION

class LRegression():
    '''This is the Class for the Linear Regression model.
    FIT THE MODEL
    .fit(X, y, alpha=0.05) - function performs a linear fit over X-y data.
    
    RETURNS
    .params_  - gives you the parameters with their uncertainty.
    .rsquare_ - gives you the r2 value
    
    DEFINED FUNCTIONS
    .plot() - plot the data + fit line
    .predict_y(x_pred) predicts the dependant value based on X input data.'''
    
    def fit(self, X, y, alpha=0.05):
        '''Create the model fit.
        
        INPUT:
        X = x data array
        y = y data array
        alpha = t_student confidant value, default set to 0
        ------
        RETURNS:
        .params_  - Numpy array, gives you the parameters with their uncertainty.
        .rsquare_ - Integer, gives you the r2 value'''
        X2, y2 = np.square(X), np.square(y)
        Xy = np.multiply(X, y)
        self.N = len(X)      
        self.EX = np.sum(X2) - np.square(np.sum(X))/self.N
        Ey = np.sum(y2) - np.square(np.sum(y))/self.N
        SXy = np.sum(Xy) - np.sum(X) * np.sum(y) / self.N
        m = SXy / self.EX #slope
        n = np.sum(y) / self.N - m * np.sum(X) / self.N #independant
        SXy2 = (Ey - np.square(m) * self.EX) / (self.N-2)
        Sm = np.sqrt(SXy2 / self.EX)
        Sn = np.sqrt(SXy2 * np.sum(X2) / (self.N * self.EX))
        
        t = scipy.stats.t.ppf(1-(alpha/2), self.N-2)
        
        m_error = t * Sm
        n_error = t * Sn
        
        r = SXy / np.sqrt(self.EX * Ey)
        self.rsquare_ = np.square(r)
        self.input_ = np.array([X, y])
        self.params_ = np.array([m, m_error, n, n_error])
        self.y_mean = np.mean(y)
        self.X_mean = np.mean(X)
        self.SXy = SXy
        
    def plot(self):
        x_pred = np.linspace(self.input_[0].min(), self.input_[0].max(), 15)
        y_pred = self.params_[2] + self.params_[0] * x_pred
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.input_[0], y=self.input_[1], mode='markers',
                                 name="Experimental data"))
        fig.add_trace(go.Scatter(x=x_pred, y=y_pred, mode="lines", name="Prediction line"))
        fig.update_layout({"title": {"text": "Linear Regression Data and Fit",
                             "font": {"size": 30}}})
        fig.update_xaxes(title_text='X - Dependant variable')
        fig.update_yaxes(title_text='Y - Independant variable')
        fig.show()
        

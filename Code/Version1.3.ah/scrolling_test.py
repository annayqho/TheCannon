from PySide import QtGui, QtCore
import pylab as plt
import numpy as np

N_SAMPLES = 1e6

def test_plot():
    time = np.arange(N_SAMPLES)*1e-3
    sample = np.random.randn(N_SAMPLES)
    plt.plot(time, sample, label="Gaussian noise")
    plt.legend(fancybox=True)
    plt.title("Use the slider to scroll and the spin-box to set the width")
    q = ScrollingToolQT(plt.gcf())
    return q

class ScrollingToolQT(object):
    def __init__(self, fig):
        self.fig = fig
        self.xmin, self.xmax = fig.axes[0].get_xlim()
        self.step = 1 # axis units
        self.scale = 1e3
        QMainWin = fig.canvas.parent()
        toolbar = QtGui.QToolBar(QMainWin)
        QMainWin.addToolBar(QtCore.Qt.BottomToolBarArea, toolbar)
        self.set_slider(toolbar)
        self.set_spinbox(toolbar)
        self.set_xlim = self.fig.axes[0].set_xlim
        self.draw_idle = self.fig.canvas.draw_idle
        self.ax = self.fig.axes[0]
        self.set_xlim(0, self.step)
        self.fig.canvas.draw()
    
    def set_slider(self, parent):
        smin, smax = self.xmin*self.scale, self.xmax*self.scale
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, parent=parent)
        self.slider.setTickPosition(QtGui.QSlider.TicksAbove)
        self.slider.setTickInterval((smax-smin)/10.)
        self.slider.setMinimum(smin)
        self.slider.setMaximum(smax-self.step*self.scale)
        self.slider.setSingleStep(self.step*self.scale/5.)
        self.slider.setPageStep(self.step*self.scale)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.xpos_changed)
        parent.addWidget(self.slider)

    def set_spinbox(self, parent):
        self.spinb = QtGui.QDoubleSpinBox(parent=parent)
        self.spinb.setDecimals(3)
        self.spinb.setRange(0.001, 3600.)
        self.spinb.setSuffix(" s")
        self.spinb.setValue(self.step)   # set the initial width
        self.spinb.valueChanged.connect(self.xwidth_changed)
        parent.addWidget(self.spinb)

    def xpos_changed(self, pos):
        pos /= self.scale
        self.set_xlim(pos, pos + self.step)
        self.draw_idle()

    def xwidth_changed(self, xwidth):
        if xwidth <= 0: return
        self.step = xwidth
        self.slider.setSingleStep(self.step*self.scale/5.)
        self.slider.setPageStep(self.step*self.scale)
        old_xlim = self.ax.get_xlim()
        self.xpos_changed(old_xlim[0] * self.scale)

if __name__ == "__main__":
    q = test_plot()
    plt.show()

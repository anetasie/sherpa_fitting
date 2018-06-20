import numpy as np
from sherpa.models import Gauss2D
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel, Tophat2DKernel
from astropy.convolution import CustomKernel

def rebin_int(array,factorx,factory):
    xedge = np.shape(array)[0]%factorx
    yedge = np.shape(array)[1]%factory
    array_binned1 = array[xedge:,yedge:]

    binim = np.reshape(array_binned1,
                       (np.shape(array_binned1)[0]//factorx,factorx,np.shape(array_binned1)[1]//factory,factory))
    binim = np.sum(binim,axis=3)
    binim = np.sum(binim,axis=1)

    return binim


class CountsModel:
    def __init__(self, psf_model):
        self.psf_model = psf_model

    def __call__(self, pars, *args):
        return counts_func(self.psf_model, pars)


def counts_func(psf_data, pars):
    (fwhm, xpos, ypos, ellip, theta, ampl) = pars
    x0low, x0high = 0,21
    x1low, x1high = 0,21
    dx = 1
    x1, x0 = np.mgrid[x1low:x1high:dx, x0low:x0high:dx]

    # Convert to 1D arrays
    shape = x0.shape
    x0, x1 = x0.flatten(), x1.flatten()

    truth = Gauss2D()
    truth.ampl = ampl
    truth.xpos, truth.ypos = xpos, ypos
    truth.fwhm, truth.ellip = fwhm, ellip
    truth.theta = theta  
    
    #  This is where we convolve and add a Poisson background
    kernel = CustomKernel(psf_data)
    astropy_conv = convolve_fft(truth(x0, x1).reshape(shape), kernel,
                                normalize_kernel=True)
    
    # Now we rebin
    model_rebin = rebin_int(astropy_conv,1,1) 

    return model_rebin.flatten()
    

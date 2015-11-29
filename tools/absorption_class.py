import spectral_functions
from scipy.optimize import curve_fit
import numpy as np


class AbsorptionSpectrum(object):
	"""
    A collection of up to 5 gaussian peaks with independent parameters.

	Methods:
    generate_spectrum():
        builds the absorption spectrum based on the number of peaks
        and the provided peak_params.

    info():
        print the number of peaks and the current peak parameters

    fit(xvals, yvals):
        fits the peak parameters to the data provided

    predict(xvals, yvals):
        returns a scale factor for a spectrum from a different concentration

    get_fit_vals(xvals):
        returns an array with the fitted spectrum
    """
	SPECTRUM_MAPPING = {
		1: spectral_functions.gaussian1,
		2: spectral_functions.gaussian2,
		3: spectral_functions.gaussian3,
		4: spectral_functions.gaussian4,
		5: spectral_functions.gaussian5,
	}

	def __init__(self, n_peaks, guessed_params):
		self.spectrum = None
		self.n_peaks = n_peaks
		self.peak_params = np.concatenate((np.array(guessed_params),
		                                   np.array([0.0])))
		self.cov = 0
		self.h = 1.0

	def generate_spectrum(self):
		self.spectrum = self.SPECTRUM_MAPPING[self.n_peaks]

	def info(self):
		print ' n_peaks: %i\n guessed_params: ' % self.n_peaks, \
			self.peak_params

	def add_peak(self, guessed_params):
		"""
		adds a peak to the spectrum
		:param guessed_params:
			guessed parameters for added peak: [height, mean, variance]
		"""
		self.n_peaks += 1
		orig_params = self.peak_params
		self.peak_params = np.concatenate((np.array(guessed_params),
		                                   orig_params), axis=0)
		self.generate_spectrum()

	def fit(self, xvals, yvals):
		"""
		optimizes all individual peak parameters to fit the spectrum provided.
        :param xvals:
            length n array of wavelengths
        :param yvals:
            length n array of absorbances
        """
		self.generate_spectrum()
		self.peak_params, self.cov = curve_fit(
			self.spectrum,
			xvals,
			yvals,
			self.peak_params
		)

	def predict(self, xvals, yvals):
		"""
        optimizes a scale factor to fit the provided spectrum
        :param xvals:
            length n array of wavelengths
        :param yvals:
            length n array of absorbances
        :return:
            h scale factor (float)
        """

		def fn(xvals, h):
			return h * self.spectrum(xvals, *self.peak_params)

		self.h, _ = curve_fit(fn, xvals, yvals)

		return self.h

	def get_fit_vals(self, xvals):
		"""
		show the predicted spectrum
        :param xvals:
            length n array of wavelengths
        :return:
            length n array of predicted absorbances
        """
		return self.spectrum(xvals, *self.peak_params)


class SecondSpectrum(AbsorptionSpectrum):
	def __init__(self, first_spectrum, n_peaks, guessed_params):
		self.first_spectrum = first_spectrum
		self.spectrum = None
		self.n_peaks = n_peaks
		self.peak_params = np.concatenate((np.array(guessed_params),
		                                   np.array([0.0])))
		self.cov = 0
		self.h = 1.0

	def fit_second_spectrum(self, xvals, yvals):
		self.generate_spectrum()

		def func(xvals, h1, *peak_params):
			return (
				h1 * self.first_spectrum.get_fit_vals(xvals) +
				self.spectrum(xvals, *peak_params)
			)

		h1 = np.array([1.0])
		guess = np.concatenate((h1, self.peak_params))
		vals, _ = curve_fit(func, xvals, yvals, guess)
		self.peak_params = vals[1:]
		self.h1 = vals[0]

	def get_fit_vals(self, xvals):
		self.generate_spectrum()

		def func(xvals, h1, *peak_params):
			return (
				h1 * self.first_spectrum.get_fit_vals(xvals) +
				self.spectrum(xvals, *peak_params)
			)

		return func(xvals, self.h1, *self.peak_params)

# for testing
if __name__ == '__main__':
	import numpy as np
	from spectral_functions import gaussian1 as gauss
	import matplotlib.pyplot as plt

	npts = 500
	xvals = np.linspace(0, 30, npts)
	yvals = gauss(xvals, 15, 4, 2, 0)
	yvals += gauss(xvals, 7, 18, 1, 0)
	yvals += np.random.randn(npts) + 1.13

	p1 = AbsorptionSpectrum(1, [13, 3.5, 2])
	p1.fit(xvals, yvals)
	fit1 = p1.get_fit_vals(xvals)

	p2 = SecondSpectrum(p1, 1, [7, 18, 1])
	p2.fit_second_spectrum(xvals, yvals)
	fit2 = p2.get_fit_vals(xvals)

	plt.figure()
	plt.plot(xvals, yvals, '.', alpha=0.7, label='simulated spectrum')
	plt.plot(xvals, fit1, color='red', linewidth=2, alpha=0.6,
	         label='fit with 1 peak')
	plt.plot(xvals, fit2, color='blue', linewidth=2, alpha=0.6,
	         label='fit with 2 peaks')
	plt.legend()
	plt.show()


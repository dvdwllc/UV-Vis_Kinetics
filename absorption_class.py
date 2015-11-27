from tools import spectral_functions
from scipy.optimize import curve_fit
import numpy as np


class absorption_spectrum(object):
	"""
	A collection of up to 5 gaussian peaks with independent parameters.

	.info():
		print the number of peaks and the current peak parameters

	.fit(xvals, yvals):
		fits the peak parameters to the spectrum provided

	.predict(xvals, yvals):
		returns a scale factor for a spectrum from a different concentration

	.get_fit_vals(xvals):
		returns an array with the fitted spectrum
	"""
	def __init__(self, n_peaks, guessed_params):
		self.n_peaks = n_peaks
		self.peak_params = guessed_params
		self.cov = 0
		self.h = 1.0


	def info(self):
		print ' n_peaks: %i\n guessed_params: ' % self.n_peaks, self.peak_params

	def add_peak(self, guessed_params):
		"""
		adds a peak to the spectrum
		:param guessed_params:
			guessed parameters for added peak: [height, mean, variance]
		"""
		self.n_peaks += 1
		orig_params = self.peak_params
		self.peak_params = np.array([orig_params, guessed_params]).flatten()
		# print 'peak added'

	def fit(self, xvals, yvals):
		"""
		optimizes all individual peak parameters to fit the spectrum provided.
		:param xvals:
			length n array of wavelengths
		:param yvals:
			length n array of absorbances
		"""
		if self.n_peaks == 1:
			self.peak_params, self.cov = curve_fit(
				spectral_functions.gaussian1,
				xvals,
				yvals,
				self.peak_params
			)
		elif self.n_peaks == 2:
			self.peak_params, self.cov = curve_fit(
				spectral_functions.gaussian2,
				xvals,
				yvals,
				self.peak_params
			)
		elif self.n_peaks == 3:
			self.peak_params, self.cov = curve_fit(
				spectral_functions.gaussian3,
				xvals,
				yvals,
				self.peak_params
			)
		elif self.n_peaks == 4:
			self.peak_params, self.cov = curve_fit(
				spectral_functions.gaussian4,
				xvals,
				yvals,
				self.peak_params
			)
		elif self.n_peaks == 5:
			self.peak_params, self.cov = curve_fit(
				spectral_functions.gaussian5,
				xvals,
				yvals,
				self.peak_params
			)
		else:
			print 'invalid number of peaks!'

	def predict(self, xvals, yvals):
		"""
		optimizes a scale factor to fit the provided spectrum
		:param xvals:
			length n array of wavelengths
		:param yvals:
			length n array of absorbances
		:return:
			scale factor (float)
		"""
		if len(self.peak_params) == 1:
			print 'function not yet fit!'
			return 0

		if self.n_peaks == 1:
			def function(xvals, h):
				return h*spectral_functions.gaussian1(xvals, *self.peak_params)
			self.h, _ = curve_fit(
				function,
				xvals,
				yvals
			)
		elif self.n_peaks == 2:
			def function(xvals, h):
				return h*spectral_functions.gaussian2(xvals, *self.peak_params)
			self.h, _ = curve_fit(
				function,
				xvals,
				yvals
			)
		elif self.n_peaks == 3:
			def function(xvals, h):
				return h*spectral_functions.gaussian3(xvals, *self.peak_params)
			self.h, _ = curve_fit(
				function,
				xvals,
				yvals
			)
		elif self.n_peaks == 4:
			def function(xvals, h):
				return h*spectral_functions.gaussian4(xvals, *self.peak_params)
			self.h, _ = curve_fit(
				function,
				xvals,
				yvals
			)
		elif self.n_peaks == 5:
			def function(xvals, h):
				return h*spectral_functions.gaussian5(xvals, *self.peak_params)
			self.h, _ = curve_fit(
				function,
				xvals,
				yvals
			)
		print self.h
		return self.h

	def get_fit_vals(self, xvals):
		"""
		show the predicted spectrum
		:param xvals:
			length n array of wavelengths
		:return:
			length n array of predicted absorbances
		"""
		if self.n_peaks == 1:
			return spectral_functions.gaussian1(xvals, *self.peak_params)
		elif self.n_peaks == 2:
			return spectral_functions.gaussian2(xvals, *self.peak_params)
		elif self.n_peaks == 3:
			return spectral_functions.gaussian3(xvals, *self.peak_params)
		elif self.n_peaks == 4:
			return spectral_functions.gaussian4(xvals, *self.peak_params)
		elif self.n_peaks == 5:
			return spectral_functions.gaussian5(xvals, *self.peak_params)
		else:
			print 'invalid number of peaks!'

if __name__ == '__main__':
	import numpy as np
	from tools.spectral_functions import gaussian1 as gauss
	import matplotlib.pyplot as plt

	npts = 500
	xvals = np.linspace(0, 30, npts)
	yvals = gauss(xvals, 15, 4, 2)
	yvals += gauss(xvals, 7, 18, 1)
	yvals += np.random.randn(npts)

	p = absorption_spectrum(1, [13, 3.5, 2])
	p.fit(xvals,yvals)
	fit1 = p.get_fit_vals(xvals)

	p.add_peak([7, 18, 2])
	p.fit(xvals,yvals)
	fit2 = p.get_fit_vals(xvals)

	plt.figure()
	plt.plot(xvals, yvals, '.', alpha=0.7, label='simulated spectrum')
	plt.plot(xvals, fit1, color='red', linewidth=2, alpha=0.6, label='fit with 1 peak')
	plt.plot(xvals, fit2, color='blue', linewidth=2, alpha=0.6,  label='fit with 2 peaks')
	plt.legend()
	plt.show()

	yvals *= 0.75
	p.predict(xvals, yvals)

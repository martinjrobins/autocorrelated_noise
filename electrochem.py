from __future__ import print_function
import pints
import sys
import electrochemistry
import pints.plot
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy
import scipy.stats
from scipy.stats import invwishart
import pickle
import os.path
from math import pi, sqrt


class AR1LogLikelihood(pints.ProblemLogLikelihood):
    """
    Calculates a log-likelihood assuming AR1 noise model

    Arguments:

    ``problem``
        A :class:`SingleOutputProblem` or :class:`MultiOutputProblem`. For a
        single-output problem a single parameter is added, for a multi-output
        problem ``n_outputs`` parameters are added.

    *Extends:* :class:`ProblemLogLikelihood`
    """

    def __init__(self, problem):
        super(AR1LogLikelihood, self).__init__(problem)

        # Get number of times, number of outputs
        self._nt = len(self._times)-1
        self._no = problem.n_outputs()

        # Add parameters to problem
        self._n_parameters = problem.n_parameters() + 2*self._no

        # Pre-calculate parts
        self._logn = 0.5 * (self._nt) * np.log(2 * np.pi)

    def __call__(self, x):
        sigma = np.asarray(x[-2*self._no:-self._no])
        rho = np.asarray(x[-self._no:])
        error = self._values - self._problem.evaluate(x[:-self._no])
        autocorr_error = error[1:] - rho*error[:-1]
        return np.sum(- self._logn - self._nt * np.log(sigma)
                      - np.sum(autocorr_error**2, axis=0) / (2 * sigma**2))


DEFAULT = {
    'reversed': True,
    'Estart': 0.5,
    'Ereverse': -0.1,
    'omega': 9.0152,
    'phase': 0,
    'dE': 0.08,
    'v': -0.08941,
    't_0': 0.001,
    'T': 297.0,
    'a': 0.07,
    'c_inf': 1 * 1e-3 * 1e-3,
    'D': 7.2e-6,
    'Ru': 8.0,
    'Cdl': 20.0 * 1e-6,
    'E0': 0.214,
    'k0': 0.0101,
    'alpha': 0.53,
}

filenames = ['GC01_FeIII-1mM_1M-KCl_02_009Hz.txt',
             'GC02_FeIII-1mM_1M-KCl_02a_009Hz.txt',
             'GC03_FeIII-1mM_1M-KCl_02_009Hz.txt',
             'GC04_FeIII-1mM_1M-KCl_02_009Hz.txt',
             'GC05_FeIII-1mM_1M-KCl_02_009Hz.txt',
             'GC06_FeIII-1mM_1M-KCl_02_009Hz.txt',
             'GC07_FeIII-1mM_1M-KCl_02_009Hz.txt',
             'GC08_FeIII-1mM_1M-KCl_02_009Hz.txt',
             'GC09_FeIII-1mM_1M-KCl_02_009Hz.txt',
             'GC10_FeIII-1mM_1M-KCl_02_009Hz.txt']

model = electrochemistry.ECModel(DEFAULT)
data0 = electrochemistry.ECTimeData(
    filenames[0], model, ignore_begin_samples=5, ignore_end_samples=0,
    samples_per_period=5000)
max_current = np.max(data0.current)
sim_current = model.simulate(data0.times)
plt.plot(data0.times, data0.current, label='exp')
plt.plot(data0.times, sim_current, label='sim')
plt.legend()
plt.savefig('default.pdf')
max_k0 = model.non_dimensionalise(1000, 'k0')
e0_buffer = 0.1 * (model.params['Ereverse'] - model.params['Estart'])
names = ['k0', 'E0', 'Cdl', 'Ru', 'alpha']
true = [model.params[name] for name in names]

lower_bounds = [
    0.0,
    model.params['Estart'] + e0_buffer,
    0.0,
    0.0,
    0.4,
    0.001 * max_current,
    0.0,
]

upper_bounds = [
    100 * model.params['k0'],
    model.params['Ereverse'] - e0_buffer,
    10 * model.params['Cdl'],
    10 * model.params['Ru'],
    0.6,
    0.03 * max_current,
    1.0,
]

print('lower true upper')
for u, l, t in zip(upper_bounds, lower_bounds, true):
    print(l, ' ', t, ' ', u)
print(lower_bounds[-1], ' ', -1, ' ', upper_bounds[-1])


# Load a forward model
pints_model = electrochemistry.PintsModelAdaptor(model, names)
values = pints_model.simulate(true, data0.times)
print(values.shape)
plt.clf()
plt.plot(data0.times, data0.current, label='exp')
plt.plot(data0.times, values, label='sim')
plt.legend()
plt.savefig('default_pints.pdf')


nexp = len(filenames)

# cmaes params
x0 = np.array([0.5 * (u + l) for l, u in zip(lower_bounds, upper_bounds)])
sigma0 = [0.5 * (h - l) for l, h in zip(lower_bounds, upper_bounds)]

# parameters = np.zeros((samples,len(mean)))
# values = np.zeros((samples,len(times)))

log_posteriors = []
pickle_file = 'log_posteriors.pickle'
if not os.path.isfile(pickle_file):
    for i, filename in enumerate(filenames):

        data = electrochemistry.ECTimeData(
            filename, model, ignore_begin_samples=5, ignore_end_samples=0,
            samples_per_period=5000)

        current = data.current
        times = data.times

        problem = pints.SingleOutputProblem(pints_model, times, current)

        # Create a new log-likelihood function
        log_likelihood = AR1LogLikelihood(problem)

        # Create a new prior
        log_prior = pints.UniformLogPrior(lower_bounds, upper_bounds)

        # Create a posterior log-likelihood (log(likelihood * prior))
        log_posterior = pints.LogPosterior(log_likelihood, log_prior)
        log_posteriors.append(log_posterior)
    pickle.dump(log_posteriors, open(pickle_file, 'wb'))
else:
    log_posteriors = pickle.load(open(pickle_file, 'rb'))


pickle_file = 'fit_parameters.pickle'
if not os.path.isfile(pickle_file):
    fit_parameters = []
    for i, filename in enumerate(filenames):
        log_posterior = log_posteriors[i]
        score = pints.ProbabilityBasedError(log_posterior)
        boundaries = pints.RectangularBoundaries(lower_bounds, upper_bounds)

        found_parameters, found_value = pints.optimise(
            score,
            x0,
            sigma0,
            boundaries,
            method=pints.CMAES
        )
        fit_parameters.append(found_parameters)

    pickle.dump(fit_parameters, open(pickle_file, 'wb'))
else:
    fit_parameters = pickle.load(open(pickle_file, 'rb'))

print('using starting points:')
for i, found_parameters in enumerate(fit_parameters):
    print('\t', found_parameters)

    plt.clf()
    log_posterior = log_posteriors[i]
    times = log_posterior._log_likelihood._problem._times
    values = log_posterior._log_likelihood._problem._values
    sim_values = log_posterior._log_likelihood._problem.evaluate(
        found_parameters)
    E_0, T_0, L_0, I_0 = model._calculate_characteristic_values()
    dim_values = I_0*values*1e6
    dim_sim_values = I_0*sim_values*1e6
    plt.plot(T_0*times, dim_values, label='experiment')
    plt.plot(T_0*times, dim_sim_values, label='simulation')
    plt.xlabel(r'$t$ (s)')
    plt.ylabel(r'$I_{tot}$ ($\mu A$)')
    plt.legend()
    filename = filenames[i]
    print('plotting fit for ', filename)
    plt.savefig('fit%s.pdf' % filename)

    plt.clf()
    residuals = dim_values - dim_sim_values
    plt.plot(T_0*times, residuals, label='residuals')
    plt.xlabel(r'$t$ (s)')
    plt.ylabel(r'$I_{tot}-I{sim}$ ($\mu A$)')
    print('plotting residual versus time for ', filename)
    plt.savefig('residual_v_time_%s.pdf' % filename)

    plt.clf()
    plt.hist(residuals, bins=100)
    plt.xlabel(r'$I_{tot}$ ($\mu A$)')
    print('plotting residual histogram for ', filename)
    plt.savefig('residual_histogram_%s.pdf' % filename)

    plt.clf()
    autocorr = np.correlate(residuals, residuals, mode='full')
    autocorr = autocorr[autocorr.size//2:]
    autocorr = autocorr[:400]/autocorr[0]
    plt.plot(autocorr, label='autocorrelation of residuals')
    plt.xlabel(r'index')
    plt.ylabel(r'autocorrelation')
    print('plotting residual autocorrelation for ', filename)
    plt.savefig('residual_autocorrelation_%s.pdf' % filename)

    plt.clf()
    plt.plot(autocorr[:10], label='autocorrelation of residuals')
    plt.xlabel(r'index')
    plt.ylabel(r'autocorrelation')
    print('plotting residual autocorrelation zoom for ', filename)
    plt.savefig('residual_autocorrelation_zoom_%s.pdf' % filename)

    plt.clf()
    plt.scatter(residuals[:-1], residuals[1:], label='residuals')
    plt.xlabel(r'$I_{tot}^{t-1}-I{sim}^{t-1}$ ($\mu A$)')
    plt.ylabel(r'$I_{tot}^{t}-I{sim}^{t}$ ($\mu A$)')
    print('plotting residual versus residual for ', filename)
    plt.savefig('residual_v_residual_%s.pdf' % filename)

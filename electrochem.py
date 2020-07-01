from __future__ import print_function
import pints
import sys
import electrochemistry
import pints.plot
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import pickle
import os.path
from math import pi, sqrt


def plot_fits(fit_parameters, log_posteriors, filenames, ext):
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
        plt.savefig(ext + 'fit%s.pdf' % filename)

        plt.clf()
        residuals = dim_values - dim_sim_values
        np.savetxt(ext + 'residuals_v_time_{}.csv'.format(filename), (times,residuals))
        plt.plot(T_0*times, residuals, label='residuals')
        plt.xlabel(r'$t$ (s)')
        plt.ylabel(r'$I_{tot}-I{sim}$ ($\mu A$)')
        print('plotting residual versus time for ', filename)
        plt.savefig(ext + 'residual_v_time_%s.pdf' % filename)

        plt.clf()
        plt.hist(residuals, bins=100)
        plt.xlabel(r'$I_{tot}$ ($\mu A$)')
        print('plotting residual histogram for ', filename)
        plt.savefig(ext + 'residual_histogram_%s.pdf' % filename)

        plt.clf()
        autocorr = np.correlate(residuals, residuals, mode='full')
        autocorr = autocorr[autocorr.size//2:]
        autocorr = autocorr[:400]/autocorr[0]
        plt.plot(autocorr, label='autocorrelation of residuals')
        plt.xlabel(r'index')
        plt.ylabel(r'autocorrelation')
        print('plotting residual autocorrelation for ', filename)
        plt.savefig(ext + 'residual_autocorrelation_%s.pdf' % filename)

        plt.clf()
        plt.plot(autocorr[:10], label='autocorrelation of residuals')
        plt.xlabel(r'index')
        plt.ylabel(r'autocorrelation')
        print('plotting residual autocorrelation zoom for ', filename)
        plt.savefig(ext + 'residual_autocorrelation_zoom_%s.pdf' % filename)

        plt.clf()
        plt.scatter(residuals[:-1], residuals[1:], label='residuals')
        plt.xlabel(r'$I_{tot}^{t-1}-I{sim}^{t-1}$ ($\mu A$)')
        plt.ylabel(r'$I_{tot}^{t}-I{sim}^{t}$ ($\mu A$)')
        print('plotting residual versus residual for ', filename)
        plt.savefig(ext + 'residual_v_residual_%s.pdf' % filename)



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
             'GC03_FeIII-1mM_1M-KCl_02_009Hz.txt']
#'GC04_FeIII-1mM_1M-KCl_02_009Hz.txt',
#'GC05_FeIII-1mM_1M-KCl_02_009Hz.txt',
#'GC06_FeIII-1mM_1M-KCl_02_009Hz.txt',
#'GC07_FeIII-1mM_1M-KCl_02_009Hz.txt',
#'GC08_FeIII-1mM_1M-KCl_02_009Hz.txt',
#'GC09_FeIII-1mM_1M-KCl_02_009Hz.txt',
#'GC10_FeIII-1mM_1M-KCl_02_009Hz.txt']

model = electrochemistry.ECModel(DEFAULT)
data0 = electrochemistry.ECTimeData(
    filenames[0], model, ignore_begin_samples=5, ignore_end_samples=0, samples_per_period=5000)
max_current = np.max(data0.current)
sim_current = model.simulate(data0.times)
plt.plot(data0.times, data0.current, label='exp')
plt.plot(data0.times, sim_current, label='sim')
plt.legend()
plt.savefig('default.pdf')
max_k0 = model.non_dimensionalise(1000, 'k0')
e0_buffer = 0.1 * (model.params['Ereverse'] - model.params['Estart'])
names = ['k0', 'E0', 'Cdl', 'Ru', 'alpha', 'omega']
true = [model.params[name] for name in names]

lower_bounds = [
    0.0,
    model.params['Estart'] + e0_buffer,
    0.0,
    0.0,
    0.4,
    0.9* model.params['omega'],
]
upper_bounds = [
    100 * model.params['k0'],
    model.params['Ereverse'] - e0_buffer,
    10 * model.params['Cdl'],
    10 * model.params['Ru'],
    0.6,
    1.1* model.params['omega'],
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

# parameters = np.zeros((samples,len(mean)))
# values = np.zeros((samples,len(times)))

log_posteriors_ar1 = []
log_posteriors_iid = []
pickle_file = 'log_posteriors.pickle'
if not os.path.isfile(pickle_file):
    for i, filename in enumerate(filenames):

        data = electrochemistry.ECTimeData(
            filename, model, ignore_begin_samples=5, ignore_end_samples=0, samples_per_period=5000)

        current = data.current
        times = data.times

        problem = pints.SingleOutputProblem(pints_model, times, current)

        # Create a new log-likelihood function
        log_likelihood_ar1 = pints.AR1LogLikelihood(problem)
        log_likelihood_iid = pints.GaussianLogLikelihood(problem)

        # Create a new prior
        log_prior_ar1 = pints.UniformLogPrior(lower_bounds+[0,0.001*max_current],
                upper_bounds+[1,0.03*max_current])
        log_prior_iid = pints.UniformLogPrior(lower_bounds+[0.001*max_current],
                upper_bounds+[0.03*max_current])

        # Create a posterior log-likelihood (log(likelihood * prior))
        log_posterior_ar1 = pints.LogPosterior(log_likelihood_ar1, log_prior_ar1)
        log_posterior_iid = pints.LogPosterior(log_likelihood_iid, log_prior_iid)
        log_posteriors_ar1.append(log_posterior_ar1)
        log_posteriors_iid.append(log_posterior_iid)
    pickle.dump((log_posteriors_ar1,log_posteriors_iid), open(pickle_file, 'wb'))
else:
    log_posteriors_ar1, log_posteriors_iid = pickle.load(open(pickle_file, 'rb'))


pickle_file = 'fit_parameters.pickle'
if not os.path.isfile(pickle_file):
    fit_parameters_ar1 = []
    fit_parameters_iid = []
    for log_posteriors,fit_parameters in zip([log_posteriors_ar1, log_posteriors_iid],
                                            [fit_parameters_ar1, fit_parameters_iid]):
        for i, filename in enumerate(filenames):
            log_posterior = log_posteriors[i]
            lower_bounds = log_posterior._log_prior._boundaries.lower()
            upper_bounds = log_posterior._log_prior._boundaries.upper()
            x0 = np.array([0.5 * (u + l) for l, u in zip(lower_bounds, upper_bounds)])
            sigma0 = [0.5 * (h - l) for l, h in zip(lower_bounds, upper_bounds)]
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

    pickle.dump((fit_parameters_ar1, fit_parameters_iid), open(pickle_file, 'wb'))
else:
    fit_parameters_ar1,fit_parameters_iid = pickle.load(open(pickle_file, 'rb'))

plot_fits(fit_parameters_ar1, log_posteriors_ar1, filenames, 'ar1')
plot_fits(fit_parameters_iid, log_posteriors_iid, filenames, 'iid')

#pickle_file = 'all_chains.pickle'
#if not os.path.isfile(pickle_file):
#    all_chains = []
#    for i, log_posterior in enumerate(log_posteriors):
#        nchains = 5
#        xs = [
#            fit_parameters[i]*1.1,
#            fit_parameters[i]*1.05,
#            fit_parameters[i]*1.04,
#            fit_parameters[i]*0.95,
#            fit_parameters[i]*0.98,
#        ]
#        print(xs)
#        mcmc = pints.MCMCController(log_posterior, nchains, xs,
#                                  method=pints.HaarioBardenetACMC)
#        mcmc.set_parallel(True)
#        iters = 10000
#        mcmc.set_max_iterations(iters)
#        chains = mcmc.run()
#        # Run!
#        print('Running...')
#        chains = mcmc.run()
#        print('Done!')
#        all_chains.append(chains)
#
#    pickle.dump(all_chains, open(pickle_file, 'wb'))
#else:
#    all_chains = pickle.load(open(pickle_file, 'rb'))
#
#chains = np.empty(
#    (len(all_chains)-1, all_chains[0].shape[1]//2, all_chains[0].shape[2]))
#print(all_chains[0].shape)
#for i, c in enumerate(all_chains[1:]):
#    chains[i, :, :] = c[0, c.shape[1]//2:, :]
#
#pints.plot.trace(chains)
#plt.savefig('trace_all.pdf')

# for i, chains in enumerate(all_chains):
#    chains = chains[:, chains.shape[1]//2:, :]
#
#    pints.plot.trace(chains)
#    print('R-hat:')
#    print(pints.rhat_all_params(chains))
#
#    plt.savefig('trace_%s.pdf' % filenames[i])

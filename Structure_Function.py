import numpy as np
from scipy.optimize import leastsq, minimize
import pandas as pd
from matplotlib import pyplot as plt
from string import atof
from math import isnan
from javelin.emcee import EnsembleSampler
import scipy.io as sio
import corner



def SF_PL(params, del_t):
	A = np.exp(params[0])
	gamma = params[1]
	# print "parameters: ", params
	# print "SF_PL: ", A * (abs(del_t) / 365.25)**gamma
	return abs(A) * (abs(del_t) / 365.25)**gamma

def SF_DRW(params, del_t):
	sigma, tau = params[0:2]
	return 2.0**0.5 * abs(sigma) * (1 - np.exp(-abs(del_t) / tau))**0.5

# c.f. Schmidt et al. (2010)
def lnLij(MJD, mag, err, params, i, j, model="DRW"):
	del_t = MJD[j] - MJD[i]
	del_mag = mag[j] - mag[i]
	# print "del_t: ", del_t, " del_mag: ", del_mag
	if model == "pow-law":
		V_eff2 = SF_PL(params, del_t)**2 + (err[i]**2 + err[j]**2)
		# print "V_eff: ", V_eff
		# yyy = raw_input('just for pause')
	elif model == "DRW":
		V_eff2 = SF_DRW(params, del_t)**2 + (err[i]**2 + err[j]**2)

	result = - np.log(2 * np.pi * V_eff2) / 2.0 - del_mag**2 / (2 * V_eff2)
	# print "Lij: ", result
	return result




def prior(params, MJD, model='DRW'):
	if model == 'DRW':
		b, sigma, tau = params
		intv = MJD[1:] - MJD[0:-1]
		med_intv = np.median(intv)
		return np.sum(np.exp(-med_intv / tau))
		# return 1.0
	elif model == 'pow-law':
		A = np.exp(params[0])
		gamma = params[1]
		return 1 / (A * (1 + gamma**2))





def lnlikelihood(params, MJD, mag, err, model="DRW", mode='KBS09'):
	if mode == 'KBS09':
		if model == 'DRW':

			intv = MJD[1:] - MJD[0:-1]
			med_intv = np.median(intv)

			b, sigma, tau = params

			if 10 < b < 30 and 0 < tau * sigma**2 / 2 < 5 and 0 < tau < 4000 and sigma > 0: 
				# x* -> x_s
				# x^ -> x_h
				Omega = np.zeros(len(mag))
				x_s = np.zeros(len(mag))
				x_h =  np.zeros(len(mag))
				a = np.zeros(len(mag))
				a[1:] = np.exp(-(intv) / tau)
				x_s[:] = mag[:] - b
				x_h[0] = 0
				Omega[0] = sigma**2 * tau / 2


				for i in range(1, len(mag)):
					Omega[i] = Omega[0] * (1 - a[i]**2) + a[i]**2 * Omega[i - 1] * (1 - Omega[i - 1] / (Omega[i - 1] + err[i - 1]**2))
					x_h[i] = a[i] * x_h[i - 1] + a[i] * Omega[i - 1] * (x_s[i - 1] - x_h[i - 1]) / (Omega[i - 1] + err[i - 1]**2)

				result = -0.5 * np.sum(np.log(Omega + err**2)) - 0.5 * np.sum((x_s - x_h)**2 / (Omega + err**2)) - len(mag) / 2.0 * np.log(2 * np.pi) 
				
				return result
			else:
				return -np.inf
	elif mode == 'S10':
		lnlikelihood = 0.0
		for i in range(len(mag)):
			for j in range(i + 1, len(mag)):
				lnlikelihood += lnLij(MJD=MJD, mag=mag, err=err, params=params, i=i, j=j, model=model)
		# print lnlikelihood
		return lnlikelihood






def lnprob(params, MJD, mag, err, model="DRW", mode='KBS09'):
	if mode == 'KBS09':
		lnp = lnlikelihood(params=params, MJD=MJD, mag=mag, err=err, model=model, mode=mode) + np.log(prior(params, MJD, model=model))
	elif mode == 'S10':
		lnp = lnlikelihood(params=params, MJD=MJD, mag=mag, err=err, model=model, mode=mode) + np.log(prior(params, MJD, model=model))
	if not np.isfinite(lnp):
		return -np.inf
	return lnp




def minimizee(params, MJD, mag, err, model="DRW"):
	return -((lnlikelihood(params, MJD, mag, err, model, mode='S10')))

def SF_fit_params(MJD, mag, err, p0=[0.5, 0.5], model="DRW", MCMC_step=1000, MCMC_threads=8, mode='KBS09'):
	# print "mag: ", mag
	p_dict = {}
	if model == "DRW":
		# p_best = minimize(minimizee, p0, args=(MJD, mag, err, model), bounds=((-10,2), (0,4000)), method='TNC')
		# p_dict['sigma'] = abs(p_best.x[0])
		# p_dict['tau'] = p_best.x[1]
	# 	# print "start minimizing."
	# 	# p_best = minimize(minimizee, p0, args=(MJD, mag, err, model), bounds=((10,30), (0,2), (0,4000)), method='SLSQP')
		ndim, nwalkers = 3, 30

		sampler = EnsembleSampler(nwalkers, ndim, lnprob, args=(MJD, mag, err, model, mode), threads=MCMC_threads)
		p0 = [np.array([20, 0.08, 30]) + 1e-3 * np.random.randn(ndim) for i in range(nwalkers)]
		pos, prob, state = sampler.run_mcmc(p0, 200)
		sampler.reset()

		sampler.run_mcmc(pos, MCMC_step)
		samples = sampler.chain[:, MCMC_step / 2:, :].reshape((-1, ndim))
		print "finished an MCMC"
		p_best = [0] * 3
		p_best[:] = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
		p_dict = {}
		p_dict['mean_mag'] = p_best[0]
		p_dict['sigma_KBS09'] = p_best[1]
		p_dict['tau'] = p_best[2]

		# calculate the real physical sigma. c.f. Macleod et al. (2010)
		p_dict['sigma_M10'] = [0, 0, 0]
		p_dict['sigma_M10'][0] = p_best[1][0] * (p_best[2][0] / 2)**0.5
		p_dict['sigma_M10'][1] = (p_best[0][1]**2 * p_best[2][0] / 2 + p_best[0][0]**2 * p_best[2][1]**2 / (8.0 * p_best[2][0]))**0.5
		p_dict['sigma_M10'][2] = (p_best[0][2]**2 * p_best[2][0] / 2 + p_best[0][0]**2 * p_best[2][2]**2 / (8.0 * p_best[2][0]))**0.5
	
	elif model == "pow-law":
		# print "PL"
		# p_best = minimize(minimizee, p0, args=(MJD, mag, err, model), bounds=((-10,1), (0,1)), method='TNC')
		# p_dict['A'] = abs(p_best.x[0])
		# p_dict['gamma'] = p_best.x[1]

		ndim, nwalkers = 2, 30

		sampler = EnsembleSampler(nwalkers, ndim, lnprob, args=(MJD, mag, err, model, mode), threads=MCMC_threads)
		p0 = [np.array([np.log(0.1), 0.1]) + 0.05 * np.random.randn(ndim) for i in range(nwalkers)]
		pos, prob, state = sampler.run_mcmc(p0, 200)
		sampler.reset()

		sampler.run_mcmc(pos, MCMC_step)
		samples = sampler.chain[:, MCMC_step / 2:, :].reshape((-1, ndim))

		fig = corner.corner(samples, labels=["$lnA$", r"$\gamma$"])
		fig.savefig("./test.png")



		p_best = [0] * 2
		p_best[:] = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
		p_dict = {}
		p_dict['A'] = np.exp(p_best[0][0])
		p_dict['gamma'] = p_best[1][0]

	# print p_best.x
	# print p_best.success
	return p_dict


def SF_true(t, y, yerr, zp1, n_dt = 10):

	t = t / 365.25
	del_t_set = []
	del_y_set = []

	for i in range(len(t)):
		for j in range(i + 1, len(t)):
			del_t = t[j] - t[i]
			del_y = (np.pi / 2)**0.5 * abs(y[j] - y[i]) - (yerr[i]**2 + yerr[j]**2)**0.5
			del_t_set.append(del_t)
			del_y_set.append(del_y)

	del_t_set = np.array(np.log10(del_t_set))
	del_y_set = np.array(del_y_set)

	t_intv_set = np.linspace(np.min(del_t_set), np.max(del_t_set), n_dt + 2)
	print t_intv_set
	sf = []

	for i in range(0, n_dt + 2):
		if i == 0:
			ind  = np.argwhere(del_t_set < t_intv_set[1])
			print ind
		elif i == n_dt + 1:
			ind = np.argwhere(del_t_set > t_intv_set[-2])
		else:
			ind1 = np.argwhere(del_t_set > t_intv_set[i - 1])
			ind2 = np.argwhere(del_t_set < t_intv_set[i + 1])
			ind = np.intersect1d(ind1, ind2)
		cur_del_t_set = del_t_set[ind].reshape((1, len(ind)))
		cur_del_y_set = del_y_set[ind].reshape((1, len(ind)))
		sf.append(np.mean(cur_del_y_set))

	return t_intv_set[0:n_dt+2], np.array(sf)


if __name__ == "__main__":
	drw_path = '/Users/zhanghaowen/Desktop/AGN/BroadBand_RM/data/s82drw_g.dat'
	drw_data = pd.read_csv(drw_path, header=None, skiprows=3, \
			   names=['SDR5ID', 'lgTau', 'lgSigma', 'Plike', 'Pnoise', 'Pinf', 'npts'],\
			   usecols=[0, 7, 8, 14, 15, 16, 18], delim_whitespace=True,\
			   dtype={'SDR5ID': str, 'lgTau': np.float64, 'lgSigma': np.float64, \
			   'Plike': np.float64, 'Pnoise': np.float64, 'Pinf': np.float64, 'npts': int})
	dbID_path = '/users/zhanghaowen/Desktop/AGN/BroadBand_RM/data/DB_QSO_S82.dat'
	dbID_data = pd.read_csv(dbID_path, header=None, names=['dbID', 'SDR5ID'],\
							usecols=[0, 3], delim_whitespace=True, dtype=str)

	paper_sigma = [] 
	paper_tau = []
	fit_sigma = []
	fit_tau = []
	test_set = np.random.choice(9258, size=300)
	j = 0
	for i in range(1):

		if drw_data['lgSigma'][i] <= -10 or drw_data['npts'][i] < 10 or\
		   drw_data['Plike'][i] - drw_data['Pnoise'][i] <= 2 or\
		   drw_data['Plike'][i] - drw_data['Pinf'][i] <= 0.05:

		   print "skip."
		   continue
		# print np.where(dbID_data['SDR5ID'] == drw_data['SDR5ID'][i])
		dbID = np.array(dbID_data['dbID'])[np.where(dbID_data['SDR5ID'] == '71597')][0]
		lc_path = '/Users/zhanghaowen/Desktop/AGN/BroadBand_RM/QSO_S82/%s' %dbID
		lc_data = pd.read_csv(lc_path, header=None, names=['MJD', 'mag', 'err'],\
							  usecols=[3, 4, 5], delim_whitespace=True, dtype=np.float64)

		MJD = np.array(lc_data['MJD'])
		mag = np.array(lc_data['mag'])
		err = np.array(lc_data['err'])
		mask = mag == -99.99
		MJD = np.ma.array(MJD, mask=mask).compressed()
		mag = np.ma.array(mag, mask=mask).compressed()
		err = np.ma.array(err, mask=mask).compressed()

		try:
			params = SF_fit_params(MJD, mag, err, model='pow-law', mode='S10')
			print "A: ", params['A']
			print "gamma: ", params['gamma']
		except:
			continue
		j += 1
		# sigma = params['sigma_KBS09'][0]
		# tau = params['tau'][0]



		# sigma = sigma * (2.0 / tau)**0.5

	# 	paper_sigma.append((np.array(drw_data['lgSigma'][i])) - np.log10(365.25**0.5))
	# 	paper_tau.append((np.array(drw_data['lgTau'][i])))
	# 	fit_sigma.append(np.log10(sigma))
	# 	fit_tau.append(np.log10(tau))
	# 	print np.log10(sigma), np.log10(tau)
	# 	print "finished %d fitting." %j

	# paper_sigma = np.array(paper_sigma)
	# paper_tau = np.array(paper_tau)
	# fit_sigma = np.array(fit_sigma)
	# fit_tau = np.array(fit_tau)

	# DRW_dict = {}

	# plt.scatter(paper_sigma, fit_sigma, color='blue')
	# plt.plot(paper_sigma, paper_sigma, color='black')
	# plt.xlabel('paper sigma')
	# plt.ylabel('fitted sigma')
	# plt.show()
	# # plt.savefig('./sigma_paper_vs_code.png')
	# plt.close()

	# plt.scatter(paper_tau, fit_tau, color='red')
	# plt.plot(paper_tau, paper_tau, color='black')
	# plt.xlabel('paper tau')
	# plt.ylabel('fitted tau')
	# plt.show()
	# # plt.savefig('./sigma_paper_vs_code.png')
	# plt.close()

	# DRW_dict['paper_sigma'] = paper_sigma
	# DRW_dict['paper_tau'] = paper_tau
	# DRW_dict['fit_sigma'] = fit_sigma
	# DRW_dict['fit_tau'] = fit_tau

	# sio.savemat('./DRW_params_2.mat', DRW_dict)











from functions_and_class import *
import numpy as np
from scipy.optimize import minimize



def ESDfactors_est(Y, W, Xy, Z, Xz, Ry, Rz, x0=None, itr_times=9, corr_times=1, message=True, options=None):
    """
    Python implementation of ESDfactors_est MATLAB function.

    Parameters:
        Y: (n, T+1) NumPy array - matrix of outcomes.
        W: (n, n, T+1) NumPy array - spatial weight matrix.
        Xy: (n, T+1, ky) NumPy array - independent variables in main equation.
        Z: (n, p, T+1) NumPy array - dependent variables in auxiliary equation.
        Xz: (n, T+1, max_kzp, p) NumPy array - independent variables in auxiliary equation.
        Ry: number of factors in y
        Rz: number of factors in z
        x0: Initial guess for optimization.
        itr_times: default value is 9. If x0 is [], then the algorithm will generate "itr_times" number of start points
                    to avoid local optimal solution
        corr_times: default value is 1. Times for bias correction. After each bias correction, the estimator could be
                    used to calculate more precise asymptotic variance, and based  on this asymptotic variance, the
                    route can update the bias-corrected estimator.
        message: Boolean to enable/disable message output.
        options: Dictionary for optimization options.

    Returns:
        est: estimation results.
    """

    if options is None:
        options = {
            "tol_x": 1e-9,
            "max_fun_evals": 100000,
            "max_iter": 100000,
            "tol_fun": 1e-9,
            "display": "notify",
            "use_parallel": False
        }

    # Initialize the Estimation object
    est = Estimation()

    # Get rid of the first observations.
    Y0 = Y[:, 0].reshape(-1, 1)  # 将 Y[:, 0] 调整为列向量 (n, 1)
    W0, Z0 = W[:, :, 0], Z[:, :, 0]
    Y, W, Xy, Z, Xz = Y[:, 1:], W[:, :, 1:], Xy[:, 1:, :], Z[:, :, 1:], Xz[:, 1:, :, :]



    # Get model parameters
    n, T, ky = Xy.shape
    _, _, max_kzp, p = Xz.shape
    J = int(p * (p + 1) / 2)

    # Compute kz for each p
    kzp = np.array([
            max_kzp if not np.isnan(Xz[:, :, :, j]).any() else
            int( np.nanmax(np.where( np.isnan(Xz[:, :, :, j]) )[2]) )
            for j in range(p)
          ])
    kz = int(np.sum(kzp))
    K = [ky, 3, p, kz, p**2, 1, J]


    # Update estimation parameters
    est.param.Ry = Ry
    est.param.Rz = Rz
    est.param.n = n
    est.param.T = T
    est.param.ky = ky
    est.param.max_kzp = max_kzp
    est.param.p = p
    est.param.J = J
    est.param.kzp = kzp
    est.param.kz = kz
    est.param.K = K


    # Objective function for optimization
    def objective_function(x):
        return obj_Q(x, Y, W, Xy, Z, Xz, Y0, Z0, W0, Ry, Rz)


    # Perform optimization
    if x0 is None:
        fval = np.inf * np.ones(itr_times)
        x = np.zeros((sum(K), itr_times))
        exitflags = np.zeros(itr_times)

        if message:
            print("Estimating...  0%", end="")

        for itr in range(itr_times):

            percent_done = 100 * (itr+1) / itr_times

            if message:
                print(f"\b\b\b\b{percent_done:3.0f}%", end="")

            x0 = np.random.randn(sum(K))
            res = minimize(objective_function, x0, method="L-BFGS-B",
                           options={"maxiter": options["max_iter"],
                                    'ftol': 1e-9,
                                    'eps': 1e-9})
            fval[itr], x[:, itr], exitflags[itr] = res.fun, res.x, res.status

        if message:
            print(f"\b\b\b\b{100:3.0f}%", end="")
            print()

        est.nb.fval = np.min(fval)
        final_x = x[:, np.argmin(fval)]
        est.exitflag = exitflags[np.argmin(fval)]

    else:

        if message:
            print("Estimating...  0%", end="")

        x0 = x0.flatten()
        res = minimize(objective_function, x0, method="L-BFGS-B",
                       options={"maxiter": options["max_iter"],
                                'ftol': 1e-9,
                                'eps': 1e-9})
        final_x, est.nb.fval, est.exitflag = res.x, res.fun, res.status

        if message:
            print(f"\b\b\b\b{100:3.0f}%", end="")
            print()



    # Adjust estimators
    final_x[ky : ky + 3] = np.tanh(final_x[ky : ky + 3])
    final_x[ky + 3 + p + kz + p ** 2] = np.exp(final_x[ky + 3 + p + kz + p ** 2])

    # here we make sure the diagonal elements of Sigma_epsilon_neg_half_temp is positive
    temp_alpha = final_x[ky + 3 + p + kz + p ** 2 + 1 : ky + 3 + p + kz + p ** 2 + 1 + J]
    Sigma_epsilon_neg_half_temp = np.zeros((p, p))
    tril_indices = np.tril_indices(p)
    Sigma_epsilon_neg_half_temp[tril_indices] = temp_alpha
    Sigma_epsilon_neg_half_temp += np.tril(Sigma_epsilon_neg_half_temp, -1).T
    Sigma_epsilon_temp = np.linalg.inv(Sigma_epsilon_neg_half_temp) @ np.linalg.inv(Sigma_epsilon_neg_half_temp).T
    Sigma_epsilon_neg_half_temp = sqrtm(np.linalg.inv(Sigma_epsilon_temp))
    final_x[ky + 3 + p + kz + p ** 2 + 1 : ky + 3 + p + kz + p ** 2 + 1 + J] = Sigma_epsilon_neg_half_temp[tril_indices]


    #     final_x = theta_0.flatten()

    # original higher moments of epsilon and xi, variance and bias corrector
    if message:
        print("Calculating original asymptotic variance...  0%", end="")

    est.nb.theta = final_x
    est.nb.theta_to_params(est.param)


    # estimate factors and factor loadings and define the projection matrix
    Hy, Hz = H(est.nb.theta, Y, W, Xy, Z, Xz, Y0, Z0, W0, est.param)
    est.nb.factors_and_proj_matrix(Hz, Hy, est.param)

    # original higher moments of epsilon and xi, variance and bias corrector
    est.nb.moments_and_residuals(est.param, Y, W, Xy, Z, Xz, Y0, Z0, W0)



    est.nb.asymp_var(est.param, Y, W, Xy, Z, Xz, Y0, Z0, W0, message)
    est.nb.bcorr = mldivide(est.nb.Sigma, est.nb.phi) / np.sqrt(n*T)



    # bias corrected estimators
    if message:
        print(f"Calculating bias corrected asymptotic variance ({1}/{corr_times})...  0%", end="")

    est.bc.theta = est.nb.theta - est.nb.bcorr.flatten()
    est.bc.theta_to_params(est.param)
    Hy, Hz = H(est.bc.theta, Y, W, Xy, Z, Xz, Y0, Z0, W0, est.param)
    est.bc.factors_and_proj_matrix(Hz, Hy, est.param)
    est.bc.moments_and_residuals(est.param, Y, W, Xy, Z, Xz, Y0, Z0, W0)
    est.bc.asymp_var(est.param, Y, W, Xy, Z, Xz, Y0, Z0, W0, message)
    est.bc.bcorr = mldivide(est.bc.Sigma, est.bc.phi) / np.sqrt(n*T)
    est.bc.theta = est.nb.theta - est.bc.bcorr.flatten()
    est.bc.theta_to_params(est.param)


    # we have already corrected theta above (for one time), so i start at 1 instead of 0
    for i in range(1, corr_times-1):
        if message:
            print(f"Calculating bias corrected asymptotic variance ({i+1}/{corr_times})...  0%", end="")

        Hy, Hz = H(est.bc.theta, Y, W, Xy, Z, Xz, Y0, Z0, W0, est.param)
        est.bc.factors_and_proj_matrix(Hz, Hy, est.param)
        est.bc.moments_and_residuals(est.param, Y, W, Xy, Z, Xz, Y0, Z0, W0)
        est.bc.asymp_var(est.param, Y, W, Xy, Z, Xz, Y0, Z0, W0, message)
        est.bc.bcorr = mldivide(est.bc.Sigma, est.bc.phi) / np.sqrt(n * T)
        est.bc.theta = est.nb.theta - est.bc.bcorr.flatten()
        est.bc.theta_to_params(est.param)

    est.bc.fval = objective_function(est.bc.theta)

    return est
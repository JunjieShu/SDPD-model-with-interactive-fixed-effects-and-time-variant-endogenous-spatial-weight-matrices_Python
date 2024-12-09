import numpy as np
from scipy.linalg import block_diag, det, sqrtm
from scipy.sparse import tril, eye



def com_mat(m, n):
    # determine permutation applied by K
    w = np.arange(m * n).reshape((m, n), order="F").T.ravel(order="F")

    # apply this permutation to the rows (i.e. to each column) of identity matrix and return result
    return np.eye(m * n)[w, :]

def mldivide(A, B):
    return np.linalg.solve(A, B)


def mrdivide(A, B):
    return np.linalg.solve(B.T, A.T).T



# Define the Param class
class Param:
    def __init__(self):
        self.wt_Ry = None
        self.Ry = None
        self.Rz = None
        self.n = None
        self.T = None
        self.ky = None
        self.max_kzp = None
        self.p = None
        self.J = None
        self.kzp = None
        self.kz = None
        self.K = None

    def __repr__(self):
        return str(self.__dict__)

class Error:
    def __init__(self):
        self.Ec = None
        self.xi = None
        self.vec_xi_L = None
        self.vec_epsilon_Lp = None

class Moments:
    def __init__(self):
        self.mu_3_vec_xi_L = None
        self.mu_4_vec_xi_L = None
        self.mu_3_vec_epsilon_Lp = None
        self.mu_4_vec_epsilon_Lp = None

# Define the Estimator (for nb and bc) class
class Estimator:
    def __init__(self):
        self.theta = None
        self.bcorr = None
        self.beta_y = None
        self.gamma = None
        self.rho = None
        self.lambda_ = None
        self.delta = None
        self.beta_z = None
        self.vect_Upsilon = None
        self.Upsilon = None
        self.alpha_xi = None
        self.sigma_xi = None
        self.alpha = None
        self.Sigma_epsilon_neg_half = None
        self.Sigma_epsilon = None
        self.sigma_v_epsilon = None
        self.sigma_v = None
        self.Sigma_v_epsilon = None
        self.Fz = None
        self.Gz = None
        self.Fy = None
        self.Gy = None
        self.wt_Gz = None
        self.wt_Gy_wt_Fy = None
        self.error = None
        self.moments = None
        self.Sigma = None
        self.phi = None
        self.Vtheta = None
        self.fval = None
        self.error = Error()
        self.moments = Moments()


    def theta_to_params(self, param_class):
        # self = est.nb; param_class = est.param
        ky = param_class.ky
        p = param_class.p
        kz = param_class.kz
        J = int(p * (p + 1) / 2)

        # Extract parameters from theta
        self.beta_y = self.theta[:ky]
        self.gamma = self.theta[ky]
        self.rho = self.theta[ky + 1]
        self.lambda_ = self.theta[ky + 2]
        self.delta = self.theta[ky + 3 : ky + 3 + p]
        self.beta_z = self.theta[ky + 3 + p : ky + 3 + p + kz]
        self.vect_Upsilon = self.theta[ky + 3 + p + kz : ky + 3 + p + kz + p ** 2]
        self.Upsilon = self.vect_Upsilon.reshape(p, p, order='F')
        self.alpha_xi = self.theta[ky + 3 + p + kz + p ** 2]
        self.sigma_xi = 1 / self.alpha_xi
        self.alpha = self.theta[ky + 3 + p + kz + p ** 2 + 1 : ky + 3 + p + kz + p ** 2 + 1 + J]

        # Compute Sigma_epsilon_neg_half and Sigma_epsilon
        self.Sigma_epsilon_neg_half = np.zeros((p, p))
        tril_indices = np.tril_indices(p)
        self.Sigma_epsilon_neg_half[tril_indices] = self.alpha
        self.Sigma_epsilon_neg_half += np.tril(self.Sigma_epsilon_neg_half, -1).T
        self.Sigma_epsilon = np.linalg.inv(self.Sigma_epsilon_neg_half) @ np.linalg.inv(self.Sigma_epsilon_neg_half).T
        self.Sigma_epsilon_neg_half = sqrtm(np.linalg.inv(self.Sigma_epsilon))


        # Compute sigma_v_epsilon, sigma_v, and Sigma_v_epsilon
        self.sigma_v_epsilon = self.Sigma_epsilon @ self.delta
        self.sigma_v = np.sqrt(self.sigma_xi ** 2 + self.sigma_v_epsilon.T @ self.delta)
        self.Sigma_v_epsilon = np.block([
            [np.array([[self.sigma_v]]), self.sigma_v_epsilon.reshape(-1, 1).T],
            [self.sigma_v_epsilon.reshape(-1, 1), self.Sigma_epsilon]
        ])

    def factors_and_proj_matrix(self, Hz, Hy, param_class):
        # self = est.nb; param_class = est.param
        Rz = param_class.Rz
        Ry = param_class.Ry
        n, T = Hy.shape
        p = Hz.shape[0] // n

        ### Factors for Z
        # eigvals, eigvecs = eig(Hz.T @ Hz)
        # sorted_indices = np.argsort(eigvals)[::-1]
        # self.Fz = eigvecs[:, sorted_indices[:Rz]]
        #
        # eigvals, eigvecs = eig(Hz @ Hz.T)
        # sorted_indices = np.argsort(eigvals)[::-1]
        # self.Gz = eigvecs[:, sorted_indices[:Rz]]

        U, S, Vt = np.linalg.svd(Hz, full_matrices=False)
        self.Fz = Vt.T[:, :Rz]  # Fz 来自 V 的前 Rz 列
        self.Gz = U[:, :Rz]  # Gz 来自 U 的前 Rz 列

        ### Factors for Y
        # eigvals, eigvecs = eig(Hy.T @ Hy)
        # sorted_indices = np.argsort(eigvals)[::-1]
        # self.Fy = eigvecs[:, sorted_indices[:Ry]]
        #
        # eigvals, eigvecs = eig(Hy @ Hy.T)
        # sorted_indices = np.argsort(eigvals)[::-1]
        # self.Gy = eigvecs[:, sorted_indices[:Ry]]

        U, S, Vt = np.linalg.svd(Hy, full_matrices=False)
        self.Fy = Vt.T[:, :Ry]  # Fy 来自 V 的前 Ry 列
        self.Gy = U[:, :Ry]  # Gy 来自 U 的前 Ry 列


        self.wt_Gz = mldivide(np.kron(self.Sigma_epsilon_neg_half, np.eye(n)), self.Gz)
        self.wt_Gy_wt_Fy = (self.sigma_xi * self.Gy @ self.Fy.T
                            + np.kron(self.delta.T, np.eye(n)) @ self.wt_Gz @ self.Fz.T)

        # Projection matrices
        self.P_Fz = self.Fz @ self.Fz.T
        self.M_Fz = np.eye(T) - self.P_Fz
        self.P_Gz = self.Gz @ self.Gz.T
        self.M_Gz = np.eye(n * p) - self.P_Gz

        self.P_Fy = self.Fy @ self.Fy.T
        self.M_Fy = np.eye(T) - self.P_Fy
        self.P_Gy = self.Gy @ self.Gy.T
        self.M_Gy = np.eye(n) - self.P_Gy


    def moments_and_residuals(self, param_class, Y, W, Xy, Z, Xz, Y0, Z0, W0):
        # self = est.nb; param_class = est.param
        p = param_class.p
        n = param_class.n
        T = param_class.T
        ky = param_class.ky
        kzp = param_class.kzp

        In = np.eye(n)  # Identity matrix for n
        Ip = np.eye(p)  # Identity matrix for p

        self.error.Ec = np.full((n * p, T), np.nan)
        self.error.xi = np.full((n * T), np.nan)

        Last_Z = Z0
        last_Y = Y0
        Last_WY = W0 @ Y0

        for t in range(T):
            # Build Xntz using block diagonal structure
            Xntz = []
            for j in range(p):
                if j == 0:
                    Xntz = Xz[:, t, :kzp[j], j].reshape(n, kzp[j])
                else:
                    Xntz = block_diag(Xntz, Xz[:, t, :kzp[j], j].reshape(n, kzp[j]))

            # Compute epsilon_t
            epsilon_t = Z[:, :, t].reshape(n*p,1,order="F") - \
                    np.kron(Ip, Last_Z) @ self.vect_Upsilon.reshape(-1,1,order="F") - \
                    Xntz @ self.beta_z.reshape(-1,1) - \
                    self.wt_Gz @ self.Fz[t, :].reshape(-1,1,order="F")

            self.error.Ec[:, t] = epsilon_t.flatten()

            epsilon_np = epsilon_t.reshape(n, p, order='F')

            # Compute residual v
            v = (
                    Y[:, t].reshape(-1,1)
                    - self.lambda_ * W[:, :, t] @ Y[:, t].reshape(-1,1)
                    - self.gamma * last_Y.reshape(-1,1)
                    - self.rho * Last_WY.reshape(-1,1)
                    - Xy[:, t, :].reshape(n, ky) @ self.beta_y.reshape(-1,1)
                    - self.wt_Gy_wt_Fy[:, t].reshape(-1,1)
            )

            self.error.xi[t * n : (t+1) * n] = (v - epsilon_np @ self.delta.reshape(-1,1)).flatten()

            # Update Last_Z, last_Y, and Last_WY
            Last_Z = Z[:, :, t]
            last_Y = Y[:, t]
            Last_WY = W[:, :, t] @ Y[:, t]

        # Compute additional error and moments
        self.error.vec_xi_L = self.alpha_xi * self.error.xi
        self.moments.mu_3_vec_xi_L = np.mean(self.error.vec_xi_L ** 3)
        self.moments.mu_4_vec_xi_L = np.mean(self.error.vec_xi_L ** 4)

        self.error.vec_epsilon_Lp = (np.kron(self.Sigma_epsilon_neg_half, In) @ self.error.Ec).reshape(-1,1)
        self.moments.mu_3_vec_epsilon_Lp = np.mean(self.error.vec_epsilon_Lp ** 3)
        self.moments.mu_4_vec_epsilon_Lp = np.mean(self.error.vec_epsilon_Lp ** 4)

    #  asymptotic variance
    def asymp_var(self, param_class, Y, W, Xy, Z, Xz, Y0, Z0, W0, message):
        # self = est.nb; param_class = est.param
        # Parameters
        K = param_class.K
        n = param_class.n
        T = param_class.T
        ky = param_class.ky
        p = param_class.p
        J = param_class.J
        L = n * T
        kzp = param_class.kzp
        kz = param_class.kz

        # some matrices will be needed
        In = np.eye(n)
        Ip = np.eye(p)
        I_T = np.eye(T)
        K_pL = com_mat(p, L)
        K_np = com_mat(n, p)
        M_Fy_otimes_M_Gy = np.kron(self.M_Fy, self.M_Gy)

        S = np.stack([In - self.lambda_ * W[:, :, t] for t in range(T)],  axis=2)
        G = np.stack([mrdivide(W[:,:,t], S[:,:,t]) for t in range(T)],  axis=2)

        del S



        # Reshape equations for Y
        Y_last_m = np.hstack([Y0, Y[ :, :-1]])
        WY_last_m = np.hstack( [W0 @ Y0, np.hstack([W[:, :, t] @ Y[:, t][:, np.newaxis] for t in range(T - 1)])] )
        WY_m = np.hstack([W[:, :, t] @ Y[:, t][:, np.newaxis] for t in range(T)])

        # Block diagonal matrices
        wt_G = block_diag(*[G[:, :, t] for t in range(T)])
        W1L = block_diag(*[W[:, :, t] for t in range(T)])

        W2L = np.block([  [np.zeros((n, n * (T - 1))), np.zeros((n, n))],
                          [np.eye(n * (T - 1)), np.zeros((n * (T - 1), n))]  ])

        del G

        W3L = W2L @ W1L

        WL = self.lambda_ * W1L + self.gamma * W2L + self.rho * W3L
        SL = np.eye(n * T) - WL

        del WL

        G1L = mrdivide(W1L, SL)  # W1L / SL
        G2L = mrdivide(W2L, SL)  # W2L / SL
        G3L = mrdivide(W3L, SL)  # W3L / SL

        del W1L, W3L, SL

        # Reshape equations for Z
        S2L_U = np.eye(L * p) - np.kron(self.Upsilon.T, W2L)
        G2L_U = mrdivide(np.kron(Ip, W2L), S2L_U  )

        del W2L, S2L_U

        # Bias and other matrices
        phi = np.zeros(sum(K))
        a_Lp = np.zeros((L * p, sum(K)))
        b_L = np.zeros((L, sum(K)))
        A_Lp = [0] * sum(K)
        B_L = [0] * sum(K)
        D_L_Lp = [0] * sum(K)

        # beta_y
        for k in range(ky):
            Z_y_k = Xy[:, :, k]
            b_L[:, k] = (self.alpha_xi * self.M_Gy @ Z_y_k @ self.M_Fy).flatten(order='F')
        phi[:ky] = 0

        del Z_y_k

        # gamma
        b_L[:, ky] = (self.alpha_xi * self.M_Gy @ Y_last_m @ self.M_Fy).flatten(order='F')
        phi[ky] = np.trace(M_Fy_otimes_M_Gy @ G2L) / np.sqrt(n * T)
        del G2L

        # rho
        b_L[:, ky + 1] = (self.alpha_xi * self.M_Gy @ WY_last_m @ self.M_Fy).flatten(order='F')
        phi[ky + 1] = np.trace(M_Fy_otimes_M_Gy @ G3L) / np.sqrt(n * T)
        del G3L

        # lambda
        b_L[:, ky + 2] = (self.alpha_xi * self.M_Gy @ WY_m @ self.M_Fy).flatten(order='F')
        phi[ky + 2] = (np.trace(M_Fy_otimes_M_Gy @ G1L) - np.trace(wt_G)) / np.sqrt(n * T)
        del G1L, wt_G

        # delta
        for k in range(p):
            ek = np.zeros(p)
            ek[k] = 1
            b_L[:, ky + 3 + k] = (self.alpha_xi * self.M_Gy @ np.kron(ek.T, In) @ self.wt_Gz @ self.Fz.T @ self.M_Fy).flatten(order='F')
            D_L_Lp[ky + 3 + k] = np.kron( self.alpha_xi * self.M_Fy, self.M_Gy @ np.kron(ek.T @ np.linalg.inv(self.Sigma_epsilon_neg_half), In))

        phi[ky + 3 : ky + 3 + p] = 0
        del ek


        # beta_z
        X_z_np_kz = np.zeros((n * p, kz, T))
        for t in range(T):
            Xntz = []
            for j in range(p):
                if j == 0:
                    Xntz = Xz[:, t, :kzp[j], j].reshape(n, kzp[j], order='F')
                else:
                    Xntz = block_diag(Xntz, Xz[:, t, :kzp[j], j].reshape(n, kzp[j], order='F'))

            X_z_np_kz[:, :, t] = Xntz

        for k in range(kz):
            Z_z_k = X_z_np_kz[:, k, :]
            a_Lp[:, ky + 3 + p + k] = (self.M_Gz @ np.kron(self.Sigma_epsilon_neg_half, In) @ Z_z_k @ self.M_Fz).flatten(order='F')
            b_L[:, ky + 3 + p + k] = - self.alpha_xi * (self.M_Gy @ np.kron(self.delta.T, In) @ Z_z_k @ self.M_Fy).flatten(order='F')

        phi[ky + 3 + p: ky + 3 + p + kz] = 0
        del X_z_np_kz, Z_z_k


        # vect_Upsilon
        Q = np.zeros((n, L, T))
        for t in range(T):
            Q[:, t * n : (t+1) * n, t] = In

        for k in range(p ** 2):
            Ek = np.zeros((p, p))
            Ek.T.flat[k] = 1

            temp_E_Q = []
            for t in range(T):
                temp_E_Q.append(np.kron(Ek.T, Q[:, :, t]))
            temp_E_Q = np.vstack(temp_E_Q)

            wt_A = np.kron(self.M_Fz, self.M_Gz @ np.kron(self.Sigma_epsilon_neg_half, In)) @ \
                    temp_E_Q @ G2L_U @ K_pL @ \
                    np.kron(I_T, K_np @ np.kron( np.linalg.inv(self.Sigma_epsilon_neg_half), In))

            phi[ky + 3 + p + kz + k] = np.trace(wt_A) / np.sqrt(n * T)

            Z_z_k = np.full((n * p, T), np.nan)
            for t in range(T):
                if t == 0:
                    tempZk = np.kron(Ip, Z0)  # For the first time step
                else:
                    tempZk = np.kron(Ip, Z[:, :, t - 1])  # For subsequent time steps

                Z_z_k[:, t] = tempZk[:, k]

            a_Lp[:, ky + 3 + p + kz + k] = (self.M_Gz @ np.kron(self.Sigma_epsilon_neg_half, In) @ Z_z_k @ self.M_Fz).flatten(order='F')
            b_L[:, ky + 3 + p + kz + k] = - self.alpha_xi * ( self.M_Gy @ np.kron(self.delta.T, In) @ Z_z_k @ self.M_Fy).flatten(order='F')

        del Q, Ek, G2L_U, temp_E_Q, wt_A

        # alpha_xi
        B_L[ky + 3 + p + kz + p ** 2] = - self.sigma_xi * M_Fy_otimes_M_Gy
        phi[ky + 3 + p + kz + p ** 2] = (np.sqrt(n / T) + np.sqrt(T / n)) * param_class.Ry * self.sigma_xi
        del M_Fy_otimes_M_Gy

        #  alpha
        for k in range(J):

            ek = np.zeros((J,1))
            ek[k] = 1

            # Construct Ek matrix
            Ek = np.zeros((p, p))
            Ek[np.tril_indices(p)] = ek.flatten(order='F')
            Ek = Ek + np.tril(Ek, -1).T

            # Compute temp_S_E
            temp_S_E = mldivide(self.Sigma_epsilon_neg_half, Ek)

            # Compute wt_A
            wt_A = - np.kron(self.M_Fz, np.kron(temp_S_E, In) @ self.M_Gz)

            # Update A_Lp and phi
            A_Lp[ky + 3 + p + kz + p**2 + 1 + k] = (wt_A.T + wt_A) / 2
            phi[ky + 3 + p + kz + p**2 + 1 + k] = (
                np.sqrt(n / T) * param_class.Rz * np.trace(temp_S_E)
                + np.sqrt(T / n) * np.trace(np.kron(temp_S_E, In) @ self.P_Gz)
            )

        del wt_A, temp_S_E, Ek


        # Sigma
        self.Sigma = np.zeros((np.sum(K), np.sum(K)))
        for k1 in range(np.sum(K)):

            percent_done = 100 * ((k1 + 1) * sum(K) - (k1 * (k1 + 1)) / 2) / (sum(K) * (sum(K) + 1) / 2)

            if message:
                print(f"\b\b\b\b{percent_done:3.0f}%", end="")

            self.Sigma[k1, k1:] = (
                np.array([np.sum(D_L_Lp[k1] * c) for c in D_L_Lp[k1:]])
                + 2 * np.array([np.sum(A_Lp[k1] * c) for c in A_Lp[k1:]])
                + 2 * np.array([np.sum(B_L[k1] * c) for c in B_L[k1:]])
        )

        self.Sigma = (self.Sigma + np.triu(self.Sigma, k=1).T + a_Lp.T @ a_Lp + b_L.T @ b_L) / (n*T)

        if message:
            print(f"\b\b\b\b{100:3.0f}%", end="")
            print()  # 换行，表示进度结束

        self.phi = phi.reshape(-1,1)
        self.Vtheta = np.linalg.inv(self.Sigma) / (n*T)


    def __repr__(self):
        return str(self.__dict__)


# Define the main Estimation class
class Estimation:
    def __init__(self):
        self.param = Param()
        self.nb = Estimator()  # Updated to nb
        self.bc = Estimator()  # Updated to bc
        self.exitflag = None

    def __repr__(self):
        return f"Estimation(\n  param={self.param},\n  nb={self.nb},\n  bc={self.bc},\n  exitflag={self.exitflag}\n)"



# Define the obj_Q function
def obj_Q(arg, Y, W, Xy, Z, Xz, Y0, Z0, W0, Ry, Rz):
    """
    Objective function
    """
    # arg = theta_0
    p = Z.shape[1]
    J = p * (p + 1) // 2
    n, T = Y.shape
    ky = Xy.shape[2]
    max_kzp = Xz.shape[2]

    # Compute kzp for each j
    kzp = np.array([
            max_kzp if not np.isnan(Xz[:, :, :, j]).any() else
            int( np.nanmax(np.where( np.isnan(Xz[:, :, :, j]) )[2]) )
            for j in range(p)
          ])
    kz = int(np.sum(kzp))

    # Assign parameters
    beta_y = arg[:ky]
    gamma = np.tanh(arg[ky])
    rho = np.tanh(arg[ky + 1])
    lambda_ = np.tanh(arg[ky + 2])
    delta = arg[ky + 3:ky + 3 + p]
    beta_z = arg[ky + 3 + p:ky + 3 + p + kz]
    vect_Upsilon = arg[ky + 3 + p + kz:ky + 3 + p + kz + p ** 2]
    alpha_xi = np.exp(arg[ky + 3 + p + kz + p ** 2])
    sigma_xi = np.sqrt(1 / alpha_xi ** 2)

    alpha = arg[ky + 3 + p + kz + p ** 2 + 1:ky + 3 + p + kz + p ** 2 + 1 + J]
    Sigma_epsilon_neg_half = np.zeros((p, p))
    Sigma_epsilon_neg_half[np.tril_indices(p)] = alpha.T
    Sigma_epsilon_neg_half += tril(Sigma_epsilon_neg_half, -1).T
    Sigma_epsilon = np.linalg.inv(Sigma_epsilon_neg_half) @ np.linalg.inv(Sigma_epsilon_neg_half).T

    In = np.eye(n)
    Ip = np.eye(p)
    S = np.zeros((n, n, T))
    for t in range(T):
        S[:, :, t] = In - lambda_ * W[:, :, t]

    # Write variable in matrix form
    Xy_m = Xy
    last_Y_m = np.hstack([Y0, Y[:, :-1]])
    last_WY_m = np.hstack( [W0 @ Y0 ,np.hstack([W[:, :, t] @ Y[:, t].reshape(-1, 1) for t in range(T - 1)])]  )
    SY_m = np.hstack([S[:,:,t] @ Y[:, t].reshape(-1,1) for t in range(T)])

    Z_m = Z.reshape(n * p, T, order='F')
    Xz_m = np.full((n * p, T, kz), np.nan)
    last_Z_m = np.full((n * p, T, p * p), np.nan)

    for t in range(T):
        if t != 0:
            last_Z_m[:, t, :] = np.kron(Ip, Z[:, :, t - 1])
        else:
            last_Z_m[:, t, :] = np.kron(Ip, Z0)

        Xntz = []
        for j in range(p):
            if j == 0:
                Xntz = Xz[:, t, :kzp[j], j].reshape(n, kzp[j])
            else:
                Xntz = block_diag(Xntz, Xz[:, t, :kzp[j], j].reshape(n, kzp[j]))

        Xz_m[:, t, :] = Xntz

    # Dz
    Xzsum = np.zeros((n * p, T))
    for k in range(p ** 2):
        Xzsum += last_Z_m[:, :, k] * vect_Upsilon[k]
    for k in range(kz):
        Xzsum += Xz_m[:, :, k] * beta_z[k]
    Dz = np.kron(Sigma_epsilon_neg_half, In) @ (Z_m - Xzsum)

    # Dy
    Xysum = gamma * last_Y_m + rho * last_WY_m
    for k in range(ky):
        Xysum += Xy_m[:, :, k] * beta_y[k]
    Dy = 1 / sigma_xi * (SY_m - Xysum - np.kron(delta.T, In) @ (Z_m - Xzsum))



    # Components of the objective function
    sum_ln_det_Snt = np.sum([np.log(det(S[:,:,t])) for t in range(T)])

    ez = np.sort(np.linalg.eigvalsh(Dz @ Dz.T))[ : : -1]
    Lz = np.sum(ez[Rz:]) / (n * T)

    ey = np.sort(np.linalg.eigvalsh(Dy @ Dy.T))[ : : -1]
    Ly = np.sum(ey[Ry:]) / (n * T)

    # Q
    Q = (-0.5 * np.log(det(Sigma_epsilon))
         - 0.5 * np.log(sigma_xi ** 2)
         + sum_ln_det_Snt / (n * T)
         - 0.5 * Lz - 0.5 * Ly)

    return -Q


# estimate factors and factor loadings and define the projection matrix
def H(theta, Y, W, Xy, Z, Xz, Y0, Z0, W0, param_class):
    # theta = est.nb.theta; param_class = est.param
    p = param_class.p
    J = param_class.J
    n = param_class.n
    T = param_class.T
    ky = param_class.ky
    kzp = param_class.kzp
    kz = param_class.kz

    In = eye(n).toarray()  # Sparse identity matrix for n
    Ip = eye(p).toarray()  # Sparse identity matrix for p

    # Assign parameters
    beta_y = theta[:ky].reshape(-1,1)
    gamma = theta[ky]
    rho = theta[ky + 1]
    lambda_ = theta[ky + 2]
    delta = theta[ky + 3:ky + 3 + p].reshape(-1,1)
    beta_z = theta[ky + 3 + p:ky + 3 + p + kz].reshape(-1,1)
    vect_Upsilon = theta[ky + 3 + p + kz:ky + 3 + p + kz + p ** 2].reshape(-1,1)
    alpha_xi = theta[ky + 3 + p + kz + p ** 2]
    alpha = theta[ky + 3 + p + kz + p ** 2 + 1:ky + 3 + p + kz + p ** 2 + 1 + J].reshape(-1,1)

    # Construct Sigma_epsilon_neg_half
    Sigma_epsilon_neg_half = np.zeros((p, p))
    Sigma_epsilon_neg_half[np.tril_indices(p)] = alpha.T
    Sigma_epsilon_neg_half += tril(Sigma_epsilon_neg_half, -1).T


    # Initialize Hz and Hy
    Hz = np.full((n * p, T), np.nan)
    Hy = np.full((n, T), np.nan)

    # Initial conditions
    last_Z = Z0
    last_y = Y0
    last_Wy = W0 @ Y0
    S = np.full((n, n, T), np.nan)

    for t in range(T):
        S[:, :, t] = In - lambda_ * W[:, :, t]
        z_nt = Z[:, :, t].reshape(-1,1,order='F')
        y_nt = Y[:, t].reshape(-1,1,order='F')
        Xnty = Xy[:, t, :].reshape(n, ky,order='F')

        Xntz = []
        for j in range(p):
            if j == 0:
                Xntz = Xz[:, t, :kzp[j], j].reshape(n, kzp[j])
            else:
                Xntz = block_diag(Xntz, Xz[:, t, :kzp[j], j].reshape(n, kzp[j]))

        # Compute dz
        dz = np.kron(Sigma_epsilon_neg_half, In) @ (
                z_nt - np.kron(Ip, last_Z) @ vect_Upsilon - Xntz @ beta_z
        )
        Hz[:, t] = dz.flatten()

        # Compute dy
        dy = alpha_xi * (
                S[:, :, t] @ y_nt.reshape(-1,1)
                - gamma * last_y
                - rho * last_Wy
                - Xnty @ beta_y.reshape(-1,1)
                - np.kron(delta.T, In) @ (
                        z_nt - np.kron(Ip, last_Z) @ vect_Upsilon - Xntz @ beta_z
                )
        )
        Hy[:, t] = dy.flatten()

        # Update last_Z, last_y, and last_Wy
        last_Z = Z[:, :, t]
        last_y = y_nt
        last_Wy = W[:, :, t] @ y_nt

    return Hy, Hz



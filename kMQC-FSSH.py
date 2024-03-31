#!/usr/bin/python -tt
from libraries import *
import sys
from numba import jit
import ray
import scipy

ray.init(ignore_reinit_error=True)
inputfile = str(sys.argv[1])
with open(inputfile) as f:
    for line in f:
        line1 = line.replace(" ", "")
        line1 = line1.rstrip('\n')
        name, value = line1.split("=")
        # exec('global '+str(name))
        exec(str(line))

kB = 1
hbar=1

if space == 'k-space':
    # initialize k-space grids and truncations
    a = np.pi  # lattice parameter (should be pi)
    ran = [-np.pi / (rvar * a), np.pi / (rvar * a)]  # range of BZ included in calculation
    kgrid = np.delete(np.linspace(-np.pi / a, np.pi / a, npoints + 1), -1)  # untruncated k grid
    wgrid = np.zeros_like(kgrid) + w  # classical oscillator frequency dispersion for one optical mode
    wgridQ = 2 * J * np.cos(2 * np.pi * (kgrid / (kgrid[1] - kgrid[0])) / npoints)  # quantum dispersion

if space == 'r-space':
    rvar = 1 # in real-space rvar must always be 1, no truncation
    a = np.pi  # lattice parameter (should be pi)
    ran = [-np.pi / (rvar * a), np.pi / (rvar * a)]  # range of BZ included in calculation
    kgrid = np.delete(np.linspace(-np.pi / a, np.pi / a, npoints + 1), -1)  # untruncated k grid
    wgrid = np.zeros_like(kgrid) + w  # classical oscillator frequency dispersion for one optical mode
    wgridQ = 2 * J * np.cos(2 * np.pi * (kgrid / (kgrid[1] - kgrid[0])) / npoints)  # quantum dispersion
dk = kgrid[1] - kgrid[0] # grid spacing
aux = '_FSSH'

foldername = str(space) + '_' + str(model) + '_t_' + str(tmax) + '_dt_' + str(dt) + '_npts_' + str(
    npoints) + '_J_' + str(J) + '_w_' + str(w) + '_g_' \
             + str(gc) + '_T0_' + str(temp) + '_ran_' + str(ran[0]) + '_' + str(ran[1]) + aux
filename = foldername
if not (path.exists(foldername)):
    os.mkdir(foldername)
filename = foldername + '/' + filename
if space == 'r-space':
    if rvar != 1:
        print('ERROR: no truncation allowed')
        sys.exit()

########## Shared Functions #############
nkgrid = np.arange(0, npoints, dtype=int)


def ktoIndex1(k): # convert k value to grid index
    index = round((k + (np.pi / a)) / dk, 0)
    if index < 0:
        index = index + (npoints)
    if index > npoints - 1:
        index = index - (npoints)
    return int(round(index))

# construct truncated energy and frequency grids
untkgrid = kgrid # untruncated grid
kgrid = np.array([k for k in kgrid if k <= ran[1] and k >= ran[0]]) # truncated grid
wgrid = np.array([wgrid[ktoIndex1(k)] for k in kgrid if k <= ran[1] and k >= ran[0]]) # truncated classical frequency
tnkgrid = np.array([nkgrid[ktoIndex1(k)] for k in kgrid if k <= ran[1] and k >= ran[0]]) # truncated grid index
wgridQ = np.array([wgridQ[ktoIndex1(k)] for k in kgrid if k <= ran[1] and k >= ran[0]]) # truncated quantum frequency
egridQ = hbar * wgridQ # truncated quantum energy


# construct full fourier transform matrix
F_nk_full = np.zeros((len(nkgrid),len(nkgrid)),dtype=complex)
for n in range(len(nkgrid)):
    for k_n in range(len(nkgrid)):
        F_nk_full[n, k_n] = (1 / np.sqrt(len(nkgrid))) * np.exp(1.0j * np.pi * n * untkgrid[k_n])

# construct truncated foruier transform matrix
F_nk_trunc = np.zeros((len(nkgrid),len(tnkgrid)),dtype=complex)
for n in range(len(nkgrid)):
    for k_n in range(len(tnkgrid)):
        F_nk_trunc[n, k_n] = (1 / np.sqrt(len(nkgrid))) * np.exp(1.0j * np.pi * n * kgrid[k_n])
@jit(nopython=True)
def cycle(index): # loop grid index over the grid
    while index > npoints - 1 or index < 0:
        if index > npoints - 1:
            index = index - npoints
        if index < 0:
            index = index + npoints
    return index


cyclev = np.vectorize(cycle)


def init_classical(): # sample classical coordinates from a boltzmann distribution
    p = np.array([])
    q = np.array([])
    for w in wgrid:
        if w == 0.0:
            q = np.append(q, np.array([0]))
            p = np.append(p, np.array([0]))
        else:
            q = np.append(q, np.random.normal(0, np.sqrt(kB * temp) / w, 1))
            p = np.append(p, np.random.normal(0, np.sqrt(kB * temp), 1))
    return np.array(p), np.array(q)


def init_classical_parallel(nt): # generate coordinates for each trajectory
    p = np.zeros((nt, len(wgrid)))
    q = np.zeros((nt, len(wgrid)))
    for i in range(nt):
        p[i], q[i] = init_classical()
    return p, q

# initialize wavefunction
with open(coeff_file) as f:
    for line in f:
        line = line.rstrip('\n')
        exec(str(line))


@jit(nopython=True)
def RK4(p_bath, q_bath, QF, dt): # Evovle classical coordinates with RK4
    Fq, Fp = QF
    K1 = dt * (p_bath + Fp)
    L1 = -dt * (wgrid ** 2 * q_bath + Fq)  # [wn2] is w_alpha ^ 2
    K2 = dt * ((p_bath + 0.5 * L1) + Fp)
    L2 = -dt * (wgrid ** 2 * (q_bath + 0.5 * K1) + Fq)
    K3 = dt * ((p_bath + 0.5 * L2) + Fp)
    L3 = -dt * (wgrid ** 2 * (q_bath + 0.5 * K2) + Fq)
    K4 = dt * ((p_bath + L3) + Fp)
    L4 = -dt * (wgrid ** 2 * (q_bath + K3) + Fq)
    q_bath = q_bath + 0.166667 * (K1 + 2 * K2 + 2 * K3 + K4)
    p_bath = p_bath + 0.166667 * (L1 + 2 * L2 + 2 * L3 + L4)
    return p_bath, q_bath


def timestepRK_Q(mat, cgrid, dt): # Evolve quantum coefficients with RK4
    def f_qho(t, c):
        return -1.0j * np.matmul(mat, c)
    soln4 = it.solve_ivp(f_qho, (0, dt[-1]), cgrid, method='RK45', max_step=dt[0],
                         t_eval=dt)  # , rtol=1e-10, atol=1e-10)
    return np.transpose(soln4.y)


def timestepRK_Q(mat, cgrid, dt): # Evolve quantum coefficients with RK4
    def f_qho(t, c):
        return -1.0j * np.matmul(mat, c)
    soln4 = it.solve_ivp(f_qho, (0, dt[-1]), cgrid, method='RK45', max_step=dt[0],
                         t_eval=dt)  # , rtol=1e-10, atol=1e-10)
    return np.transpose(soln4.y)

def rho_0_adb_to_db(rho_0_adb, eigvec): # Transform initial adiabatic density matrix to diabatic basis
    rho_0_db = np.dot(np.dot(eigvec, rho_0_adb), np.conj(eigvec).transpose())
    return rho_0_db


def rho_0_db_to_adb(rho_0_db, eigvec): # Transform initial diabatic density matrix to adiabatic basis
    rho_0_adb = np.dot(np.dot(np.conj(eigvec).transpose(), rho_0_db), eigvec)
    return rho_0_adb


def rho_adb_to_db(rho_adb, eigvec): # Transform branched adiabatic density matrix to diabatic basis
    rho_db = np.zeros_like(rho_adb)
    # for i in range(npoints):
    #    rho_db[i] = np.dot(np.dot(eigvec[i],rho_adb[i]),np.conj(eigvec[i]).transpose())

    # transpose eigvec matrix for each branch (page)
    rho_db = np.matmul(np.matmul(eigvec, rho_adb), np.transpose(np.conj(eigvec), axes=(0, 2, 1)))
    return rho_db


def rho_db_to_adb(rho_db, eigvec): # transform branched diabatic density matrix to adiabatic basis
    # transpose eigvec matrix for each branch (page)
    rho_adb = np.matmul(np.matmul(np.transpose(np.conj(eigvec), axes=(0, 2, 1)), rho_db), eigvec)
    return rho_adb


def vec_adb_to_db(psi_adb, eigvec):
    # in each branch, take eigvector matrix (last two indices) and multiply by psi (a raw in a matrix):
    psi_db = np.einsum('...ij,...j', eigvec, psi_adb)
    return psi_db


def vec_db_to_adb(psi_db, eigvec):
    # in each branch, take eigvector matrix (last two indices), transpose it, and multiply by psi (a raw in a matrix):
    psi_ad = np.einsum('...ij,...i', eigvec, psi_db)
    return psi_ad


@jit(nopython=True)
def nan_num(num):
    if np.isnan(num):
        return 0.0
    if num == np.inf:
        return 100e100
    if num == -np.inf:
        return -100e100
    else:
        return num


nan_num_vec = np.vectorize(nan_num)
@jit(nopython=True) # gauge transform eigenvectors to ensure parallel transport
def sign_adjust(eigvec_sort, eigvec_prev):
    wf_overlap = np.sum(np.conj(eigvec_prev) * eigvec_sort, axis=0)
    phase = wf_overlap / np.abs(wf_overlap)
    eigvec_out = np.zeros_like(eigvec_sort + 0.0 + 0.0j)
    for n in range(len(eigvec_sort)):
        eigvec_out[:, n] = eigvec_sort[:, n] * np.conj(phase[n])
    return eigvec_out

def hamilt_diag(hamilt, eigvec_previous): # diagonalize Hamiltonian and adjust sign in each branch
    eigval_out = np.zeros((len(kgrid), len(kgrid)), dtype=complex)
    eigvec_out = np.zeros((len(kgrid), len(kgrid), len(kgrid)), dtype=complex)
    eigval, eigvec = np.linalg.eigh(hamilt)
    for i in range(len(hamilt)):
        eigval_out[i], eigvec_out[i] = eigval[i], sign_adjust(eigvec[i], eigvec_previous[i])
    return eigval_out, eigvec_out

def rescale_dkk(dkkq, dkkp): # choose gauge that turns dkk real
    phase = np.angle(dkkq[np.argmax(np.abs(dkkq))])
    return dkkq*np.exp(-1.0j*phase), dkkp*np.exp(-1.0j*phase)

def boltz_grid(t, egrid): # compute boltzmann populations
    z = np.sum(np.exp(-1.0 * (1.0 / (kB * t)) * egrid), axis=1)
    return np.exp(-1.0 * (1.0 / (kB * t)) * egrid) / (z.reshape((-1, 1)))

########## Real-Space Functions #########
if space == 'r-space':
    if model == 'holstein':
        @jit(nopython=True)
        def q_mat():  # quantum Hamiltonian
            e1 = np.diag(np.zeros(npoints))
            for n in range(npoints):
                e1[cycle(n + 1), cycle(n)] += J
                e1[cycle(n), cycle(n + 1)] += J
            return e1


        @jit(nopython=True)
        def qc_mat(p, q):  # quantum-classical Hamiltonian
            out_mat = np.asfortranarray(np.zeros((npoints, npoints, npoints)))
            for n in range(npoints):
                out_mat[n] = np.diag(gc * wgrid * np.sqrt(2 * wgrid) * q[n])
            return out_mat


        @jit(nopython=True)
        def quantumForce(coeffgrid):  # quantum Force
            return np.real(np.conj(coeffgrid) * gc * wgrid * np.sqrt(2 * wgrid) * coeffgrid), np.real(0 * coeffgrid)


        # @jit(nopython=True)
        def get_dkk(eig_k, eig_j, evdiff): # nonadiabatic couplings
            dkkq = np.real((np.conj(eig_k) * gc * wgrid * np.sqrt(2 * wgrid) * eig_j) / evdiff)
            dkkp = np.zeros(npoints)
            return dkkq, dkkp
    if model == 'peierls':
        @jit(nopython=True)
        def q_mat(): # quantum Hamiltonian in single branch
            e1 = np.diag(np.ones(npoints))
            for n in range(npoints):
                e1[cycle(n + 1), cycle(n)] += J
                e1[cycle(n), cycle(n + 1)] += J
            return e1


        @jit(nopython=True)
        def qc_mat_gen(p, q): # quantum-classical Hamiltonian in single branch
            mat = np.zeros((npoints, npoints))
            for n in range(npoints):
                mat[cycle(n + 1), cycle(n)] += gc * wgrid[n] * np.sqrt(2) * (
                            np.sqrt(wgrid[cycle(n)]) * q[cycle(n)] - np.sqrt(wgrid[cycle(n + 1)]) * q[cycle(n + 1)])
                mat[cycle(n), cycle(n + 1)] += gc * wgrid[n] * np.sqrt(2) * (
                            np.sqrt(wgrid[cycle(n)]) * q[cycle(n)] - np.sqrt(wgrid[cycle(n + 1)]) * q[cycle(n + 1)])
            return mat


        @jit(nopython=True)
        def qc_mat(p, q): # quantum-classical Hamiltonian in all branches
            out_mat = np.asfortranarray(np.zeros((npoints, npoints, npoints)))
            for n in range(npoints):
                out_mat[n] = qc_mat_gen(p[n], q[n])
            return out_mat



        @jit(nopython=True)
        def quantumForce(coeffgrid): # quantum force
            Fq = np.zeros_like(coeffgrid)
            Fp = np.zeros_like(coeffgrid)
            for n in range(npoints):
                Fq[n] += gc * wgrid[n] * np.sqrt(2 * wgrid[n]) * 2 * (
                            np.real(np.conj(coeffgrid[cycle(n + 1)]) * coeffgrid[cycle(n)]) - np.real(
                        np.conj(coeffgrid[cycle(n - 1)]) * coeffgrid[cycle(n)]))
                Fp[n] += 0
            return np.real(Fq), np.real(Fp)


        # @jit(nopython=True)
        def get_dkk(eig_k, eig_j, evdiff): # nonadiabatic coupling
            dkkq = np.ascontiguousarray(np.zeros(npoints))
            for n in range(npoints):
                dkkq[n] = np.real(gc * wgrid[n] * np.sqrt(2 * wgrid[n]) * 2 * (
                            np.real(np.conj(eig_k[cycle(n + 1)]) * eig_j[cycle(n)]) - np.real(
                        np.conj(eig_k[cycle(n - 1)]) * eig_j[cycle(n)])) / evdiff)
            dkkp = np.ascontiguousarray(np.zeros(npoints))
            return dkkq, dkkp
########### K-Space Functions ###########
if space == 'k-space':
    if model == 'holstein_impurity':

        @jit(nopython=True)
        def q_mat_site(): # quantum Hamiltonian (nearest neighbor) in site basis
            e1 = np.diag(np.zeros(npoints))
            for n in range(npoints):
                e1[cycle(n + 1), cycle(n)] += J
                e1[cycle(n), cycle(n + 1)] += J
            return e1


        #@jit(nopython=True)
        def q_mat(): # quantum Hamiltonian in truncated k-space basis
            H_q_site = q_mat_site()  # full real-space matrix
            H_q_site[impurity_site, impurity_site] += impurity_energy
            H_q_k_trunc = np.matmul(np.conjugate(np.transpose(F_nk_trunc)), np.matmul(H_q_site, F_nk_trunc))
            return H_q_k_trunc


        @jit(nopython=True)
        def quantumForce(coeffgrid): # quantum force on truncated grid
            cg = np.ascontiguousarray(np.zeros(npoints) + 0.0j)
            cg[tnkgrid] = coeffgrid
            fq = np.ascontiguousarray(np.zeros(npoints))
            fp = np.ascontiguousarray(np.zeros(npoints))
            for kappa in nkgrid:
                cgroll_pk = np.ascontiguousarray(np.conj(np.concatenate((cg[-kappa:], cg[:-kappa]))))
                fq[cycle(int(npoints / 2) + kappa)] = gc * (np.sqrt(2 * w ** 3) / np.sqrt(npoints)) * np.real(
                    np.dot(cgroll_pk, cg))
                fp[cycle(int(npoints / 2) + kappa)] = -gc * (np.sqrt(2 * w) / np.sqrt(npoints)) * np.imag(
                    np.dot(cgroll_pk, cg))
            return fq[tnkgrid], fp[tnkgrid]


        @jit(nopython=True)
        def qc_mat_gen(p, q): # quantum-classical Hamiltonian in single branch
            pn = np.zeros(npoints)
            qn = np.zeros(npoints)
            pn[tnkgrid], qn[tnkgrid] = p, q
            kaparray = np.asfortranarray(np.zeros(npoints) + 0.0j)
            for kappa in nkgrid:
                kaparray[cycle(int(npoints / 2) + kappa)] += gc * np.sqrt(w / (2 * npoints)) * (
                            w * (qn[-cycle(kappa)] + qn[cycle(kappa)]) - 1.0j * (pn[-cycle(kappa)] - pn[cycle(kappa)]))
            outmat = np.asfortranarray(np.zeros((npoints, npoints)) + 0.0j)
            for n in nkgrid:
                outmat[n] = kaparray
                kaparray = np.concatenate((kaparray[-1:], kaparray[:-1]))
            return outmat[tnkgrid][:, tnkgrid]


        @jit(nopython=True)
        def qc_mat(p, q):  # accepts all branches
            outmat = np.asfortranarray(np.zeros((len(p), len(tnkgrid), len(tnkgrid))) + 0.0j)
            for i in range(len(p)):
                outmat[i] = qc_mat_gen(p[i], q[i])
            return outmat


        @jit(nopython=True)
        def get_dkk(eig_k_in, eig_j_in, evdiff): # nonadiabaitc coupling in truncated basis
            dkkq = np.ascontiguousarray(np.zeros(npoints) + 0.0j)
            dkkp = np.ascontiguousarray(np.zeros(npoints) + 0.0j)
            eig_k_in = np.ascontiguousarray(eig_k_in)
            eig_j_in = np.ascontiguousarray(eig_j_in)
            if space == 'k-space':
                eig_k = np.ascontiguousarray(np.zeros(npoints) + 0.0j)
                eig_j = np.ascontiguousarray(np.zeros(npoints) + 0.0j)
            if space == 'r-space':
                eig_k = np.ascontiguousarray(np.zeros(npoints))
                eig_j = np.ascontiguousarray(np.zeros(npoints))
            eig_k[tnkgrid], eig_j[tnkgrid] = eig_k_in, eig_j_in
            for kappa in nkgrid:
                eig_k_mk = np.ascontiguousarray(np.conj(np.concatenate((eig_k[kappa:], eig_k[:kappa]))))
                eig_k_pk = np.ascontiguousarray(np.conj(np.concatenate((eig_k[-kappa:], eig_k[:-kappa]))))
                dkkq[cycle(int(npoints / 2) + kappa)] = gc * (np.sqrt(w ** 3) / np.sqrt(2 * npoints)) * (
                            np.dot(eig_k_mk, eig_j) + np.dot(eig_k_pk, eig_j))
                dkkp[cycle(int(npoints / 2) + kappa)] = -1.0j * (gc / np.sqrt(npoints)) * np.sqrt(w / 2) * (
                            np.dot(eig_k_mk, eig_j) - np.dot(eig_k_pk, eig_j))
            dkkq = dkkq / evdiff
            dkkp = dkkp / evdiff
            return dkkq[tnkgrid], dkkp[tnkgrid]
    if model == 'holstein':
        @jit(nopython=True)
        def quantumForce(coeffgrid): # quantum force
            cg = np.ascontiguousarray(np.zeros(npoints) + 0.0j)
            cg[tnkgrid] = coeffgrid
            fq = np.ascontiguousarray(np.zeros(npoints))
            fp = np.ascontiguousarray(np.zeros(npoints))
            for kappa in nkgrid:
                cgroll_pk = np.ascontiguousarray(np.conj(np.concatenate((cg[-kappa:], cg[:-kappa]))))
                fq[cycle(int(npoints / 2) + kappa)] = gc * (np.sqrt(2 * w ** 3) / np.sqrt(npoints)) * np.real(
                    np.dot(cgroll_pk, cg))
                fp[cycle(int(npoints / 2) + kappa)] = -gc * (np.sqrt(2 * w) / np.sqrt(npoints)) * np.imag(
                    np.dot(cgroll_pk, cg))
            return fq[tnkgrid], fp[tnkgrid]


        @jit(nopython=True)
        def q_mat(): # quantum Hamiltonian in k-space
            return np.diag(egridQ)

        @jit(nopython=True)
        def qc_mat_gen(p, q): # quantum-classical Hamiltonian in a single branch
            pn = np.zeros(npoints)
            qn = np.zeros(npoints)
            pn[tnkgrid], qn[tnkgrid] = p, q
            kaparray = np.asfortranarray(np.zeros(npoints) + 0.0j)
            for kappa in nkgrid:
                kaparray[cycle(int(npoints / 2) + kappa)] += gc * np.sqrt(w / (2 * npoints)) * (
                            w * (qn[-cycle(kappa)] + qn[cycle(kappa)]) - 1.0j * (pn[-cycle(kappa)] - pn[cycle(kappa)]))
            outmat = np.asfortranarray(np.zeros((npoints, npoints)) + 0.0j)
            for n in nkgrid:
                outmat[n] = kaparray
                kaparray = np.concatenate((kaparray[-1:], kaparray[:-1]))
            return outmat[tnkgrid][:, tnkgrid]

        @jit(nopython=True)
        def qc_mat(p, q):  # accepts all branches
            outmat = np.asfortranarray(np.zeros((len(p), len(tnkgrid), len(tnkgrid))) + 0.0j)
            for i in range(len(p)):
                outmat[i] = qc_mat_gen(p[i], q[i])
            return outmat

        @jit(nopython=True)
        def get_dkk(eig_k_in, eig_j_in, evdiff):
            dkkq = np.ascontiguousarray(np.zeros(npoints) + 0.0j)
            dkkp = np.ascontiguousarray(np.zeros(npoints) + 0.0j)
            eig_k_in = np.ascontiguousarray(eig_k_in)
            eig_j_in = np.ascontiguousarray(eig_j_in)
            if space == 'k-space':
                eig_k = np.ascontiguousarray(np.zeros(npoints) + 0.0j)
                eig_j = np.ascontiguousarray(np.zeros(npoints) + 0.0j)
            if space == 'r-space':
                eig_k = np.ascontiguousarray(np.zeros(npoints))
                eig_j = np.ascontiguousarray(np.zeros(npoints))
            eig_k[tnkgrid], eig_j[tnkgrid] = eig_k_in, eig_j_in
            for kappa in nkgrid:
                eig_k_mk = np.ascontiguousarray(np.conj(np.concatenate((eig_k[kappa:], eig_k[:kappa]))))
                eig_k_pk = np.ascontiguousarray(np.conj(np.concatenate((eig_k[-kappa:], eig_k[:-kappa]))))
                dkkq[cycle(int(npoints / 2) + kappa)] = gc * (np.sqrt(w ** 3) / np.sqrt(2 * npoints)) * (
                            np.dot(eig_k_mk, eig_j) + np.dot(eig_k_pk, eig_j))
                dkkp[cycle(int(npoints / 2) + kappa)] = -1.0j * (gc / np.sqrt(npoints)) * np.sqrt(w / 2) * (
                            np.dot(eig_k_mk, eig_j) - np.dot(eig_k_pk, eig_j))
            dkkq = dkkq / evdiff
            dkkp = dkkp / evdiff
            return dkkq[tnkgrid], dkkp[tnkgrid]
    if model == 'peierls':
        # generate phase-shift matrix for peierls coupling in k-space
        expmat2 = np.array([])
        for k1 in kgrid:
            for k2 in kgrid:
                if ktoIndex1(k2 - k1) >= ktoIndex1(kgrid[0]) and ktoIndex1(k2 - k1) <= ktoIndex1(kgrid[-1]):
                    if ktoIndex1(k1 - k2) >= ktoIndex1(kgrid[0]) and ktoIndex1(k1 - k2) <= ktoIndex1(kgrid[-1]):
                        expmat2 = np.append(expmat2, 2.0j * (np.sin(a * k1) - np.sin(a * k2)))
                    else:
                        expmat2 = np.append(expmat2, 0)
                else:
                    expmat2 = np.append(expmat2, 0)
        expmat2 = expmat2.reshape(len(kgrid), len(kgrid))


        @jit(nopython=True)
        def quantumForce(coeffgrid): # quantum force
            cg = np.ascontiguousarray(np.zeros(npoints) + 0.0j)
            cg[tnkgrid] = coeffgrid
            fq = np.ascontiguousarray(np.zeros(npoints))
            fp = np.ascontiguousarray(np.zeros(npoints))
            fact = np.ascontiguousarray(np.zeros(npoints))
            for kappa in nkgrid:
                fact[tnkgrid] = \
                (np.sin(a * (untkgrid + untkgrid[cycle(int(npoints / 2) + kappa)])) - np.sin(a * untkgrid))[tnkgrid]
                cgroll_pk = np.ascontiguousarray(np.conj(np.concatenate((cg[kappa:], cg[:kappa]))))
                fq[cycle(int(npoints / 2) + kappa)] = -gc * (np.sqrt(8 * w ** 3) / np.sqrt(npoints)) * np.imag(
                    np.dot(cgroll_pk, cg * fact))
                fp[cycle(int(npoints / 2) + kappa)] = -gc * (np.sqrt(8 * w) / np.sqrt(npoints)) * np.real(
                    np.dot(cgroll_pk, cg * fact))
            return fq[tnkgrid], fp[tnkgrid]


        @jit(nopython=True)
        def q_mat(): # quantum Hamiltonian
            return np.diag(egridQ)


        @jit(nopython=True)
        def qc_mat_gen(p, q): # quantum-classical Hamiltonian in a single branch
            pn = np.zeros(npoints)
            qn = np.zeros(npoints)
            pn[tnkgrid], qn[tnkgrid] = p, q
            kaparray = np.asfortranarray(np.zeros(npoints) + 0.0j)
            for kappa in nkgrid:
                kaparray[cycle(int(npoints / 2) + kappa)] += gc * np.sqrt(w / (2 * npoints)) * (
                            w * (qn[-kappa] + qn[kappa]) - 1.0j * (pn[-kappa] - pn[kappa]))
            outmat = np.asfortranarray(np.zeros((npoints, npoints)) + 0.0j)
            for n in nkgrid:
                outmat[n] = kaparray
                kaparray = np.concatenate((kaparray[-1:], kaparray[:-1]))
            outmat = np.transpose(outmat)
            return outmat[tnkgrid][:, tnkgrid] * expmat2


        @jit(nopython=True)
        def qc_mat(p, q):  # accepts all branches
            outmat = np.asfortranarray(np.zeros((len(p), len(tnkgrid), len(tnkgrid))) + 0.0j)
            for i in range(len(p)):
                outmat[i] = qc_mat_gen(p[i], q[i])
            return outmat

        @jit(nopython=True)
        def get_dkk(eig_k_in, eig_j_in, evdiff): # nonadiabatic coupling in in truncated basis
            dkkq = np.ascontiguousarray(np.zeros(npoints) + 0.0j)
            dkkp = np.ascontiguousarray(np.zeros(npoints) + 0.0j)
            eig_k_in = np.ascontiguousarray(eig_k_in)
            eig_j_in = np.ascontiguousarray(eig_j_in)
            fact_pk = np.ascontiguousarray(np.zeros(npoints) + 0.0j)
            fact_mk = np.ascontiguousarray(np.zeros(npoints) + 0.0j)
            eig_k = np.ascontiguousarray(np.zeros(npoints) + 0.0j)
            eig_j = np.ascontiguousarray(np.zeros(npoints) + 0.0j)
            eig_k[tnkgrid], eig_j[tnkgrid] = eig_k_in, eig_j_in
            for kappa in nkgrid:
                eig_j_pk = np.ascontiguousarray(np.concatenate((eig_j[kappa:], eig_j[:kappa])))
                eig_k_pk = np.ascontiguousarray(np.concatenate((eig_k[kappa:], eig_k[:kappa])))
                fact_pk[tnkgrid] = \
                (np.sin(a * (untkgrid + untkgrid[cycle(int(npoints / 2) + kappa)])) - np.sin(a * untkgrid))[tnkgrid]
                dkkq[cycle(int(npoints / 2) + kappa)] = 1.0j * gc * np.sqrt(2 * (w ** 3) / npoints) * (
                            np.dot(np.conj(eig_k_pk), eig_j * fact_pk) - np.dot(np.conj(eig_k), eig_j_pk * fact_pk))
                dkkp[cycle(int(npoints / 2) + kappa)] = -1.0 * gc * np.sqrt(2 * w / npoints) * (
                            np.dot(np.conj(eig_k_pk), eig_j * fact_pk) + np.dot(np.conj(eig_k), eig_j_pk * fact_pk))
            dkkq = dkkq / evdiff
            dkkp = dkkp / evdiff
            return dkkq[tnkgrid], dkkp[tnkgrid]


@ray.remote
def runSim(index, p0, q0):
    npoints = len(tnkgrid)
    start_time = time.time()  # initialize starting time for trajectory

    ## initialize classical coordinates ##
    p = np.zeros((npoints, npoints))  # put initial conditions into p, q, array for all branches.
    q = np.zeros((npoints, npoints))
    p[:] = p0
    q[:] = q0

    ## initialize active surface
    act_surf_ind = np.arange(0, npoints)  # array containing the position of the active surface index in each branch
    act_surf = np.diag(np.ones(npoints))  # initialize tha active surface matrix containing a 1 at the active surface for each branch

    # initialize Hamiltonian
    qmat = q_mat()  # quantum hamiltonian is time independent
    qcmat_store = qc_mat(p, q)
    eigval_0, eigvec_0 = np.linalg.eigh(qmat + qcmat_store[0])
    eigval = np.zeros((npoints, npoints), dtype=complex)
    eigval[:] = eigval_0
    eigvec = np.zeros((npoints, npoints, npoints), dtype=complex)
    eigvec[:] = eigvec_0

    # initialize wavefunctions and density matrices
    den_mat_0 = np.dot(np.conjugate(coeffgrid.reshape((-1, 1))), coeffgrid.reshape((1, -1)))  # initial density matrix
    den_mat_adb = np.zeros((npoints, npoints, npoints), dtype=complex)  # initialize density matrix
    den_mat_0_adb = rho_0_db_to_adb(den_mat_0, eigvec_0)
    cg_db_0 = coeffgrid # initial diabatic wavefunction
    cg_db = np.zeros((npoints, npoints), dtype=complex)
    cg_db[:] = cg_db_0 # initial diabatic wavefunction in all branches
    cg_adb = np.zeros((npoints, npoints), dtype=complex)
    cg_adb_0 = vec_db_to_adb(cg_db_0, eigvec_0)
    cg_adb[:] = cg_adb_0 # initial adiabatic wavefuntion in all branches

    ## initialize arrays to store outputs
    hop_count = 0  # storing hop numbers
    tdat = np.arange(0, tmax + dt, dt)  # time array
    tdat_bath = np.arange(0, tmax + dt_bath, dt_bath)  # time array for bath
    den_mat_db_tot = np.zeros((len(tdat), npoints, npoints, npoints), dtype=complex) # diabatic density matrix in each branch
    den_mat_adb_tot = np.zeros((len(tdat), npoints, npoints, npoints), dtype=complex) # adiabatic density matrix in each branch
    b_pop_db_tot = np.zeros((len(tdat), npoints, npoints, npoints), dtype=complex) # diabatic boltzmann population matrix
    b_pop_adb_tot = np.zeros((len(tdat), npoints, npoints, npoints), dtype=complex) # adiabatic boltzmann population matrix
    eq_out = np.zeros((len(tdat))) # quantum energy
    ec_out = np.zeros((len(tdat))) # classical energy


    # intialize quantum forces
    fq = np.zeros((npoints, npoints))  # initialize the force on the position coordinates (momentum derivative)
    fp = np.zeros((npoints, npoints))  # initialize the force on the momentum coordinates (position derivative)
    for i in range(len(act_surf_ind)):
        if space == 'r-space':
            fq0, fp0 = quantumForce(eigvec[i][:, act_surf_ind[i]])
            fq[i] = fq0
            fp[i] = fp0

        if space == 'k-space':
            fq0, fp0 = quantumForce(eigvec[i][:, act_surf_ind[i]])
            fq[i] = fq0
            fp[i] = fp0
    t_ind = 0
    for t_bath_ind in range(len(tdat_bath)):
        if tdat[t_ind] <= tdat_bath[t_bath_ind] + 0.5 * dt_bath or t_bath_ind == len(tdat_bath) - 1:
            for i in range(npoints):
                den_mat_adb[i] = den_mat_0_adb[i, i] * np.dot(np.conj(cg_adb[i]).reshape((-1, 1)),
                                                              np.array([cg_adb[i]]))
            den_mat_diag = np.diag(den_mat_0_adb).reshape((-1, 1)) * act_surf
            np.einsum('...jj->...j', den_mat_adb)[...] = den_mat_diag  # put act_surf on diagonal of den_mat_adb
            den_mat_db = rho_adb_to_db(den_mat_adb, eigvec)  # transform from adiabatic to diabatic basis set
            den_mat_db_tot[t_ind, :, :, :] = den_mat_db_tot[t_ind, :, :, :] + den_mat_db
            den_mat_adb_tot[t_ind, :, :, :] = den_mat_adb_tot[t_ind, :, :, :] + den_mat_adb
            for i in range(len(act_surf)):
                ec_out[t_ind] += np.sum((1 / 2) * p[i] ** 2 + (1 / 2) * (w ** 2) * q[i] ** 2)
                eq_out[t_ind] += np.real(eigval[i, act_surf_ind[i]])

            e_change = 100 * (ec_out[t_ind] + eq_out[t_ind] - (ec_out[0] + eq_out[0])) / (ec_out[0] + eq_out[0])
            if e_change > 3:
                print('ERROR: energy conservation!',e_change)
            b_fac = boltz_grid(temp, eigval)  ## matrix with the expected boltzman populations for each branch
            b_pop_adb = np.zeros((npoints, npoints, npoints), dtype=complex)
            np.einsum('...jj->...j', b_pop_adb)[...] = b_fac  ## place this on the diagonal of adiabatic density matrix
            b_pop_db = rho_adb_to_db(b_pop_adb, eigvec)  ## transform to the diabatic basis
            b_pop_db_tot[t_ind, :, :, :] = b_pop_db_tot[t_ind, :, :, :] + b_pop_db  # store boltz pop matrices
            b_pop_adb_tot[t_ind, :, :, :] = b_pop_adb_tot[t_ind, :, :, :] + b_pop_adb

            t_ind += 1
        p, q = RK4(p, q, (fq, fp), dt_bath)
        qcmat_store = qc_mat(p, q)
        eigvec_previous = np.copy(eigvec)
        [eigval, eigvec] = hamilt_diag(qmat + qcmat_store, eigvec_previous)
        diag_matrix = np.zeros((npoints, npoints, npoints), dtype=complex)
        eigval_exp = np.exp(-1j * eigval * dt_bath)
        np.einsum('...jj->...j', diag_matrix)[...] = eigval_exp
        cg_adb = np.einsum('...ij,...i', diag_matrix, vec_db_to_adb(cg_db, eigvec))
        cg_db = vec_adb_to_db(cg_adb, np.conj(eigvec))
        rand = np.random.rand()
        for i in range(len(act_surf)):
            prod_A1_0 = (np.matmul(np.conj(eigvec[i][:, act_surf_ind[i]]), eigvec_previous[i]))
            hop_prob = -2 * np.real((cg_adb[i] / cg_adb[i][act_surf_ind[i]]) * prod_A1_0)
            hop_prob[act_surf_ind[i]] = 0
            bin_edge = 0
            for k in range(len(hop_prob)):
                hop_prob[k] = nan_num(hop_prob[k])
                bin_edge = bin_edge + hop_prob[k]
                if rand < bin_edge:
                    eig_k = eigvec[i][:, act_surf_ind[i]]
                    eig_j = eigvec[i][:, k]
                    eigval_k = eigval[i][act_surf_ind[i]]
                    eigval_j = eigval[i][k]
                    ev_diff = eigval_j - eigval_k
                    dkkq, dkkp = get_dkk(eig_k, eig_j, ev_diff)
                    dkkq, dkkp = rescale_dkk(dkkq, dkkp)
                    im_dkkq = np.imag(dkkq)
                    im_dkkp = np.imag(dkkp)
                    if np.sum(np.abs(im_dkkq)) > 1e-6 or np.sum(np.abs(im_dkkq)) > 1e-6:
                        print('ERROR: imaginary dkk')
                    dkkq = np.real(dkkq)
                    dkkp = np.real(dkkp)
                    akkq = (1 / 2) * np.sum(dkkq * dkkq)
                    akkp = (1 / 2) * (w ** 2) * np.sum(dkkp * dkkp)
                    bkkq = np.sum(p[i] * dkkq)
                    bkkp = -(w ** 2) * np.sum(q[i] * dkkp)
                    disc = (bkkq + bkkp) ** 2 - 4 * (akkq + akkp) * ev_diff
                    if disc >= 0:
                        if bkkq + bkkp < 0:
                            gamma = (bkkq + bkkp) + np.sqrt(disc)
                        else:
                            gamma = (bkkq + bkkp) - np.sqrt(disc)
                        if akkp + akkq == 0:
                            gamma = 0
                        else:
                            gamma = gamma / (2 * (akkq + akkp))
                        p[i] = p[i] - np.real(gamma) * dkkq  # rescale
                        q[i] = q[i] + np.real(gamma) * dkkp  # rescale
                        act_surf_ind[i] = k
                        act_surf[i] = np.zeros_like(act_surf[i])
                        act_surf[i][act_surf_ind[i]] = 1
                        hop_count += 1
                    break
            if space == 'r-space':
                fq0, fp0 = quantumForce(eigvec[i][:, act_surf_ind[i]])
                fq[i] = fq0
                fp[i] = fp0

            if space == 'k-space':
                fq0, fp0 = quantumForce(eigvec[i][:, act_surf_ind[i]])
                fq[i] = fq0
                fp[i] = fp0
    den_mat_db_sum = np.sum(den_mat_db_tot[:, :, :, :], axis=1)
    pops_db = np.real(np.einsum('...jj->...j', den_mat_db_sum))
    pops_db_fft = np.real(np.einsum('...jj->...j',np.einsum('aj,njk,bk->nab',F_nk_trunc,den_mat_db_sum,np.conjugate(F_nk_trunc))))
    b_pop_db_sum = np.sum(b_pop_db_tot[:, :, :, :], axis=1) / npoints
    b_pop_db = np.real(np.einsum('...jj->...j', b_pop_db_sum))
    b_pop_db_fft = np.real(np.einsum('...jj->...j',np.einsum('aj,njk,bk->nab',F_nk_trunc,b_pop_db_sum,np.conjugate(F_nk_trunc))))

    simFcpopB = b_pop_db_fft
    simPopB = b_pop_db
    simCdb = pops_db
    simFcdb = pops_db_fft
    simT = tdat
    end_time = time.time()
    print('trial index: ', index, ' hop count: ', hop_count, ' time: ',end_time - start_time)
    return simCdb, simFcdb, simT, simFcpopB, simPopB, eq_out, ec_out


def parallel_run_ray(nt, proc):
    trials = nt
    r_ind = 0
    for run in range(0, int(nt / proc)):
        p, q = init_classical_parallel(proc)
        results = [runSim.remote(run * proc + i, p[i], q[i]) for i in range(proc)]
        for r in results:
            simCdb, simFcdb, simT, simFcpopB, simPopB, simEq, simEc = ray.get(r)
            if run == 0 and r_ind == 0:
                simCdbdat = np.zeros_like(simCdb)
                simFcdbdat = np.zeros_like(simFcdb)
                simTdat = np.zeros_like(simT)
                simFcpopBdat = np.zeros_like(simFcpopB)
                simPopBdat = np.zeros_like(simPopB)
                simEqdat = np.zeros_like(simEq)
                simEcdat = np.zeros_like(simEc)
            simCdbdat += simCdb
            simFcdbdat += simFcdb
            simTdat += simT
            simFcpopBdat += simFcpopB
            simPopBdat += simPopB
            simEqdat += simEq
            simEcdat += simEc
            r_ind += 1

    if path.exists(filename + '_resCdb.csv'):
        simCdbdat += np.loadtxt(filename + '_resCdb.csv', delimiter=",")
    if path.exists(filename + '_resFcdb.csv'):
        simFcdbdat += np.loadtxt(filename + '_resFcdb.csv', delimiter=",")
    if path.exists(filename + '_resT.csv'):
        simTdat += np.loadtxt(filename + '_resT.csv', delimiter=",")
    if path.exists(filename + '_resFcpopB.csv'):
        simFcpopBdat += np.loadtxt(filename + '_resFcpopB.csv', delimiter=",")
    if path.exists(filename + '_resPopB.csv'):
        simPopBdat += np.loadtxt(filename + '_resPopB.csv', delimiter=",")
    if path.exists(filename + '_resEq.csv'):
        simEqdat += np.loadtxt(filename + '_resEq.csv', delimiter=",")
    if path.exists(filename + '_resEc.csv'):
        simEqdat += np.loadtxt(filename + '_resEc.csv', delimiter=",")
    return simCdbdat, simFcdbdat, simTdat, simFcpopBdat, simPopBdat, simEqdat, simEcdat


print('Starting Calculation FSSH\n')
print(filename, '\n')
start_time = time.time()
resCdb, resFcdb, resT, resFcpopB, resPopB, resEq, resEc = parallel_run_ray(ntrials, nprocs)

np.savetxt(filename + '_resCdb.csv', resCdb, delimiter=",")
np.savetxt(filename + '_resFcdb.csv', resFcdb, delimiter=",")
np.savetxt(filename + '_resT.csv', resT, delimiter=",")
np.savetxt(filename + '_resFcpopB.csv', resFcpopB, delimiter=",")
np.savetxt(filename + '_resPopB.csv', resPopB, delimiter=",")
np.savetxt(filename + '_resEq.csv', resEq, delimiter=",")
np.savetxt(filename + '_resEc.csv', resEc, delimiter=",")
np.savetxt(foldername + '/wgrid.csv', wgrid, delimiter=",")
np.savetxt(foldername + '/wgridQ.csv', wgridQ, delimiter=",")
if not (path.exists(foldername + '/' + 'output_file')):
    with open(inputfile) as f:
        f_out = open(foldername + '/' + 'output_file', "a")
        for line in f:
            f_out.write(line)
end_time = time.time()
print('Calculation Duration: ', end_time - start_time, '\n')
print(filename)
#!/usr/bin/python -tt
from libraries import *
import sys
from numba import jit
import ray
import scipy

ray.init(ignore_reinit_error=True)
inputfile = str(sys.argv[1])
rev = False
with open(inputfile) as f:
    for line in f:
        line1 = line.replace(" ", "")
        line1 = line1.rstrip('\n')
        name, value = line1.split("=")
        # exec('global '+str(name))
        exec(str(line))

q_res = 1
c_res = 1
E=0
kB = 1
hbar=1
m=1

io = True
rev=False
te = True



if space == 'k-space':
    a = np.pi  # lattice parameter (should be pi)
    ran = [-np.pi / (rvar * a), np.pi / (rvar * a)]  # range of BZ included in calculation
    kgrid = np.delete(np.linspace(-np.pi / a, np.pi / a, npoints + 1), -1)  # untruncated k grid
    wgrid = np.zeros_like(kgrid) + w  # classical oscillator frequency dispersion for one optical mode
    wgridQ = 2 * J * np.cos(2 * np.pi * (kgrid / (kgrid[1] - kgrid[0])) / npoints)  # quantum dispersion

if space == 'r-space':
    rvar = 1
    a = np.pi  # lattice parameter (should be pi)
    ran = [-np.pi / (rvar * a), np.pi / (rvar * a)]  # range of BZ included in calculation
    kgrid = np.delete(np.linspace(-np.pi / a, np.pi / a, npoints + 1), -1)  # untruncated k grid
    wgrid = np.zeros_like(kgrid) + w  # classical oscillator frequency dispersion for one optical mode
    wgridQ = 2 * J * np.cos(2 * np.pi * (kgrid / (kgrid[1] - kgrid[0])) / npoints)  # quantum dispersion
dk = kgrid[1] - kgrid[0]
aux = '_MF'
io = True

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


def kron(a, b):
    if a == b:
        return 1
    if a != b:
        return 0


def ktoIndex(k):
    index = round((k + (np.pi / (a))) / dk, 0)
    if index < 0:
        index = index + (npoints)
    if index > npoints - 1:
        index = index - (npoints)
    return int(round(index - ((npoints / 2) + (kgrid[0] / dk))))


def ktoIndex1(k):
    index = round((k + (np.pi / a)) / dk, 0)
    if index < 0:
        index = index + (npoints)
    if index > npoints - 1:
        index = index - (npoints)
    return int(round(index))


untkgrid = kgrid
kgrid = np.array([k for k in kgrid if k <= ran[1] and k >= ran[0]])
wgrid = np.array([wgrid[ktoIndex1(k)] for k in kgrid if k <= ran[1] and k >= ran[0]])
tnkgrid = np.array([nkgrid[ktoIndex1(k)] for k in kgrid if k <= ran[1] and k >= ran[0]])
wgridQ = np.array([wgridQ[ktoIndex1(k)] for k in kgrid if k <= ran[1] and k >= ran[0]])
egridQ = hbar * wgridQ


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
def cycle(index):
    while index > npoints - 1 or index < 0:
        if index > npoints - 1:
            index = index - npoints
        if index < 0:
            index = index + npoints
    return index


cyclev = np.vectorize(cycle)


def init_classical():
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


def init_classical_parallel_branch(nt):
    p = np.zeros((nt, len(wgrid), len(wgrid)))
    q = np.zeros((nt, len(wgrid), len(wgrid)))
    for i in range(nt):
        for branch in range(len(wgrid)):
            p[i,branch], q[i,branch] = init_classical()
    return p, q


with open(coeff_file) as f:
    for line in f:
        line = line.rstrip('\n')
        exec(str(line))


@jit(nopython=True)
def RK4(p_bath, q_bath, QF, dt):
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


def timestepRK_Q(mat, cgrid, dt):
    def f_qho(t, c):
        return -1.0j * np.matmul(mat, c)

    soln4 = it.solve_ivp(f_qho, (0, dt[-1]), cgrid, method='RK45', max_step=dt[0] / q_res,
                         t_eval=dt)  # , rtol=1e-10, atol=1e-10)
    return np.transpose(soln4.y)



def rho_0_adb_to_db(rho_0_adb, eigvec):
    rho_0_db = np.dot(np.dot(eigvec, rho_0_adb), np.conj(eigvec).transpose())
    return rho_0_db


def rho_0_db_to_adb(rho_0_db, eigvec):
    rho_0_adb = np.dot(np.dot(np.conj(eigvec).transpose(), rho_0_db), eigvec)
    return rho_0_adb


def rho_adb_to_db(rho_adb, eigvec):
    rho_db = np.zeros_like(rho_adb)
    # for i in range(npoints):
    #    rho_db[i] = np.dot(np.dot(eigvec[i],rho_adb[i]),np.conj(eigvec[i]).transpose())

    # transpose eigvec matrix for each branch (page)
    rho_db = np.matmul(np.matmul(eigvec, rho_adb), np.transpose(np.conj(eigvec), axes=(0, 2, 1)))
    return rho_db


def rho_db_to_adb(rho_db, eigvec):
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
@jit(nopython=True)
def sign_adjust(eigvec_sort, eigvec_prev):
    wf_overlap = np.sum(np.conj(eigvec_prev) * eigvec_sort, axis=0)
    phase = wf_overlap / np.abs(wf_overlap)
    eigvec_out = np.zeros_like(eigvec_sort + 0.0 + 0.0j)
    for n in range(len(eigvec_sort)):
        eigvec_out[:, n] = eigvec_sort[:, n] * np.conj(phase[n])
    return eigvec_out

def hamilt_diag(hamilt, eigvec_previous):
    eigval_out = np.zeros((len(kgrid), len(kgrid)), dtype=complex)
    eigvec_out = np.zeros((len(kgrid), len(kgrid), len(kgrid)), dtype=complex)
    eigval, eigvec = np.linalg.eigh(hamilt)
    for i in range(len(hamilt)):
        eigval_out[i], eigvec_out[i] = eigval[i], sign_adjust(eigvec[i], eigvec_previous[i])
    return eigval_out, eigvec_out

def boltz_grid(t, egrid):
    z = np.sum(np.exp(-1.0 * (1.0 / (kB * t)) * egrid), axis=1)
    return np.exp(-1.0 * (1.0 / (kB * t)) * egrid) / (z.reshape((-1, 1)))

########## Real-Space Functions #########
if space == 'r-space':
    if model == 'holstein':
        @jit(nopython=True)
        def q_mat():
            e1 = E * np.diag(np.zeros(npoints))
            for n in range(npoints):
                e1[cycle(n + 1), cycle(n)] += J
                e1[cycle(n), cycle(n + 1)] += J
            return e1


        def qc_mat_old(p, q):
            dim = len(p)
            wgrid1 = np.zeros((int(dim / npoints), npoints))
            wgrid1[:] = wgrid
            wgrid1 = wgrid1.reshape(-1)
            qc_mat_gen = np.zeros((int(dim / npoints), npoints, npoints))
            diag = (gc * wgrid1 * np.sqrt(2 * m * wgrid1) * q)
            np.einsum('...jj->...j', qc_mat_gen)[...] = diag.reshape(int(dim / npoints), npoints)
            return qc_mat_gen  # .reshape((-npoints,npoints))


        @jit(nopython=True)
        def qc_mat(p, q):
            out_mat = np.asfortranarray(np.zeros((npoints, npoints, npoints)))
            for n in range(npoints):
                out_mat[n] = np.diag(gc * wgrid * np.sqrt(2 * m * wgrid) * q[n])
            return out_mat


        def gen_mat(p, q):
            return q_mat() + qc_mat(p, q)


        @jit(nopython=True)
        def quantumForce(coeffgrid):
            return np.real(np.conj(coeffgrid) * gc * wgrid * np.sqrt(2 * m * wgrid) * coeffgrid), np.real(0 * coeffgrid)

    if model == 'peierls':
        @jit(nopython=True)
        def q_mat():
            e1 = E * np.diag(np.ones(npoints))
            for n in range(npoints):
                e1[cycle(n + 1), cycle(n)] += J
                e1[cycle(n), cycle(n + 1)] += J
            return e1


        @jit(nopython=True)
        def qc_mat_gen(p, q):
            mat = np.zeros((npoints, npoints))
            for n in range(npoints):
                mat[cycle(n + 1), cycle(n)] += gc * wgrid[n] * np.sqrt(2) * (
                            np.sqrt(wgrid[cycle(n)]) * q[cycle(n)] - np.sqrt(wgrid[cycle(n + 1)]) * q[cycle(n + 1)])
                mat[cycle(n), cycle(n + 1)] += gc * wgrid[n] * np.sqrt(2) * (
                            np.sqrt(wgrid[cycle(n)]) * q[cycle(n)] - np.sqrt(wgrid[cycle(n + 1)]) * q[cycle(n + 1)])
            return mat


        @jit(nopython=True)
        def qc_mat(p, q):
            out_mat = np.asfortranarray(np.zeros((npoints, npoints, npoints)))
            for n in range(npoints):
                out_mat[n] = qc_mat_gen(p[n], q[n])
            return out_mat




        @jit(nopython=True)
        def quantumForce(coeffgrid):
            Fq = np.zeros_like(coeffgrid)
            Fp = np.zeros_like(coeffgrid)
            for n in range(npoints):
                Fq[n] += gc * wgrid[n] * np.sqrt(2 * wgrid[n]) * 2 * (
                            np.real(np.conj(coeffgrid[cycle(n + 1)]) * coeffgrid[cycle(n)]) - np.real(
                        np.conj(coeffgrid[cycle(n - 1)]) * coeffgrid[cycle(n)]))
                Fp[n] += 0
            return np.real(Fq), np.real(Fp)

########### K-Space Functions ###########
if space == 'k-space':
    if model == 'holstein_impurity':

        @jit(nopython=True)
        def q_mat_site():
            e1 = E * np.diag(np.zeros(npoints))
            for n in range(npoints):
                e1[cycle(n + 1), cycle(n)] += J
                e1[cycle(n), cycle(n + 1)] += J
            return e1


        #@jit(nopython=True)
        def q_mat():
            H_q_site = q_mat_site()  # full real-space matrix
            H_q_site[impurity_site, impurity_site] += impurity_energy
            H_q_k_trunc = np.matmul(np.conjugate(np.transpose(F_nk_trunc)), np.matmul(H_q_site, F_nk_trunc))

            return H_q_k_trunc


        @jit(nopython=True)
        def quantumForce(coeffgrid):
            cg = np.ascontiguousarray(np.zeros(npoints) + 0.0j)
            cg[tnkgrid] = coeffgrid
            fq = np.ascontiguousarray(np.zeros(npoints))
            fp = np.ascontiguousarray(np.zeros(npoints))
            coeffgrid = np.ascontiguousarray(coeffgrid)
            for kappa in nkgrid:
                cgroll_pk = np.ascontiguousarray(np.conj(np.concatenate((cg[-kappa:], cg[:-kappa]))))
                fq[cycle(int(npoints / 2) + kappa)] = gc * (np.sqrt(2 * w ** 3) / np.sqrt(npoints)) * np.real(
                    np.dot(cgroll_pk, cg))
                fp[cycle(int(npoints / 2) + kappa)] = -gc * (np.sqrt(2 * w) / np.sqrt(npoints)) * np.imag(
                    np.dot(cgroll_pk, cg))
            return fq[tnkgrid], fp[tnkgrid]


        @jit(nopython=True)
        def qc_mat_gen(p, q):
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


    if model == 'holstein':
        @jit(nopython=True)
        def quantumForce(coeffgrid):
            cg = np.ascontiguousarray(np.zeros(npoints) + 0.0j)
            cg[tnkgrid] = coeffgrid
            fq = np.ascontiguousarray(np.zeros(npoints))
            fp = np.ascontiguousarray(np.zeros(npoints))
            coeffgrid = np.ascontiguousarray(coeffgrid)
            for kappa in nkgrid:
                cgroll_pk = np.ascontiguousarray(np.conj(np.concatenate((cg[-kappa:], cg[:-kappa]))))
                fq[cycle(int(npoints / 2) + kappa)] = gc * (np.sqrt(2 * w ** 3) / np.sqrt(npoints)) * np.real(
                    np.dot(cgroll_pk, cg))
                fp[cycle(int(npoints / 2) + kappa)] = -gc * (np.sqrt(2 * w) / np.sqrt(npoints)) * np.imag(
                    np.dot(cgroll_pk, cg))
            return fq[tnkgrid], fp[tnkgrid]


        @jit(nopython=True)
        def q_mat():
            return np.diag(egridQ)


        @jit(nopython=True)
        def qc_mat_gen(p, q):
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


    if model == 'peierls':
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
        def quantumForce(coeffgrid):
            cg = np.ascontiguousarray(np.zeros(npoints) + 0.0j)
            cg[tnkgrid] = coeffgrid
            fq = np.ascontiguousarray(np.zeros(npoints))
            fp = np.ascontiguousarray(np.zeros(npoints))
            coeffgrid = np.ascontiguousarray(coeffgrid)
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
        def q_mat():
            return np.diag(egridQ)


        @jit(nopython=True)
        def qc_mat_gen(p, q):
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



@ray.remote
def runSim(index, p0, q0):
    npoints = len(tnkgrid)
    start_time = time.time()  # initialize starting time for trajectory

    ## initialize classical coordinates ##
    p = p0
    q = q0

    # initialize Hamiltonian
    qmat = q_mat()  # quantum hamiltonian is time independent
    qcmat_store = qc_mat(p, q)
    eigval, eigvec = np.linalg.eigh(qmat + qcmat_store)

    # initialize wavefunctions and density matrices
    den_mat_db = np.zeros((npoints, npoints, npoints), dtype=complex)  # initialize density matrix
    cg_db_0 = coeffgrid # initial diabatic wavefunction
    cg_db = np.zeros((npoints, npoints), dtype=complex)
    cg_db[:] = cg_db_0 # initial diabatic wavefunction in all branches

    ## initialize arrays to store outputs
    hop_count = 0  # storing hop numbers
    tdat = np.arange(0, tmax + dt, dt)  # time array
    tdat_bath = np.arange(0, tmax + dt_bath, dt_bath)  # time array for bath
    den_mat_db_tot = np.zeros((len(tdat), npoints, npoints, npoints), dtype=complex) # diabatic density matrix in each branch
    b_pop_db_tot = np.zeros((len(tdat), npoints, npoints, npoints), dtype=complex) # diabatic boltzmann population matrix
    b_pop_adb_tot = np.zeros((len(tdat), npoints, npoints, npoints), dtype=complex) # adiabatic boltzmann population matrix
    eq_out = np.zeros((len(tdat))) # quantum energy
    ec_out = np.zeros((len(tdat))) # classical energy


    # intialize quantum forces
    fq = np.zeros((npoints, npoints))  # initialize the force on the position coordinates (momentum derivative)
    fp = np.zeros((npoints, npoints))  # initialize the force on the momentum coordinates (position derivative)
    for i in range(npoints):
        if space == 'r-space':
            fq0, fp0 = quantumForce(cg_db[i])
            fq[i] = fq0
            fp[i] = fp0

        if space == 'k-space':
            fq0, fp0 = quantumForce(cg_db[i])
            fq[i] = fq0
            fp[i] = fp0
    t_ind = 0
    for t_bath_ind in range(len(tdat_bath)):
        if tdat[t_ind] <= tdat_bath[t_bath_ind] + 0.5 * dt_bath or t_bath_ind == len(tdat_bath) - 1:
            for i in range(npoints):
                den_mat_db[i] = np.dot(np.conj(cg_db[i]).reshape((-1, 1)), np.array([cg_db[i]]))

            den_mat_db_tot[t_ind, :, :, :] = den_mat_db_tot[t_ind, :, :, :] + den_mat_db
            for i in range(npoints):
                ec_out[t_ind] += np.sum((1 / 2) * p[i] ** 2 + (1 / 2) * (w ** 2) * q[i] ** 2)
                eq_out[t_ind] += np.real(np.matmul(np.conjugate(cg_db[i]),np.matmul(qmat + qcmat_store[i], cg_db[i])))

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
        for i in range(npoints):
            if space == 'r-space':
                fq0, fp0 = quantumForce(cg_db[i])
                fq[i] = fq0
                fp[i] = fp0

            if space == 'k-space':
                fq0, fp0 = quantumForce(cg_db[i])
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
    print('trial index: ', index, ' time: ',end_time - start_time)
    return simCdb, simFcdb, simT, simFcpopB, simPopB, eq_out, ec_out


def parallel_run_ray(nt, proc):
    r_ind = 0
    for run in range(0, int(nt / proc)):
        p, q = init_classical_parallel_branch(proc)
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

    if path.exists(filename + '_resCdb.csv') & io:
        simCdbdat += np.loadtxt(filename + '_resCdb.csv', delimiter=",")
    if path.exists(filename + '_resFcdb.csv') & io:
        simFcdbdat += np.loadtxt(filename + '_resFcdb.csv', delimiter=",")
    if path.exists(filename + '_resT.csv') & io:
        simTdat += np.loadtxt(filename + '_resT.csv', delimiter=",")
    if path.exists(filename + '_resFcpopB.csv') & io:
        simFcpopBdat += np.loadtxt(filename + '_resFcpopB.csv', delimiter=",")
    if path.exists(filename + '_resPopB.csv') & io:
        simPopBdat += np.loadtxt(filename + '_resPopB.csv', delimiter=",")
    if path.exists(filename + '_resEq.csv') & io:
        simEqdat += np.loadtxt(filename + '_resEq.csv', delimiter=",")
    if path.exists(filename + '_resEc.csv') & io:
        simEqdat += np.loadtxt(filename + '_resEc.csv', delimiter=",")
    return simCdbdat, simFcdbdat, simTdat, simFcpopBdat, simPopBdat, simEqdat, simEcdat


print('Starting Calculation FSSH\n')
print(filename, '\n')
start_time = time.time()
resCdb, resFcdb, resT, resFcpopB, resPopB, resEq, resEc = parallel_run_ray(ntrials, nprocs)

if io:
    np.savetxt(filename + '_resCdb.csv', resCdb, delimiter=",")
    np.savetxt(filename + '_resFcdb.csv', resFcdb, delimiter=",")
    np.savetxt(filename + '_resT.csv', resT, delimiter=",")
    np.savetxt(filename + '_resFcpopB.csv', resFcpopB, delimiter=",")
    np.savetxt(filename + '_resPopB.csv', resPopB, delimiter=",")
    np.savetxt(filename + '_resEq.csv', resEq, delimiter=",")
    np.savetxt(filename + '_resEc.csv', resEc, delimiter=",")
end_time = time.time()
if io:
    np.savetxt(foldername + '/wgrid.csv', wgrid, delimiter=",")
    np.savetxt(foldername + '/wgridQ.csv', wgridQ, delimiter=",")
    if not (path.exists(foldername + '/' + 'output_file')):
        with open(inputfile) as f:
            f_out = open(foldername + '/' + 'output_file', "a")
            for line in f:
                f_out.write(line)
print('Calculation Duration: ', end_time - start_time, '\n')
print(filename)
from __future__ import print_function
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import sys
import types

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations
import copy
from textwrap import dedent

from .stencil import Stencil
from .generator import *
from .validate_dictionary import *

from .logs import setLogger
import mpi4py.MPI as mpi


proto_sch = {
    'velocities': (is_list_int,),
    'conserved_moments': (sp.Symbol, bytes, is_list_symb),
    'polynomials': (is_list_sp_or_nb,),
    'equilibrium': (is_list_sp_or_nb,),
    'relaxation_parameters': (is_list_sp_or_nb,),
    'init':(type(None), is_dico_init),
}

proto_sch_dom = {
    'velocities': (is_list_int,),
    'conserved_moments': (type(None), sp.Symbol, bytes, is_list_symb),
    'polynomials': (type(None), is_list_sp_or_nb,),
    'equilibrium': (type(None), is_list_sp_or_nb,),
    'relaxation_parameters': (type(None), is_list_sp_or_nb,),
    'init':(type(None), is_dico_init),
}

proto_stab = {
    'linearization':(type(None), is_dico_sp_float),
    'test_maximum_principle':(type(None), bool),
    'test_L2_stability':(type(None), bool),
}

proto_cons = {
    'order': (int,),
    'linearization':(type(None), is_dico_sp_sporfloat),
}

def param_to_tuple(param):
    if param is not None:
        pk, pv = param.keys(), param.values()
    else:
        pk, pv = (), ()
    return pk, pv

class Scheme:
    """
    Create the class with all the needed informations for each elementary scheme.

    Parameters
    ----------

    dico : a dictionary that contains the following `key:value`
      - dim : spatial dimension (optional if the `box` is given)
      - scheme_velocity : the value of the ratio space step over time step
        (la = dx / dt)
      - schemes : a list of dictionaries, one for each scheme
      - generator : a generator for the code, optional
        (see :py:class:`Generator <pyLBM.generator.Generator>`)
      - test_stability : boolean (optional)

    Notes
    -----

    Each dictionary of the list `schemes` should contains the following `key:value`

    - velocities : list of the velocities number
    - conserved moments : list of the moments conserved by each scheme
    - polynomials : list of the polynomial functions that define the moments
    - equilibrium : list of the values that define the equilibrium
    - relaxation_parameters : list of the value of the relaxation parameters
    - init : a dictionary to define the initial conditions (see examples)

    If the stencil has already been computed, it can be pass in argument.

    Attributes
    ----------

    dim : int
      spatial dimension
    la : double
      scheme velocity, ratio dx/dt
    nscheme : int
      number of elementary schemes
    stencil : object of class :py:class:`Stencil <pyLBM.stencil.Stencil>`
      a stencil of velocities
    P : list of sympy matrix
      list of polynomials that define the moments
    EQ : list of sympy matrix
      list of the equilibrium functions
    s  : list of list of doubles
      relaxation parameters
      (exemple: s[k][l] is the parameter associated to the lth moment in the kth scheme)
    M : sympy matrix
      the symbolic matrix of the moments
    Mnum : numpy array
      the numeric matrix of the moments (m = Mnum F)
    invM : sympy matrix
      the symbolic inverse matrix
    invMnum : numpy array
      the numeric inverse matrix (F = invMnum m)
    generator : :py:class:`Generator <pyLBM.generator.Generator>`
      the used generator (
      :py:class:`NumpyGenerator<pyLBM.generator.NumpyGenerator>`,
      :py:class:`CythonGenerator<pyLBM.generator.CythonGenerator>`,
      ...)

    Methods
    -------

    create_moments_matrix :
      Create the moments matrices
    create_relaxation_function :
      Create the relaxation function
    create_equilibrium_function :
      Create the equilibrium function
    create_transport_function :
      Create the transport function
    create_f2m_function :
      Create the function f2m
    create_m2f_function :
      Create the function m2f

    generate :
      Generate the code
    equilibrium :
      Compute the equilibrium
    transport :
      Transport phase
    relaxation :
      Relaxation phase
    f2m :
      Compute the moments from the distribution functions
    m2f :
      Compute the distribution functions from the moments
    onetimestep :
      One time step of the Lattice Boltzmann method
    set_boundary_conditions :
      Apply the boundary conditions

    Examples
    --------

    see demo/examples/scheme/

    """
    def __init__(self, dico, stencil=None):
        self.log = setLogger(__name__)
        # symbolic parameters
        self.param = dico.get('parameters', None)
        pk, pv = param_to_tuple(self.param)

        if stencil is not None:
            self.stencil = stencil
        else:
            self.stencil = Stencil(dico)
        self.dim = self.stencil.dim
        la = dico.get('scheme_velocity', None)
        if isinstance(la, (int, float)):
            self.la = la
            self.la_symb = None
        elif isinstance(la, sp.Symbol):
            self.la_symb = la
            self.la = sp.N(la.subs(zip(pk, pv)))
        else:
            self.log.error("The entry 'scheme_velocity' is wrong.")
        self.nscheme = self.stencil.nstencils
        scheme = dico['schemes']
        if not isinstance(scheme, list):
            self.log.error("The entry 'schemes' must be a list.")

        def create_matrix(L):
            """
            convert a list of strings to a sympy Matrix.
            """
            def auto_moments(tokens, local_dict, global_dict):
                """
                if the user uses a string to describe the moments like
                'm[0][0]', this function converts it as Symbol('m[0][0]').
                This fix the problem of auto_symbol that doesn't support
                indexing.
                """
                result = []
                i = 0
                while(i < len(tokens)):
                    tokNum, tokVal = tokens[i]
                    if tokVal == 'm':
                        name = ''.join([val for n, val in tokens[i:i+7]])
                        result.extend([(1, 'Symbol'),
                                       (51, '('),
                                       (3, "'{0}'".format(name)),
                                       (51, ')')])
                        i += 7
                    else:
                        result.append(tokens[i])
                        i += 1
                return result
            res = []
            for l in L:
                if isinstance(l, str):
                    res.append(parse_expr(l, transformations=(auto_moments,) + standard_transformations))
                else:
                    res.append(l)
            return sp.Matrix(res)


        self._check_entry_size(scheme, 'polynomials')
        self._check_entry_size(scheme, 'equilibrium')
        self.P = [create_matrix(s['polynomials']) for s in scheme]
        self.EQ = [create_matrix(s['equilibrium']) for s in scheme]

        self.consm = self._get_conserved_moments(scheme)
        self.ind_cons, self.ind_noncons = self._get_indices_cons_noncons()

        # rename conserved moments with the notation m[x][y]
        # needed to generate the code
        # where x is the number of the scheme
        #       y is the index in equilibrium corresponding to the conserved moment
        self._EQ = copy.deepcopy(self.EQ)

        m = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(self.stencil.unvtot)] for i in xrange(len(self.EQ))]
        for cm, icm in self.consm.iteritems():
            for i, eq in enumerate(self._EQ):
                for j, e in enumerate(eq):
                    self._EQ[i][j] = e.replace(cm, m[icm[0]][icm[1]])

        self._check_entry_size(scheme, 'relaxation_parameters')
        self.s_symb = [s['relaxation_parameters'] for s in scheme]
        self.s = [copy.deepcopy(s['relaxation_parameters']) for s in scheme]
        for k in xrange(len(self.s)):
            for l in xrange(len(self.s[k])):
                if not isinstance(self.s[k][l], (int, float)):
                    try:
                        self.s[k][l] = sp.N(self.s[k][l].subs(zip(pk, pv)))
                    except:
                        self.log.error('cannot evaluate relaxation parameter')

        self.init = self.set_initialization(scheme)

        self.M, self.invM = [], []
        self.Mnum, self.invMnum = [], []

        self.create_moments_matrices()

        # generate the code
        self.generator = dico.get('generator', NumpyGenerator)()
        self.log.info("Generator used for the scheme functions:\n{0}\n".format(self.generator))

        self.bc_compute = True

        # stability
        dicostab = dico.get('stability', None)
        if dicostab is not None:
            dico_linearization = dicostab.get('linearization', None)
            if dico_linearization is not None:
                self.list_linearization = []
                for cm, cv in dico_linearization.iteritems():
                    icm = self.consm[cm]
                    self.list_linearization.append((m[icm[0]][icm[1]], cv))
            else:
                self.list_linearization = None
            self.compute_amplification_matrix_relaxation()
            Li_stab = dicostab.get('test_monotonic_stability', False)
            if Li_stab:
                if self.is_monotonically_stable():
                    print("The scheme is monotonically stable")
                else:
                    print("The scheme is not monotonically stable")
            L2_stab = dicostab.get('test_L2_stability', False)
            if L2_stab:
                if self.is_L2_stable():
                    print("The scheme is stable for the norm L2")
                else:
                    print("The scheme is not stable for the norm L2")

        dicocons = dico.get('consistency', None)
        if dicocons is not None:
            self.compute_consistency(dicocons)


    def _check_entry_size(self, schemes, key):
        for i, s in enumerate(schemes):
            ls = len(s[key])
            nv = self.stencil.nv[i]
            if ls != nv:
                self.log.error(dedent("""\
                               the size of the entry for the key {0} in the scheme {1}
                               has not the same size of the stencil {1}: {2}, {3}""".format(key, i, ls, nv)))

    def __str__(self):
        s = "Scheme informations\n"
        s += "\t spatial dimension: dim={0:d}\n".format(self.dim)
        s += "\t number of schemes: nscheme={0:d}\n".format(self.nscheme)
        s += "\t number of velocities:\n"
        for k in xrange(self.nscheme):
            s += "    Stencil.nv[{0:d}]=".format(k) + str(self.stencil.nv[k]) + "\n"
        s += "\t velocities value:\n"
        for k in xrange(self.nscheme):
            s+="    v[{0:d}]=".format(k)
            for v in self.stencil.v[k]:
                s += v.__str__() + ', '
            s += '\n'
        s += "\t polynomials:\n"
        for k in xrange(self.nscheme):
            s += "    P[{0:d}]=".format(k) + self.P[k].__str__() + "\n"
        s += "\t equilibria:\n"
        for k in xrange(self.nscheme):
            s += "    EQ[{0:d}]=".format(k) + self.EQ[k].__str__() + "\n"
        s += "\t relaxation parameters:\n"
        for k in xrange(self.nscheme):
            s += "    s[{0:d}]=".format(k) + self.s[k].__str__() + "\n"
        s += "\t moments matrices\n"
        s += "M = " + self.M.__str__() + "\n"
        s += "invM = " + self.invM.__str__() + "\n"
        return s

    def create_moments_matrices(self):
        """
        Create the moments matrices M and M^{-1} used to transform the repartition functions into the moments
        """
        compt=0
        for v, p in zip(self.stencil.v, self.P):
            compt+=1
            lv = len(v)
            self.M.append(sp.zeros(lv, lv))
            for i in xrange(lv):
                for j in xrange(lv):
                    self.M[-1][i, j] = p[i].subs([('X', v[j].vx), ('Y', v[j].vy), ('Z', v[j].vz)])
            try:
                self.invM.append(self.M[-1].inv())
            except:
                s = 'Function create_moments_matrices: M is not invertible\n'
                s += 'The choice of polynomials is odd in the elementary scheme number {0:d}'.format(compt)
                self.log.error(s)
                sys.exit()

        self.MnumGlob = np.zeros((self.stencil.nv_ptr[-1], self.stencil.nv_ptr[-1]))
        self.invMnumGlob = np.zeros((self.stencil.nv_ptr[-1], self.stencil.nv_ptr[-1]))

        pk, pv = param_to_tuple(self.param)

        try:
            for k in xrange(self.nscheme):
                nvk = self.stencil.nv[k]
                self.Mnum.append(np.empty((nvk, nvk)))
                self.invMnum.append(np.empty((nvk, nvk)))
                for i in xrange(nvk):
                    for j in xrange(nvk):
                        self.Mnum[k][i, j] = sp.N(self.M[k][i, j].subs(zip(pk, pv)))
                        self.invMnum[k][i, j] = sp.N(self.invM[k][i, j].subs(zip(pk, pv)))
                        self.MnumGlob[self.stencil.nv_ptr[k] + i, self.stencil.nv_ptr[k] + j] = self.Mnum[k][i, j]
                        self.invMnumGlob[self.stencil.nv_ptr[k] + i, self.stencil.nv_ptr[k] + j] = self.invMnum[k][i, j]
        except TypeError:
            self.log.error("Unable to convert to float the expression {0} or {1}.\nCheck the 'parameters' entry.".format(self.M[k][i, j], self.invM[k][i, j]))
            sys.exit()

    def _get_conserved_moments(self, scheme):
        """
        return conserved moments and their indices in the scheme entry.

        Parameters
        ----------

        scheme : dictionnary that describes the LBM schemes

        Output
        ------

        consm : dictionnary where the keys are the conserved moments and
                the values their indices in the LBM schemes.
        """
        consm_tmp = [s.get('conserved_moments', None) for s in scheme]
        consm = {}

        def find_indices(ieq, list_eq, c):
            if [c] in leq:
                ic = (ieq, leq.index([c]))
                if isinstance(c, str):
                    cm = parse_expr(c)
                else:
                    cm = c
                return ic, cm

        # find the indices of the conserved moments in the equilibrium equations
        for ieq, eq in enumerate(self.EQ):
            leq = eq.tolist()
            cm_ieq = consm_tmp[ieq]
            if cm_ieq is not None:
                if isinstance(cm_ieq, sp.Symbol):
                    ic, cm = find_indices(ieq, leq, cm_ieq)
                    consm[cm] = ic
                else:
                    for c in cm_ieq:
                        ic, cm = find_indices(ieq, leq, c)
                        consm[cm] = ic
        return consm

    def _get_indices_cons_noncons(self):
        """
        return the list of the conserved moments and the list of the non conserved moments

        Output
        ------

        l_cons : the list of the indices of the conserved moments
        l_noncons : the list of the indices of the non conserver moments
        """

        ns = self.stencil.nstencils # number of stencil
        nv = self.stencil.nv # number of velocities for each stencil
        l_cons = [[] for n in nv]
        l_noncons = [range(n) for n in nv]
        for vk in self.consm.values():
            l_cons[vk[0]].append(vk[1])
            l_noncons[vk[0]].remove(vk[1])
        for n in xrange(ns):
            l_cons[n].sort()
            l_noncons[n].sort()
        return l_cons, l_noncons


    def set_initialization(self, scheme):
        """
        set the initialization functions for the conserved moments.

        Parameters
        ----------

        scheme : dictionnary that describes the LBM schemes

        Output
        ------

        init : dictionnary where the keys are the indices of the
               conserved moments and the values must be

               a constant (int or float)
               a tuple of size 2 that describes a function and its
               extra args

        """
        init = {}
        for ns, s in enumerate(scheme):
            init_scheme = s.get('init', None)
            if init_scheme is None:
                self.log.warning("You don't define initialization step for your conserved moments")
                continue
            for k, v in s['init'].iteritems():

                try:
                    if isinstance(k, str):
                        indices = self.consm[parse_expr(k)]
                    elif isinstance(k, sp.Symbol):
                        indices = self.consm[k]
                    elif isinstance(k, int):
                        indices = (ns, k)
                    else:
                        raise ValueError

                    init[indices] = v

                except ValueError:
                    sss = 'Error in the creation of the scheme: wrong dictionnary\n'
                    sss += 'the key `init` should contain a dictionnary with'
                    sss += '   key: the moment to init'
                    sss += '        should be the name of the moment as a string or'
                    sss += '        a sympy Symbol or an integer'
                    sss += '   value: the initial value'
                    sss += '        should be a constant, a tuple with a function'
                    sss += '        and extra args or a lambda function'
                    self.log.error(sss)
                    sys.exit()
        return init

    def generate(self, sorder):
        """
        Generate the code by using the appropriated generator

        Notes
        -----

        The code can be viewed. If S is the scheme

        >>> print S.generator.code
        """
        self.generator.sorder = sorder
        self.generator.setup()
        self.generator.m2f(self.invMnumGlob, 0, self.dim)
        self.generator.f2m(self.MnumGlob, 0, self.dim)
        self.generator.onetimestep(self.stencil)

        self.generator.transport(self.nscheme, self.stencil)

        pk, pv = param_to_tuple(self.param)
        EQ = []
        for e in self._EQ:
            EQ.append(e.subs(zip(pk, pv)))

        self.generator.equilibrium(self.nscheme, self.stencil, EQ)
        self.generator.relaxation(self.nscheme, self.stencil, self.s, EQ)
        self.generator.compile()

    def m2f(self, m, f):
        """ Compute the distribution functions f from the moments m """
        mod = self.generator.get_module()
        mod.m2f(m.array, f.array)

    def f2m(self, f, m):
        """ Compute the moments m from the distribution functions f """
        mod = self.generator.get_module()
        mod.f2m(f.array, m.array)

    def transport(self, f):
        """ The transport phase on the distribution functions f """
        mod = self.generator.get_module()
        mod.transport(f)

    def equilibrium(self, m):
        """ Compute the equilibrium """
        mod = self.generator.get_module()
        func = getattr(mod, "equilibrium")
        func(m.array)

    def relaxation(self, m):
        """ The relaxation phase on the moments m """
        mod = self.generator.get_module()
        mod.relaxation(m.array)

    def onetimestep(self, m, fold, fnew, in_or_out, valin):
        """ Compute one time step of the Lattice Boltzmann method """
        mod = self.generator.get_module()
        mod.onetimestep(m.array, fold.array, fnew.array, in_or_out, valin)

    def set_boundary_conditions(self, f, m, bc, interface):
        """
        Compute the boundary conditions

        Parameters
        ----------

        f : numpy array
          the array of the distribution functions
        m : numpy array
          the array of the moments
        bc : :py:class:`pyLBM.boundary.Boundary`
          the class that contains all the informations needed
          for the boundary conditions

        Returns
        -------

        Modify the array of the distribution functions f in the phantom border area
        according to the labels. In the direction parallel to the bounday, N denotes
        the number of inner points, phantom cells are added to take into account
        the boundary conditions.

        Notes
        -----

        If n is the number of outer cells on each bound and N the number of inner cells,
        the following representation could be usefull (Na = N+2*n)

         +---------------+----------------+-----------------+
         | n outer cells | N inner cells  | n outer cells   |
         +===============+================+=================+
         |               | 0 ...  N-1     |                 |
         +---------------+----------------+-----------------+
         | 0  ...  n-1   | n ... N+n-1    | N+n  ... Na-1   |
         +---------------+----------------+-----------------+

        """
        f.update()

        for method in bc.methods:
            method.update(f)

    def compute_amplification_matrix_relaxation(self):
        ns = self.stencil.nstencils # number of stencil
        nv = self.stencil.nv # number of velocities for each stencil
        nvtot = sum(nv)
        # matrix of the f2m and m2f transformations
        M = np.zeros((nvtot, nvtot))
        iM = np.zeros((nvtot, nvtot))
        # matrix of the relaxation parameters
        R = np.zeros((nvtot, nvtot))
        # matrix of the equilibrium
        E = np.zeros((nvtot, nvtot))
        k = 0
        for n in range(ns):
            l = nv[n]
            M[k:k+l, k:k+l] = self.Mnum[n]
            iM[k:k+l, k:k+l] = self.invMnum[n]
            R[k:k+l, k:k+l] = np.diag(self.s[n])
            k += l
        k = 0

        pk, pv = param_to_tuple(self.param)
        m = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(self.stencil.unvtot)] for i in xrange(len(self.EQ))]
        for n in range(ns):
            for i in range(nv[n]):
                eqi = self._EQ[n][i].subs(zip(pk, pv))
                if str(eqi) != "u[%d][%d]"%(n, i):
                    l = 0
                    for mm in range(ns):
                        for j in range(nv[mm]):
                            dummy = sp.diff(eqi, m[mm][j])
                            if self.list_linearization is not None:
                                dummy = dummy.subs(self.list_linearization)
                            E[k+i, l+j] = dummy
                        l += nv[mm]
            k += nv[n]
        C = np.dot(R, E - np.eye(nvtot))
        # global amplification matrix for the relaxation
        self.amplification_matrix_relaxation = np.eye(nvtot) + np.dot(iM, np.dot(C, M))

    def amplification_matrix(self, wave_vector):
        Jr = self.amplification_matrix_relaxation
        # matrix of the transport phase
        q = Jr.shape[0]
        J = np.zeros((q, q), dtype='complex128')
        k = 0
        for n in range(self.stencil.nstencils):
            for i in range(self.stencil.nv[n]):
                vi = [self.stencil.vx[n][i],
                      self.stencil.vy[n][i],
                      self.stencil.vz[n][i]]
                J[k+i, :] = np.exp(1j*sum([a*b for a, b in zip(wave_vector, vi)])) * Jr[k+i, :]
            k += self.stencil.nv[n]
        return J

    def vp_amplification_matrix(self, wave_vector):
        vp = np.linalg.eig(self.amplification_matrix(wave_vector))
        return vp[0]

    def is_L2_stable(self, Nk = 101):
        R = 1.
        vk = np.linspace(0., 2*np.pi, Nk)
        if self.dim == 1:
            for i in range(vk.size):
                kx = vk[i]
                vp = self.vp_amplification_matrix((kx, ))
                rloc = max(abs(vp))
                if rloc > R+1.e-14:
                    return False
        elif self.dim == 2:
            for i in range(vk.size):
                kx = vk[i]
                for j in range(vk.size):
                    ky = vk[j]
                    vp = self.vp_amplification_matrix((kx, ky))
                    rloc = max(abs(vp))
                    if rloc > R+1.e-14:
                        return False
        elif self.dim == 3:
            for i in range(vk.size):
                kx = vk[i]
                for j in range(vk.size):
                    ky = vk[j]
                    for k in range(vk.size):
                        kz = vk[k]
                        vp = self.vp_amplification_matrix((kx, ky, kz))
                        rloc = max(abs(vp))
                        if rloc > R+1.e-14:
                            return False
        else:
            self.log.warning("dim should be in [1, 3] for the scheme")
        return True

    def is_monotonically_stable(self):
        if np.min(self.amplification_matrix_relaxation) < 0:
            return False
        else:
            return True

    def compute_consistency(self, dicocons):
        t0 = mpi.Wtime()
        ns = self.stencil.nstencils # number of stencil
        nv = self.stencil.nv # number of velocities for each stencil
        nvtot = self.stencil.nv_ptr[-1] # total number of velocities (with repetition)
        N = len(self.consm) # number of conserved moments
        time_step = sp.Symbol('h')
        drondt = sp.Symbol("dt") # time derivative
        drondx = [sp.Symbol("dx"), sp.Symbol("dy"), sp.Symbol("dz")] # spatial derivatives
        if self.la_symb is not None:
            LA = self.la_symb
        else:
            LA = self.la

        m = [[sp.Symbol("m[%d][%d]"%(i,j)) for j in xrange(nvtot)] for i in xrange(ns)]
        order = dicocons['order']
        if order<1:
            order = 1
        dico_linearization = dicocons.get('linearization', None)
        if dico_linearization is not None:
            self.list_linearization = []
            for cm, cv in dico_linearization.iteritems():
                icm = self.consm[cm]
                self.list_linearization.append((m[icm[0]][icm[1]], cv))
        else:
            self.list_linearization = None

        M = sp.zeros(nvtot, nvtot)
        invM = sp.zeros(nvtot, nvtot)
        il = 0
        for n in xrange(ns):
            for k in self.ind_cons[n]:
                M[il,self.stencil.nv_ptr[n]:self.stencil.nv_ptr[n+1]] = self.M[n][k,:]
                invM[self.stencil.nv_ptr[n]:self.stencil.nv_ptr[n+1],il] = self.invM[n][:,k]
                il += 1
        for n in xrange(ns):
            for k in self.ind_noncons[n]:
                M[il,self.stencil.nv_ptr[n]:self.stencil.nv_ptr[n+1]] = self.M[n][k,:]
                invM[self.stencil.nv_ptr[n]:self.stencil.nv_ptr[n+1],il] = self.invM[n][:,k]
                il += 1
        v = self.stencil.get_all_velocities().transpose()

        # build the matrix of equilibrium
        Eeq = sp.zeros(nvtot, nvtot)
        il = 0
        for n_i in xrange(ns):
            for k_i in self.ind_cons[n_i]:
                Eeq[il, il] = 1
                ## the equilibrium value of the conserved moments is itself
                #eqk = self._EQ[n_i][k_i]
                #ic = 0
                #for n_j in xrange(len(self.ind_cons)):
                #    for k_j in self.ind_cons[n_j]:
                #        Eeq[il, ic] = sp.diff(eqk, m[n_j][k_j])
                #        ic += 1
                #for n_j in xrange(len(self.ind_noncons)):
                #    for k_j in self.ind_noncons[n_j]:
                #        Eeq[il, ic] = sp.diff(eqk, m[n_j][k_j])
                #        ic += 1
                il += 1
        for n_i in xrange(ns):
            for k_i in self.ind_noncons[n_i]:
                eqk = self._EQ[n_i][k_i]
                ic = 0
                for n_j in xrange(ns):
                    for k_j in self.ind_cons[n_j]:
                        dummy = sp.diff(eqk, m[n_j][k_j])
                        if self.list_linearization is not None:
                            dummy = dummy.subs(self.list_linearization)
                        Eeq[il, ic] = dummy
                        ic += 1
                ## the equilibrium value of the non conserved moments
                ## does not depend on the non conserved moments
                #for n_j in xrange(len(self.ind_noncons)):
                #    for k_j in self.ind_noncons[n_j]:
                #        Eeq[il, ic] = sp.diff(eqk, m[n_j][k_j])
                #        ic += 1
                il += 1

        S = sp.zeros(nvtot, nvtot)
        il = 0
        for n_i in xrange(ns):
            for k_i in self.ind_cons[n_i]:
                S[il, il] = self.s_symb[n_i][k_i]
                il += 1
        for n_i in xrange(ns):
            for k_i in self.ind_noncons[n_i]:
                S[il, il] = self.s_symb[n_i][k_i]
                il += 1

        J = sp.eye(nvtot) - S + S * Eeq

        t1 = mpi.Wtime()
        print("Initialization time: ", t1-t0)

        matA, matB, matC, matD = [], [], [], []
        Dn = sp.zeros(nvtot, nvtot)
        nnn = sp.Symbol('nnn')
        for k in xrange(nvtot):
            Dnk = (- sum([LA * sp.Integer(v[alpha, k]) * drondx[alpha] for alpha in xrange(self.dim)]))**nnn / sp.factorial(nnn)
            Dn[k,k] = Dnk
        dummyn = M * Dn * invM * J
        for n in xrange(order+1):
            dummy = dummyn.subs([(nnn,n)])
            dummy.simplify()
            matA.append(dummy[:N, :N])
            matB.append(dummy[:N, N:])
            matC.append(dummy[N:, :N])
            matD.append(dummy[N:, N:])

        t2 = mpi.Wtime()
        print("Compute A, B, C, D: ", t2-t1)

        iS = S[N:,N:].inv()
        matC[0] = iS * matC[0]
        matC[0].simplify()
        Gamma = []
        for k in xrange(1,order+1):
            for j in xrange(1,k+1):
                matA[k] += matB[j] * matC[k-j]
            Gammak = [matA[k].copy()]
            for j in xrange(k-1):
                Gammakj = sp.zeros(N,N)
                for l in xrange(1,k-j):
                    Gammakj += matA[l] * Gamma[k-l-1][j]
                Gammakj.simplify()
                Gammak.append(Gammakj)
            Gamma.append(Gammak)
            for j in xrange(1, k):
                matA[k] -= Gamma[k-1][j]/sp.factorial(j+1)
            matA[k].simplify()
            for j in xrange(1,k+1):
                matC[k] += matD[j] * matC[k-j]
            for j in xrange(k):
                Kkj = sp.zeros(nvtot-N, N)
                for l in xrange(k-j):
                    Kkj += matC[l] * Gamma[k-l-1][j]
                matC[k] -= Kkj/sp.factorial(j+1)
            matC[k] = iS * matC[k]
            matC[k].simplify()
        t3 = mpi.Wtime()
        print("Compute alpha, beta: ", t3-t2)

        W = sp.zeros(N, 1)
        dummy = [0]
        sp.init_printing()
        for n in xrange(ns):
            dummy.append(dummy[-1] + len(self.ind_cons[n]))
        for wk, ik in self.consm.iteritems():
            W[dummy[ik[0]] + self.ind_cons[ik[0]].index(ik[1]),0] = wk
        self.consistency = {}
        for k in xrange(N):
            wk = W[k,0]
            self.consistency[wk] = {'lhs':[sp.simplify(drondt * W[k,0]), sp.simplify(-(matA[1]*W)[k,0])]}
            lhs = sp.simplify(sum(self.consistency[wk]['lhs']))
            dummy = []
            for n in xrange(1,order):
                dummy.append(sp.simplify(time_step**n * (matA[n+1]*W)[k,0]))
            self.consistency[wk]['rhs'] = dummy
            rhs = sp.simplify(sum(self.consistency[wk]['rhs']))
            print("\n" + "*"*50)
            print("Conservation equation for {0} at order {1}".format(wk, order))
            sp.pprint(lhs)
            print(" "*10, "=")
            sp.pprint(rhs)
            print("*"*50)
        #print self.consistency

        t4 = mpi.Wtime()
        print("Compute equations: ", t4-t3)
        print("Total time: ", t4-t0)


def test_1D(opt):
    dim = 1 # spatial dimension
    la = 1.
    print("\n\nTest number {0:d} in {1:d}D:".format(opt,dim))
    dico = {'dim':dim, 'scheme_velocity':la}
    if (opt == 0):
        dico['number_of_schemes'] = 1 # number of elementary schemes
        dico[0] = {'velocities':[2,0,1],
           'polynomials':Matrix([1,la*X,X**2/2]),
           'equilibrium':Matrix([u[0][0], u[0][1], (0.5*la)**2/2*u[0][0]]),
           'relaxation_parameters':[0,0,1.9]
           }
    elif (opt == 1):
        dico['number_of_schemes'] = 2 # number of elementary schemes
        dico[0] = {'velocities':[2,1],
           'polynomials':Matrix([1,la*X]),
           'equilibrium':Matrix([u[0][0], u[1][0]]),
           'relaxation_parameters':[0,1.5]
           }
        dico[1] = {'velocities':[2,1],
           'polynomials':Matrix([1,la*X]),
           'equilibrium':Matrix([u[1][0], u[0][0]]),
           'relaxation_parameters':[0,1.2]
           }
    elif (opt == 2):
        dico['number_of_schemes'] = 1 # number of elementary schemes
        dico[0] = {'velocities':range(5),
           'polynomials':Matrix([1, la*X, X**2/2, X**3/2, X**4/2]),
           'equilibrium':Matrix([u[0][0], u[0][1], (0.5*la)**2/2*u[0][0], 0, 0]),
           'relaxation_parameters':[0,0,1.9, 1., 1.]
           }
    try:
        LBMscheme = Scheme(dico)
        print(LBMscheme)
        return 1
    except:
        return 0

def test_2D(opt):
    dim = 2 # spatial dimension
    la = 1.
    print("\n\nTest number {0:d} in {1:d}D:".format(opt,dim))
    dico = {'dim':dim, 'scheme_velocity':la}
    if (opt == 0):
        dico['number_of_schemes'] = 2 # number of elementary schemes
        dico[0] = {'velocities':range(1,5),
           'polynomials':Matrix([1, la*X, la*Y, X**2-Y**2]),
           'equilibrium':Matrix([u[0][0], .1*u[0][0], 0, 0]),
           'relaxation_parameters':[0, 1, 1, 1]
           }
        dico[1] = {'velocities':range(5),
           'polynomials':Matrix([1, la*X, la*Y, X**2+Y**2, X**2-Y**2]),
           'equilibrium':Matrix([u[1][0], 0, 0, 0.1*u[1][0], 0]),
           'relaxation_parameters':[0, 1, 1, 1, 1]
           }
    elif (opt == 1):
        rhoo = 1.
        dummy = 1./(la**2*rhoo)
        qx2 = dummy*u[0][1]**2
        qy2 = dummy*u[0][2]**2
        q2  = qx2+qy2
        qxy = dummy*u[0][1]*u[0][2]
        dico['number_of_schemes'] = 1 # number of elementary schemes
        dico[0] = {'velocities':range(9),
           'polynomials':Matrix([1, la*X, la*Y, 3*(X**2+Y**2)-4, (9*(X**2+Y**2)**2-21*(X**2+Y**2)+8)/2, 3*X*(X**2+Y**2)-5*X, 3*Y*(X**2+Y**2)-5*Y, X**2-Y**2, X*Y]),
           'equilibrium':Matrix([u[0][0], u[0][1], u[0][3], -2*u[0][0] + 3*q2, u[0][0]+1.5*q2, u[0][1]/la, u[0][2]/la, qx2-qy2, qxy]),
           'relaxation_parameters':[0, 0, 0, 1, 1, 1, 1, 1, 1]
           }
    try:
        LBMscheme = Scheme(dico)
        print(LBMscheme)
        return 1
    except:
        return 0

if __name__ == "__main__":
    k = 1
    compt = 0
    while (k==1):
        k = test_1D(compt)
        compt += 1
    k = 1
    compt = 0
    while (k==1):
        k = test_2D(compt)
        compt += 1

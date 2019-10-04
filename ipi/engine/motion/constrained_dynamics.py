"""Contains the classes that deal with the different dynamics required in
different types of ensembles.

Holds the algorithms required for normal mode propagators, and the objects to
do the constant temperature and pressure algorithms. Also calculates the
appropriate conserved energy quantity for the ensemble of choice.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import numpy as np
import warnings

from ipi.utils.nmtransform import nm_fft
from ipi.engine.motion import Dynamics
from ipi.engine.motion.dynamics import DummyIntegrator, NVEIntegrator
from ipi.utils.depend import depend_value, depend_array, \
                             dd, dobject, dstrip, dpipe
from ipi.engine.thermostats import Thermostat
from ipi.engine.barostats import Barostat
from ipi.engine.quasicentroids import QuasiCentroids


class ConstrainedDynamics(Dynamics):

    """self (path integral) constrained molecular dynamics class.

    Gives the standard methods and attributes needed in all the
    constrained dynamics classes.

    Attributes:
        beads: A beads object giving the atoms positions.
        cell: A cell object giving the system box.
        forces: A forces object giving the virial and the forces acting on
            each bead.
        prng: A random number generator object.
        nm: An object which does the normal modes transformation.

    Depend objects:
        econs: The conserved energy quantity appropriate to the given
            ensemble. Depends on the various energy terms which make it up,
            which are different depending on the ensemble.he
        temp: The system temperature.
        dt: The timestep for the algorithms.
        ntemp: The simulation temperature. Will be nbeads times higher than
            the system temperature as PIMD calculations are done at this
            effective classical temperature.
    """

    def __init__(self, timestep, mode="nve", splitting="obabo",
                thermostat=None, barostat=None,
                quasicentroids=None, fixcom=False, fixatoms=None,
                nmts=None, nsteps_geo=1, constraint_groups=[]):

        """Initialises a "ConstrainedDynamics" motion object.

        Args:
            dt: The timestep of the simulation algorithms.
            fixcom: An optional boolean which decides whether the centre of mass
                motion will be constrained or not. Defaults to False.
        """

        super(Dynamics, self).__init__(fixcom=fixcom, fixatoms=fixatoms)
        dd(self).dt = depend_value(name='dt', value=timestep)
        if thermostat is None:
            self.thermostat = Thermostat()
        else:
            self.thermostat = thermostat
        if barostat is None:
            self.barostat = Barostat()
        else:
            self.barostat = barostat
        if quasicentroids is None:
            self.quasicentroids = QuasiCentroids()
        else:
            self.quasicentroids = quasicentroids
        self.enstype = mode
        if nmts is None or len(nmts) == 0:
            dd(self).nmts = depend_array(name="nmts", value=np.asarray([1], int))
        else:
            dd(self).nmts = depend_array(name="nmts", value=np.asarray(nmts, int))
        if self.enstype == "nve":
            self.integrator = NVEConstrainedIntegrator()
        elif self.enstype == "nvt":
            self.integrator = NVTConstrainedIntegrator()
        else:
            self.integrator = DummyIntegrator()
        # splitting mode for the integrators
        dd(self).splitting = depend_value(name='splitting', value=splitting)
        # constraints
        self.fixcom = fixcom
        if fixatoms is None:
            self.fixatoms = np.zeros(0, int)
        else:
            self.fixatoms = fixatoms
        self.constraint_groups = constraint_groups
        self.csolver = ConstraintSolver(self.constraint_groups)
        self.nsteps_geo = nsteps_geo

    def bind(self, ens, beads, nm, cell, bforce, prng, omaker):
        """Binds ensemble beads, cell, bforce, and prng to the dynamics.

        This takes a beads object, a cell object, a forcefield object and a
        random number generator object and makes them members of the ensemble.
        It also then creates the objects that will hold the data needed in the
        ensemble algorithms and the dependency network.

        Args:
            ens: The ensemble object specifying the thermodynamic state
                of the system.
            beads: The beads object from whcih the bead positions are taken.
            nm: A normal modes object used to do the normal-mode
                transformation.
            cell: The cell object from which the system box is taken.
            bforce: The forcefield object from which the force and virial are
                taken.
            prng: The random number generator object which controls random number
                generation.
            omaker: output maker
        """
        
        # Bind the constraints first
        for cgp in self.constraint_groups:
            cgp.bind(beads, nm)
        self.csolver.bind(nm)
        # Rest as in dynamics
        super(ConstrainedDynamics, self).bind(ens, beads, nm, cell, 
                                              bforce, prng, omaker)

        
class ConstraintBase(dobject):
    """Base constraint class; defines the constraint function and its Jacobian.
    """
    
    # Add constrained indices and values
    def __init__(self, indices, targetvals=None,
                 tolerance=1.0e-4, domain="cartesian", ngp=0):
        """Initialise the constraint. NOTE: during propagation, constraints
        of the same kind applied to independent groups of atoms are vectorised,
        so that the coordinates are taken from a 3d array of shape 
        (ngroups, 3*natoms, nbeads), where natoms is the total number of 
        atoms in a set. The list 'indices' determines which atoms of that set
        are subject to this particular constraint.

        Args:
            indices: 1-d list of indices of the affected atoms 
            targetvals: target values of the constraint function
            tolerance: the desired tolerance to which to converge the
            constraint
            domain: ['cartesian'/'normalmode'/'centroid'] - specifies whether 
            the constraint is expressed in terms of Cartesian, normalmode 
            or centroid coordinates.
            ngp: number of constraint groups; by default calculated internally
        """
        
        self.tol = tolerance
        self.domain = domain.lower()
        self.ilist = np.asarray(indices).flatten()
        self.natoms = depend_value(name="natoms", value=len(self.ilist))
        if self.domain not in ["cartesian", "normalmode", "centroid"]:
            raise ValueError("Unknown constraint domain '{:s}'.".format(domain))
        dself = dd(self)
        dself.ngp = depend_value(name="ngp", value=ngp)
        self.targetvals = targetvals
            
    def bind(self, beads, nm):
        """Bind the beads and the normal modes to the constraint.
        """
        
        self.beads = beads
        self.nm = nm
        self.nbeads = self.beads.nbeads
        # Check that the number of groups has been set
        if self.ngp == 0:
            raise ValueError("The number of atom groups must be specified!")
        dself = dd(self)
        if self.targetvals is not None:
            dself.targetvals = depend_array(
                    name="targetvals",
                    value=np.asarray(self.targetvals).reshape(self.ngp))
        else:
            dself.targetvals = depend_array(
                    name="targetvals",
                    value=np.nan*np.ones(self.ngp))
        if self.domain == "centroid":
            arr_shape = (self.ngp, 3*self.natoms, 1)
        else:
            arr_shape = (self.ngp, 3*self.natoms, self.nbeads)
        # Configurations of the affected beads (later to be made dependent
        # on sections of arrays in grouped constraints)
        dself.q = depend_array(name="q", value=np.zeros(arr_shape))
        dself.qprev = depend_array(name="qprev", value=dstrip(self.q[:]).copy())
        dself.g = depend_array(
                name="g", value=np.zeros(self.ngp),
                func=(lambda: self.gfunc(dstrip(self.q[:]))),
                dependencies=[dself.q])
        dself.Dg = depend_array(
                name="Dg", value=np.zeros(arr_shape), 
                func=(lambda: self.Dgfunc(dstrip(self.qprev[:]))), 
                dependencies=[dself.qprev])
        
    def norm(self, x):
        """Defines the norm of the constraint function; typically just
        the absolute value.
        """
        return np.abs(x)
    
    def gfunc(self, q):
        if q.ndim != 3:
            raise ValueError(
                "Constraint.gfunc expects a three-dimensional input.")
        if self.domain == "centroid" and q.shape[-1] != 1:
            raise ValueError(
                "Constraint.gfunc given input with shape[-1] != 1 when "+
                "centroid domain was specified."
                )
    def Dgfunc(self, q):
        if q.ndim != 3:
            raise ValueError(
                    "Constraint.Dgfunc expects a three-dimensional input.")
        if self.domain == "centroid" and q.shape[-1] != 1:
            raise ValueError(
                "Constraint.gfunc given input with shape[-1] != 1 when "+
                "centroid domain was specified."
                )
            
class BondLengthConstraint(ConstraintBase):
    """Constrain the mean bond-length
    """
    def __init__(self, index_list, targetvals=None,
                 tolerance=1.0e-4, domain="cartesian", ngp=0):
        super(BondLengthConstraint, self).__init__(index_list, targetvals,
                                                   tolerance, domain, ngp)
        if self.natoms != 2:
            raise ValueError(
                    "{:s} expected natoms == 2, got {:s}".format(
                            self.__class__.__name__, self.natoms))
        if self.domain == "normalmode":
            warnings.warn(
                "Using the 'BondLength' constraint in the 'normalmode' domain "+
                "may have unpredictable effects.")

    def gfunc(self, q):
        """Calculate the bond-length, averaged over the beads. 
        """
        super(BondLengthConstraint, self).gfunc(q)
        ngp, ncart, nbeads = q.shape
        x = np.reshape(q, (ngp, 2, 3, nbeads))
        xij = x[:,1]-x[:,0]
        return np.sqrt(np.sum(xij**2, axis=1)).mean(axis=-1)

    def Dgfunc(self, q):
        """Calculate the Jacobian of the constraint function.
        """

        super(BondLengthConstraint, self).Dgfunc(q)
        ngp, ncart, nbeads = q.shape
        x = np.reshape(q, (ngp, 2, 3, nbeads))
        xij = x[:,1]-x[:,0] # (ngp, 3, nbeads)
        r = np.sqrt(np.sum(xij**2, axis=1)) # (ngp, nbeads)
        xij /= r[:,None,:]
        return np.concatenate((-xij,xij), axis=1)/nbeads
    
class BondAngleConstraint(ConstraintBase):
    """Constraint the mean bond-angle.
    """
    
    def __init__(self, index_list, targetvals=None,
                 tolerance=1.0e-4, domain="cartesian", ngp=0):
        super(BondAngleConstraint, self).__init__(index_list, targetvals,
                                                  tolerance, domain, ngp)
        if self.natoms != 3:
            raise ValueError(
                    "{:s} expected natoms == 3, got {:s}".format(
                            self.__class__.__name__, self.natoms))
        if self.domain == "normalmode":
            warnings.warn(
                "Using the 'BondAngle' constraint in the 'normalmode' domain "+
                "may have unpredictable effects.")

    def gfunc(self, q):
        """Calculate the bond-angle, averaged over the beads. 
        """
        super(BondAngleConstraint, self).gfunc(q)
        ngp, ncart, nbeads = q.shape
        x = np.reshape(q, (ngp, 3, 3, nbeads))
        x01 = x[:,1]-x[:,0]
        x01 /= np.sqrt(np.sum(x01**2, axis=1))[:,None,:]
        x02 = x[:,2]-x[:,0]
        x02 /= np.sqrt(np.sum(x02**2, axis=1))[:,None,:]
        
        return np.arccos(np.sum(x01*x02, axis=1)).mean(axis=-1)

    def Dgfunc(self, q):
        """Calculate the Jacobian of the constraint function.
        """
        super(BondAngleConstraint, self).Dgfunc(q)
        ngp, ncart, nbeads = q.shape
        x = np.reshape(q, (ngp, 3, 3, nbeads)).copy()
        # 0-1
        x01 = x[:,1]-x[:,0]
        r1 = np.expand_dims(np.sqrt(np.sum(x01**2, axis=1)), axis=1)
        x01 /= r1
        # 0-2
        x02 = x[:,2]-x[:,0]
        r2 = np.expand_dims(np.sqrt(np.sum(x02**2, axis=1)), axis=1)
        x02 /= r2
        # jacobian
        ct = np.expand_dims(np.sum(x01*x02, axis=1), axis=1)
        st = np.sqrt(1.0-ct**2)
        x[:,1] = (ct*x01-x02)/(r1*st)
        x[:,2] = (ct*x02-x01)/(r2*st)
        x[:,0] = -(x[:,1]+x[:,2])
        return np.reshape(x, (ngp, ncart, nbeads))/nbeads

class GroupedConstraints(dobject):
    """Describes a set of k constraint functions that are applied to 
    ngp non-overlapping groups of atoms.
    """
    
    def __init__(self, constraint_list, constrained_atoms, ngroups,
                 maxit=100, qnmprev=None):
        """Initialise the set of grouped constraints
        
        Args:
            constraint_list: list of constraint functions
            constrained_atoms: list of atomic indices subject to the constraint
            ngroups: number of groups of constrained atoms
            maxit: maximum numbed of iterations to converge a single step
            qnmprev: normal-mode configuration at the end of the previous
                converged constrained propagation step, flattened from
                shape=(ngp, n3unique, nbeads)
        """

        self.clist = constraint_list
        self.maxit = maxit
        self.tol = np.asarray([c.tol for c in self.clist])
        dself = dd(self)
        dself.ngp = depend_value(name="ngp", value=ngroups)
        dself.ncons = depend_value(name="ncons", func=self.get_ncons)
        for c in self.clist:
            dpipe(dself.ngp, dd(c).ngp)
        # Collate the list of all constraint indices.
        self.iunique = np.reshape(np.asarray(constrained_atoms), (self.ngp,-1))
        self.mk_idmaps()
        self.qnmprev = qnmprev
        
    def get_ncons(self):
        return len(self.clist)
            
    def bind(self, beads, nm):
        self.beads = beads
        self.nm = nm
        self.nbeads = self.beads.nbeads
        for c in self.clist:
            c.bind(beads, nm)
        dself = dd(self)
        arr_shape = (self.nbeads, self.ngp, self.n3unique)
        #-------- Set up copies of the affected phase-space -----------#
        #------------- coordinates and relevant masses ----------------#
        dself.dynm3 = depend_array(
                name="dynm3", 
                value=np.zeros((self.ngp, self.n3unique, self.nbeads)),
                func=(lambda: np.transpose(np.reshape(
                        dstrip(self.nm.dynm3[:,self.i3unique.flatten()]),
                        arr_shape), [1,2,0])),
                dependencies=[dd(self.nm).dynm3])
        # Holds all of the atoms affected by this list of constraints
        dself.qnm = depend_array(
                name="qnm", 
                value = np.zeros((self.ngp, self.n3unique, self.nbeads)), 
                func = (lambda: np.transpose(np.reshape(
                        dstrip(self.nm.qnm[:,self.i3unique.flatten()]),
                        arr_shape), [1,2,0])),
                dependencies = [dd(self.nm).qnm])
        dself.pnm = depend_array(
                name="pnm", 
                value = np.zeros((self.ngp, self.n3unique, self.nbeads)), 
                func = (lambda: np.transpose(np.reshape(
                        dstrip(self.nm.pnm[:,self.i3unique.flatten()]),
                        arr_shape), [1,2,0])),
                dependencies = [dd(self.nm).pnm])
        if self.qnmprev is None:
            dself.qnmprev = depend_array(
                    name="qnmprev", value=dstrip(self.qnm[:]).copy())
        else:
            try:
                dself.qnmprev = depend_array(
                    name="qnmprev", 
                    value=np.reshape(self.qnmprev.copy(), 
                                     (self.ngp, self.n3unique, self.nbeads)))
            except:
                raise ValueError(
"Shape of previous converged configuration supplied at initialisation\n"+
"is inconsistent with the bound system: {:s} \= {:s}.".format(
                self.qnmprev.shape.__repr__(),
                self.qnm.shape.__repr__()))
        #--------- Set up Cartesian and centroid coordinates ----------#
        # TODO: in future check for open paths
        self.nmtrans = nm_fft(self.qnm.shape[2], np.prod(self.qnm.shape[:2])//3)
        dself.q = depend_array(
                name="q",
                value = np.zeros((self.ngp, self.n3unique, self.nbeads)), 
                func = (lambda: self._to_beads(dstrip(self.qnm[:]))),
                dependencies=[dself.qnm])
        dself.qprev = depend_array(
                name="qprev", value=np.zeros_like(dstrip(self.qnmprev)),
                func=(lambda: self._to_beads(dstrip(self.qnmprev[:]))), 
                dependencies=[dself.qnmprev])
        dself.qc = depend_array(
                name="qc", 
                value=np.zeros((self.ngp, self.n3unique, 1)),
                func = (lambda: dstrip(self.qnm[...,:1])/
                            np.sqrt(1.0*self.nbeads)),
                dependencies=[dself.qnm])
        dself.qcprev = depend_array(
                name="qcprev", 
                value=np.zeros_like(dstrip(self.qc)),
                func=(lambda: dstrip(self.qnmprev[...,:1])/
                          np.sqrt(1.0*self.nbeads)), 
                dependencies=[dself.qnmprev])
        #--- Synchronise the coordinates in the individual constraints ---#
        #----------- with the local copies in the main group -------------#
        def make_arrgetter(k, arr):
            return lambda: dstrip(arr[:,self.i3list[k]])
        for k,c in enumerate(self.clist):
            if c.domain == "cartesian":
                dd(c).q.add_dependency(dself.q)
                dd(c).q._func = make_arrgetter(k, self.q)
                dd(c).qprev.add_dependency(dself.qprev)
                dd(c).qprev._func = make_arrgetter(k, self.qprev)
            elif c.domain == "normalmode":
                dd(c).q.add_dependency(dself.qnm)
                dd(c).q._func = make_arrgetter(k, self.qnm)
                dd(c).qprev.add_dependency(dself.qnmprev)
                dd(c).qprev._func = make_arrgetter(k, self.qnmprev)
            else:
                dd(c).q.add_dependency(dself.qc)
                dd(c).q._func = make_arrgetter(k, self.qc)
                dd(c).qprev.add_dependency(dself.qcprev)
                dd(c).qprev._func = make_arrgetter(k, self.qcprev)
        # Target values
        targetvals = np.zeros((self.ngp, self.ncons))
        for k, c in enumerate(self.clist):
            if np.any(np.isnan(dstrip(c.targetvals))):
                targetvals[:,k] = dstrip(c.g[:])
            else:
                targetvals[:,k] = dstrip(c.targetvals)
        dself.targetvals = depend_array(
                name="targetvals", 
                value=targetvals)
        def make_targetgetter(k):
            return lambda: dstrip(self.targetvals[:,k])
        for k, c in enumerate(self.clist):
            dd(c).targetvals.add_dependency(dself.targetvals)
            dd(c).targetvals._func = make_targetgetter(k)
        dself.g = depend_array(
                name="g", 
                value=np.zeros((self.ngp, self.ncons)),
                func=self.gfunc,
                dependencies=[dd(c).g for c in self.clist])
        # Jacobian of the constraint function (with Eckart)
        dself.Dg = depend_array(
                name="Dg", 
                value=np.zeros((self.ngp,self.ncons,self.n3unique,self.nbeads)),
                func=self.Dgfunc, 
                dependencies=[dd(c).Dg for c in self.clist])
        # The Cholesky decomposition of the Gramian matrix
        dself.GramChol = depend_array(
                name="GramChol", 
                value=np.zeros((self.ngp,self.ncons,self.ncons)),
                func=self.GCfunc, 
                dependencies=[dself.Dg])
            
    def mk_idmaps(self):
        """Construct lookup dictionary and lists to quickly access the portions
        of arrays that are affected by the constraints
        """
        
        # Check all atomic indices are unique
        counts = np.unique(self.iunique, return_counts=True)[1]
        if np.any(counts != 1):
            raise ValueError(
"GroupedConstraints given overlapping groups of atoms.")
        # List of unique indices
        self.nunique = self.iunique.shape[1]
        self.i3unique = np.zeros((self.iunique.shape[0],
                                  self.iunique.shape[1]*3), dtype=int)
        self.n3unique = self.i3unique.shape[1]
        for i, i3 in zip(self.iunique, self.i3unique):
            i3[:] = np.asarray([ 3*k + j for k in i for j in range(3)])
        # List of constraint-specific indices
        self.i3list = []
        nlist = list(range(self.n3unique//3))
        for c in self.clist:
            idx = c.ilist
            counts = np.unique(idx, return_counts=True)[1]
            if np.any(counts != 1):
                raise ValueError(
                        "Constraint {:s} given duplicate indices".format(
                                c.__class__.__name__))
            for i in idx:
                if i not in nlist:
                    raise ValueError(
                        "Constraint {:s} given out-of-bound indices".format(
                                c.__class__.__name__))            
            self.i3list.append(list([3*i+j for i in idx for j in range(3)]))

    def _to_beads(self, arr):
        """
        Convert the array contents to normal mode coordinates.
        """
        # (ngp, n3unique, nbeads) <-> (nbeads, ngp, n3unique)
        wkspace = np.reshape(np.transpose(
                arr, [2,0,1]), (self.nm.nbeads, -1))
        return np.transpose(np.reshape(
               self.nmtrans.nm2b(wkspace), 
               (self.nm.nbeads, self.ngp, self.n3unique)),
               [1,2,0])

    def _to_nm(self, arr):
        """
        Convert array to Cartesian coordinates.
        """
        
        wkspace = np.reshape(np.transpose(
                arr, [2,0,1]), (self.nm.nbeads, -1))
        return np.transpose(np.reshape(
               self.nmtrans.b2nm(wkspace), 
               (self.nm.nbeads, self.ngp, self.n3unique)),
               [1,2,0])

    def gfunc(self):
        """Return the value of each of the constraints for each of the
        atoms groups. The result has shape (ngp,ncons)
        """
        return np.column_stack(list(dstrip(c.g[:]) for c in self.clist))

    def Dgfunc(self):
        """Return the Jacobian of each of the constraints for each of the
        atoms groups. The result has shape (ngp,ncons,ndim*natoms,nbeads)
        """

        ans = np.zeros((len(self.clist),)+self.qnmprev.shape)
        arr = np.zeros(self.qnmprev.shape) # wkspace for nm conversion
        for k, (c,i3) in enumerate(zip(self.clist,self.i3list)):
            if c.domain == "cartesian":
                arr[:] = 0.0
                arr[:,i3] = dstrip(c.Dg[:])
                ans[k,:] = self._to_nm(arr)
            elif c.domain == "centroid":
                ans[k,:,i3,0] = dstrip(c.Dg[:])*np.sqrt(self.nbeads)
            elif c.domain == "normalmode":
                ans[k,:,i3] = dstrip(c.Dg[:])
        return np.transpose(ans, axes=[1,0,2,3]) 

    def GCfunc(self):
        """Return the Cholesky decomposition of the Gramian matrix
        for each of the groups of atoms.
        """

        Dg = dstrip(self.Dg[:])
        Dgm = Dg / dstrip(self.dynm3[:,None,...])
        Dg = np.reshape(Dg, (self.ngp, self.ncons, -1))
        Dgm.shape = Dg.shape
        # (ngp, ncons, n)*(ngp, n, ncons) -> (ngp, ncons, ncons)
        gram = np.matmul(Dg, np.transpose(Dgm, [0, 2, 1]))
        return np.linalg.cholesky(gram)

    def norm(self, x):
        """Return the norm of the deviations from the targetvalues
        for an input of shape (ngp, k).
        """
        ans = np.zeros_like(x)
        for k, c in enumerate(self.clist):
            ans[:,k] = c.norm(x[:,k])
        return ans
    
    
class EckartGroupedConstraints(GroupedConstraints):
    """Describes a set of k constraint functions that are applied to 
    ngp non-overlapping groups of atoms, where each of the groups of
    atoms is also subject to the Eckart conditions.
    """
    
    def __init__(self, constraint_list, constrained_atoms, ngroups,
                 maxit=100, tolerance=1.0e-04, qnmprev=None, qref=None):
        """Initialise the set of grouped constraints
        
        Args:
            constraint_list: list of constraint functions
            constrained_atoms: list of atomic indices subject to the constraint
            ngroups: number of groups of constrained atoms
            maxit: maximum numbed of iterations to converge a single step
            tolerance: convergence criterion for the Eckart constraints
            qnmprev: normal-mode configuration at the end of the previous
                converged constrained propagation step, flattened from
                shape=(ngp, n3unique, nbeads)
            qref: reference configuration for the Eckart constraints
        """
    
        super(EckartGroupedConstraints, self).__init__(constraint_list,
             constrained_atoms, ngroups, maxit, qnmprev)
        self.tol = np.concatenate((self.tol, 6*[tolerance,]))
        self.qref = qref
    
    def get_ncons(self):
        return len(self.clist)+6
    
    def bind(self, beads, nm):
        super(EckartGroupedConstraints, self).bind(beads, nm)
        # Eckart variables
        dself = dd(self)
        if self.qref is None:
            # Use qcprev
            dself.qref = depend_array(
                    name="qref",
                    value=dstrip(self.qcprev[:]).copy().reshape((self.ngp,self.nunique,3)))
        else:
            dself.qref = depend_array(
                    name="qref",
                    value=self.qref.copy().reshape((self.ngp,self.nunique,3)))
        
        dself.m3 = depend_array(
                    name="m3",
                    value=np.zeros_like(dstrip(self.qref)),
                    func=(lambda: np.reshape(
                          dstrip(self.dynm3[...,0]), 
                          (self.ngp,self.nunique,3))))
        # Total mass of the group of atoms
        dself.mtot = depend_array(name="mtot", value=np.zeros(self.ngp),
            func=(lambda: dstrip(self.m3[:,:,0]).sum(axis=-1)),
            dependencies=[dself.m3]
            )
        # Coords of reference centre of mass
        dself.qref_com = depend_array(
                name="qref_com", value=np.zeros((self.ngp,3)),
                func=(lambda: np.sum(
                      dstrip(self.qref[:])*dstrip(self.m3[:]),
                      axis=1)/dstrip(self.mtot[:,None])),
                dependencies=[dself.m3, dself.qref]
                )
        # qref in its centre of mass frame
        dself.qref_rel = depend_array(
                name="qref_rel", value=np.zeros_like(dstrip(self.qref)),
                func=(lambda: dstrip(self.qref[:]) - 
                              dstrip(self.qref_com[:,None,:])),
                dependencies=[dself.qref_com]
                )
        # qref in the CoM frame, mass-weighted
        dself.mqref_rel = depend_array(
                name="mqref_rel", value=np.zeros_like(dstrip(self.qref)),
                func=(lambda: 
                    dstrip(self.qref_rel[:])*dstrip(self.m3[:])),
                dependencies=[dself.qref_rel]
                )
        dself.g_eckart = depend_array(
                name="g_eckart", value=np.zeros((self.ngp, 6)),
                func = self.gfunc_eckart,
                dependencies=[dself.mqref_rel, dself.qc]
                )
        dself.Dg_eckart = depend_array(
                name="Dg_eckart", 
                value=np.zeros((self.ngp, 6, self.n3unique, self.nbeads)),
                func = self.Dgfunc_eckart,
                dependencies=[dself.mqref_rel]
                )
        dself.g.add_dependency(dself.g_eckart)
        dself.Dg.add_dependency(dself.Dg_eckart)
        
    def gfunc_eckart(self):
        g = np.zeros((6, self.ngp))
        q = np.reshape(dstrip(self.qc[:]), (self.ngp, -1, 3))
        qref = dstrip(self.qref[:])
        mqref_rel = dstrip(self.mqref_rel[:])
        m = dstrip(self.m3[:])
        M = dstrip(self.mtot[:]).reshape((self.ngp,1))
        Delta = q-qref
        g[:3] = np.transpose(np.sum(m*Delta, axis=1)/M)
        g[3:] = np.transpose( np.sum ( np.cross(
                mqref_rel, Delta, axis=-1), axis=1)/M)
        return g.T
    
    def Dgfunc_eckart(self):
        # Eckart constraints
        Dg = np.zeros((6, self.ngp, self.n3unique//3, 3))
        m = dstrip(self.m3[:])
        M = dstrip(self.mtot[:]).reshape((self.ngp,1,1))
        mqref_rel = dstrip(self.mqref_rel[:])
        for i in range(3):
            Dg[i,:,:,i] = m[:,:,i]
        # Eckart rotation, x-component
        Dg[3,:,:,1] =-mqref_rel[:,:,2]
        Dg[3,:,:,2] = mqref_rel[:,:,1]
        # Eckart rotation, y-component
        Dg[4,:,:,0] = mqref_rel[:,:,2]
        Dg[4,:,:,2] =-mqref_rel[:,:,0]
        # Eckart rotation, z-component
        Dg[5,:,:,0] =-mqref_rel[:,:,1]
        Dg[5,:,:,1] = mqref_rel[:,:,0]
        Dg /= M
        ans = np.zeros((self.ngp, 6, self.n3unique, self.nbeads))
        Dg.shape = (6, self.ngp, -1)
        ans[...,0] = np.transpose(Dg, [1,0,2])
        return ans
        
    def gfunc(self):
        ans = super(EckartGroupedConstraints, self).gfunc()
        return np.hstack((ans, dstrip(self.g_eckart[:])))
    
    def Dgfunc(self):
        ans = super(EckartGroupedConstraints, self).Dgfunc()
        return np.hstack((ans, dstrip(self.Dg_eckart[:])))
    

class ConstraintSolverBase(dobject):

    def __init__(self, constraint_groups, dt=1.0):
        self.constraint_groups = constraint_groups
        dd(self).dt = depend_value(name="dt", value=dt)

    def proj_cotangent(self):
        raise NotImplementedError()

    def proj_manifold(self):
        raise NotImplementedError()  
        
class ConstraintSolver(ConstraintSolverBase):

    def __init__(self, constraint_groups, dt=1.0):
        super(ConstraintSolver,self).__init__(constraint_groups, dt)
        
    def bind(self, nm, dt=1.0):
        self.nm = nm

    def proj_cotangent(self):
        """Projects onto the cotangent space of the constraint manifold.
        """
        pnm = dstrip(self.nm.pnm[:]).copy()
        for cgp in self.constraint_groups:
            dynm3 = dstrip(cgp.dynm3[:])
            p = dstrip(cgp.pnm[:])
            v = np.reshape(p/dynm3, (cgp.ngp, -1, 1))
            Dg = np.reshape(dstrip(cgp.Dg[:]), (cgp.ngp, cgp.ncons, -1))
            b = np.matmul(Dg, v)
            GramChol = dstrip(cgp.GramChol[:])
            x = np.linalg.solve(np.transpose(GramChol, [0,2,1]),
                                np.linalg.solve(GramChol, b))
            pnm[:,cgp.i3unique.flatten()] -= np.reshape(
                    np.matmul(np.transpose(Dg, [0,2,1]), x),
                    (cgp.ngp*cgp.n3unique, self.nm.nbeads)).T
        self.nm.pnm[:] = pnm

    def proj_manifold(self):
        """Projects onto the constraint manifold using the Gram matrix
        defined by self.Dg and self.Gram
        """
        
        pnm = dstrip(self.nm.pnm[:]).copy()
        qnm = dstrip(self.nm.qnm[:]).copy()
        for cgp in self.constraint_groups:
            icycle = 0
            active = np.ones(cgp.ngp, dtype=bool)
            g = np.zeros((cgp.ngp, cgp.ncons, 1))
            Dg = np.transpose(np.reshape(dstrip(cgp.Dg[:]), 
                                         (cgp.ngp, cgp.ncons,-1)), [0,2,1])
            GramChol = dstrip(cgp.GramChol[:])
            dynm3 = dstrip(cgp.dynm3[:])
            # Fetch current normal-mode coordinates and temporarily
            # suspend automatic updates 
            cgp.qnm.update_auto()
            qfunc, cgp.qnm._func = cgp.qnm._func, None
            cgp.pnm.update_auto()
            pfunc, cgp.pnm._func = cgp.pnm._func, None
            while (icycle < cgp.maxit):
                g[active,:,0] = (dstrip(cgp.g[active]) - 
                                 dstrip(cgp.targetvals[active]))
                active = np.any(cgp.norm(g[...,0]) > cgp.tol, axis=-1)
                if not np.any(active):
                    break
                gc = GramChol[active]
                dlambda = np.linalg.solve(
                        np.transpose(gc, [0,2,1]),
                        np.linalg.solve(gc, g[active]))
                delta = np.reshape(np.matmul(Dg[active], dlambda),
                                   (-1, cgp.n3unique, self.nm.nbeads))
                cgp.qnm[active] -= (delta / dynm3[active])
                cgp.pnm[active] -= delta/self.dt
                icycle += 1
                if (icycle == cgp.maxit):
                    raise ValueError('No convergence in Newton iteration '+
                                     'for positional component')
            cgp.qnmprev[:] = dstrip(cgp.qnm).copy()
            qnm[:,cgp.i3unique.flatten()] = np.reshape(
                    dstrip(cgp.qnm), (-1, self.nm.nbeads)).T
            pnm[:,cgp.i3unique.flatten()] = np.reshape(
                    dstrip(cgp.pnm), (-1, self.nm.nbeads)).T
            # Restore automatic updates
            cgp.qnm._func = qfunc
            cgp.pnm._func = pfunc
        self.nm.pnm[:] = pnm
        self.nm.qnm[:] = qnm

class NVEConstrainedIntegrator(NVEIntegrator):
    """Integrator object for constant energy simulations of constrained
    systems.

    Has the relevant conserved quantity and normal mode propagator for the
    constant energy ensemble. Note that a temperature of some kind must be
    defined so that the spring potential can be calculated.

    Attributes:

    Depend objects:
        econs: Conserved energy quantity. Depends on the bead kinetic and
            potential energy, and the spring potential energy.
    """
    
    def get_gdt(self):
        """Geodesic flow timestep
        """
        return self.dt * 0.5 / self.inmts / self.nsteps_geo
    
    def pconstraints(self):
        """This removes the centre of mass contribution to the kinetic energy
        and projects the momenta onto the contangent space of the constraint
        manifold (implicitly assuming that the two operations commute)

        Calculates the centre of mass momenta, then removes the mass weighted
        contribution from each atom. If the ensemble defines a thermostat, then
        the contribution to the conserved quantity due to this subtraction is
        added to the thermostat heat energy, as it is assumed that the centre of
        mass motion is due to the thermostat.

        If there is a choice of thermostats, the thermostat
        connected to the centroid is chosen.
        """
        self.csolver.proj_cotangent()
        super(NVEConstrainedIntegrator, self).pconstraints()

    def bind(self, motion):
        """ Reference all the variables for simpler access."""

        dself = dd(self)
        dmotion = dd(motion)
        dself.nsteps_geo = dmotion.nsteps_geo
        
        super(NVEConstrainedIntegrator,self).bind(motion)
        self.csolver = motion.csolver
        dself.gdt = depend_value(name="gdt", func=self.get_gdt,
                                 dependencies=[dself.dt, dself.nmts])
        dpipe(dself.gdt, dd(self.csolver).dt)
        
    def free_p(self):
        """Velocity Verlet momentum propagator with ring-polymer spring forces,
           followed by projection onto the cotangent space of the constraint.
        """
        self.nm.pnm += dstrip(self.nm.fspringnm)*self.qdt
        self.pconstraints()
        
    def step_A(self):
        """Unconstrained A-step"""
        self.nm.qnm += dstrip(self.nm.pnm)/dstrip(self.nm.dynm3)*self.gdt
        
    def step_Ag(self):
        """Geodesic flow
        """
        for i in range(self.nsteps_geo):
            self.step_A()
            self.csolver.proj_manifold()
            self.csolver.proj_cotangent()
        
    def free_qstep_ba(self):
        """This overrides the exact free-ring-polymer propagator, performing 
        half of standard velocity Verlet with explicit spring forces. This is 
        done to retain the symplectic property of the constrained propagator
        """
        self.free_p()
        self.step_Ag()
        
    def free_qstep_ab(self):
        """This overrides the exact free-ring-polymer propagator, performing 
        half of standard velocity Verlet with explicit spring forces. This is 
        done to retain the symplectic property of the constrained propagator
        """
        self.step_Ag()
        self.free_p()

class NVTConstrainedIntegrator(NVEConstrainedIntegrator):

    """Integrator object for constant temperature simulations of constrained
    systems.

    Has the relevant conserved quantity and normal mode propagator for the
    constant temperature ensemble. Contains a thermostat object containing the
    algorithms to keep the temperature constant.

    Attributes:
        thermostat: A thermostat object to keep the temperature constant.
    """
    
    def tstep(self):
        """Geodesic integrator thermostat step.
        """
        
        self.thermostat.step()
        # Fix momenta and correct eens accordingly
        sm = np.sqrt(dstrip(self.nm.dynm3))
        p = (dstrip(self.nm.pnm)/sm).flatten()
        self.ensemble.eens += np.dot(p,p) * 0.5
        self.csolver.proj_cotangent()
        p = (dstrip(self.nm.pnm)/sm).flatten()
        self.ensemble.eens -= np.dot(p,p) * 0.5
        # CoM constraints include own correction to eens
        super(NVEConstrainedIntegrator, self).pconstraints()

    def step(self, step=None):
        """Does one simulation time step."""

        if self.splitting == "obabo":
            # thermostat is applied for dt/2
            self.tstep()
            # forces are integerated for dt with MTS.
            self.mtsprop(0)
            # thermostat is applied for dt/2
            self.tstep()

        elif self.splitting == "baoab":

            self.mtsprop_ba(0)
            # thermostat is applied for dt
            self.tstep()
            self.mtsprop_ab(0)

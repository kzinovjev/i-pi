"""Contains classes for planetary model calculations"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.


import time
import copy
import os
import sys
import numpy as np
try: from scipy import sparse
except: from ipi.utils import sparse

from ipi.engine.motion import Motion, Dynamics
from ipi.utils.depend import *
from ipi.engine.thermostats import *
from ipi.engine.normalmodes import NormalModes
from ipi.engine.barostats import Barostat
from ipi.utils.units import Constants

class Planetary(Motion):
    """Evaluation of the matrices needed in a planetary model by 
    constrained MD. 

    Gives the standard methods and attributes needed in all the
    dynamics classes.

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

    def __init__(self, timestep, mode="md", nsamples=0, stride=1, screen=0.0, nbeads=-1, thermostat=None, barostat=None, fixcom=False, fixatoms=None, nmts=None):
        """Initialises a "dynamics" motion object.

        Args:
            dt: The timestep of the simulation algorithms.
            fixcom: An optional boolean which decides whether the centre of mass
                motion will be constrained or not. Defaults to False.
        """
        
        self.mode = mode
        self.nsamples = nsamples
        self.stride = stride
        self.nbeads = nbeads
        self.screen = screen
         
        dself = dd(self)
        # the planetary step just computes constrained-centroid properties so it 
        # should not advance the timer
        dself.dt = depend_value(name="dt", value=0.0)
        #dset(self, "dt", depend_value(name="dt", value = 0.0) ) 
        self.fixatoms = np.asarray([])
        self.fixcom = True
        # nvt-cc means contstant-temperature with constrained centroid
        self.ccdyn = Dynamics(timestep, mode="nvt-cc", thermostat=thermostat, nmts=nmts, fixcom=fixcom, fixatoms=fixatoms)


    def bind(self, ens, beads, nm, cell, bforce, prng, omaker=None):
        """Binds ensemble beads, cell, bforce, and prng to the dynamics.

        This takes a beads object, a cell object, a forcefield object and a
        random number generator object and makes them members of the ensemble.
        It also then creates the objects that will hold the data needed in the
        ensemble algorithms and the dependency network. Note that the conserved
        quantity is defined in the init, but as each ensemble has a different
        conserved quantity the dependencies are defined in bind.

        Args:
            beads: The beads object from whcih the bead positions are taken.
            nm: A normal modes object used to do the normal modes transformation.
            cell: The cell object from which the system box is taken.
            bforce: The forcefield object from which the force and virial are
                taken.
            prng: The random number generator object which controls random number
                generation.
        """

        if self.nbeads < 0:
            self.nbeads = beads.nbeads
        self.prng = prng
        self.basebeads = beads
        self.basenm = nm
        self.dbeads = beads.copy(nbeads=self.nbeads)
        self.dcell = cell.copy()
        self.dforces = bforce.copy(self.dbeads, self.dcell)
        #self.dnm = nm.copy(freqs = nm.omegak[1]*np.ones(self.dbeads.nbeads-1))
        if isinstance(self.ccdyn.thermostat, (ThermoGLE, ThermoNMGLE, ThermoNMGLEG)):
            self.dnm = nm.copy()
            self.dnm.mode = "rpmd"
        else:
            self.dnm = nm.copy(freqs = nm.omegak[1]*self.nbeads*np.sin(np.pi/self.nbeads) \
                                      * np.ones(self.nbeads-1)
                                      / (beads.nbeads*np.sin(np.pi/beads.nbeads))
                             )
            self.dnm.mode = "manual"
        self.dnm.bind(ens, self, beads=self.dbeads, forces=self.dforces)
        self.dnm.qnm[:] = nm.qnm[:self.nbeads] * np.sqrt(self.nbeads) / np.sqrt(beads.nbeads)
        self.dens = ens.copy()
        self.dbias = ens.bias.copy(self.dbeads, self.dcell)
        self.dens.bind(self.dbeads, self.dnm, self.dcell, self.dforces, self.dbias)
       
        self.natoms = self.dbeads.natoms 
        natoms3 = self.dbeads.natoms*3
        self.omega2 = np.zeros((natoms3,natoms3), float)
        
        self.tmc = 0
        self.tmtx = 0
        self.tsave =0
        self.neval = 0
        
        self.ccdyn.bind(self.dens, self.dbeads, self.dnm, self.dcell, self.dforces, prng, omaker)
        
    def increment(self, dnm):
        sm3 = dstrip(self.dbeads.sm3)
        qms = dstrip(dnm.qnm) * sm3
        fms = dstrip(dnm.fnm) / sm3
        fms[0,:]=0
        qms[0,:]=0
        qms *= (dnm.omegak**2)[:,np.newaxis]
         
        self.omega2 += np.tensordot(fms,fms,axes=(0,0))
        qffq = np.tensordot(fms,qms,axes=(0,0))
        qffq = qffq + qffq.T
        qffq *= 0.5
        self.omega2 -= qffq        
    
    def matrix_screen(self):
        q = np.array(self.dbeads[0].q).reshape(self.natoms, 3)
        sij = q[:, np.newaxis, :] - q
        sij = sij.transpose().reshape(3, self.natoms**2)    
        # find minimum distances between atoms (rigorous for cubic cell)
        sij = np.matmul(self.dcell.ih, sij)
        sij -= np.around(sij)
        sij = np.matmul(self.dcell.h, sij)
        sij = sij.reshape(3, self.natoms, self.natoms).transpose()
        # take square magnitudes of distances
        sij = np.sum(sij*sij, axis=2)
        # screen with Heaviside step function
        sij = (sij < self.screen**2).astype(float)
        # sij = np.exp(-sij / (self.screen**2))
        # acount for 3 dimensions
        sij = np.concatenate((sij,sij,sij), axis=0)  
        sij = np.concatenate((sij,sij,sij), axis=1) 
        sij = sij.reshape(-1).reshape(-1, self.natoms).transpose()
        sij = sij.reshape(-1).reshape(-1, 3*self.natoms).transpose()
        return sij
    
    def step(self, step=None):
        
        if step is not None and step % self.stride != 0:
            return
            
        print "start planetary step"
        self.dnm.qnm[:] = self.basenm.qnm[:self.nbeads] * np.sqrt(self.nbeads) / np.sqrt(self.basebeads.nbeads)
        # randomized momenta
        self.dnm.pnm = self.prng.gvec((self.dbeads.nbeads,3*self.dbeads.natoms))*np.sqrt(self.dnm.dynm3)*np.sqrt(self.dens.temp*self.dbeads.nbeads*Constants.kb)
        self.dnm.pnm[0] = 0.0
        
        self.omega2[:] = 0.0
        
        self.tmtx -= time.time()
        self.increment(self.dnm)        
        self.tmtx += time.time()
        
        for istep in xrange(self.nsamples):
            self.tmc -= time.time()
            self.ccdyn.step(step)    
            self.tmc += time.time()
            self.tmtx -= time.time()        
            self.increment(self.dnm)  
            self.tmtx += time.time()                
        
        self.neval += 1
        
        self.omega2 /= self.dbeads.nbeads*self.dens.temp*(self.nsamples+1)*(self.dbeads.nbeads-1)
        self.tsave -= time.time()
        
        if self.screen > 0.0:
            scr = self.matrix_screen()
            self.omega2 *= scr
       
        # ensure perfect symmetry
        self.omega2[:] = 0.5 * (self.omega2 + self.omega2.transpose())
        # only save lower triangular part
        self.omega2[:] = np.tril(self.omega2)
 
        # save as a sparse matrix in half precision
        save_omega2 = sparse.csc_matrix(self.omega2.astype(np.float16))
       
        # Write to temporary binary file, then cut and append contents 
        # to permanent PLANETARY file
        with open("TEMP_PLANETARY", "wb") as f:
            sparse.save_npz(f, save_omega2, compressed=True)
        with open("TEMP_PLANETARY", "r") as f:
            text = f.read()
        if step == 0:
            fmt = "w"
        else:
            fmt = "a"
        with open("PLANETARY", fmt) as f:
            f.write(text)
            f.write("\nXXXXXXXXXX\n")
        os.remove("TEMP_PLANETARY") 
        
        self.tsave += time.time()
        print "AVG TIMING: ", self.tmc/self.neval, self.tmtx/self.neval, self.tsave/self.neval
"""
Contains the classes that deal with the different dynamics required in
different types of ensembles.

Copyright (C) 2013, Joshua More and Michele Ceriotti

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http.//www.gnu.org/licenses/>.
"""

__all__ = ['SCPhononsMover']

import numpy as np
import time
from ipi.engine.motion.motion import Motion
from ipi.utils.depend import *
from ipi.utils import units
from ipi.utils.softexit import softexit
from ipi.utils.messages import verbosity, warning, info
from ipi.utils.mathtools import gaussian_inv
try:
    from ipi.utils.sobol.sobol import i4_sobol as sobol
except ImportError as sobol_expection:
    sobol = None


class SCPhononsMover(Motion):
    """
    Self consistent phonons method.
    """

    def __init__(self, fixcom=False, fixatoms=None, mode="sc", dynmat=np.zeros(0, float), prefix="", asr="none", max_steps=500, max_iter=1, tau=-1, chop=1e-9, random_type="pseudo", displace_mode="rewt", wthreshold=0.0, precheck=True, checkweights=False, nparallel=1, batch_weight_exponent=1):
        """
        Initialises SCPhononsMover.
        Args:
        fixcom  : An optional boolean which decides whether the centre of mass
                  motion will be constrained or not. Defaults to False.
        matrix  : A 3Nx3N array that stores the dynamic matrix.
        oldk    : An integr that stores the number of rows calculated.
        delta: A 3Nx3N array that stores the dynamic matrix.
        """

        super(SCPhononsMover, self).__init__(fixcom=fixcom, fixatoms=fixatoms)

        # Finite difference option.
        self.dynmatrix = dynmat
        self.frefine = False
        self.U = None
        self.mode = mode
        self.asr = asr
        self.tau = tau
        self.max_steps = max_steps
        self.max_iter = max_iter
        self.prefix = prefix
        self.chop = chop
        self.random_type = random_type
        self.displace_mode = displace_mode
        self.wthreshold = wthreshold
        self.precheck = precheck
        self.checkweights = checkweights
        self.nparallel = nparallel
        self.batch_weight_exponent = batch_weight_exponent

        if self.random_type == "sobol" and sobol == None:
            raise (sobol_expection)

        if self.prefix == "":
            self.prefix = "scphonons"

    def bind(self, ens, beads, nm, cell, bforce, prng, omaker):

        super(SCPhononsMover, self).bind(ens, beads, nm, cell, bforce, prng, omaker)

        # Raises error if nparallel is not a factor of max_steps
        if self.max_steps % self.nparallel !=0:
            raise ValueError(
                "Number of parallel force evaluations is not a factor of the maximum number of Monte Carlo steps.")

        # Raises error for nbeads not equal to 1.
        if(self.beads.nbeads > 1):
            raise ValueError(
                "Calculation not possible for number of beads greater than one")

        # Initialises a 3*number of atoms X 3*number of atoms dynamic matrix.
        if(self.dynmatrix.size != (beads.q.size * beads.q.size)):
            if(self.dynmatrix.size == 0):
                self.dynmatrix = np.zeros((beads.q.size, beads.q.size), float)
                self.dynmatrix_r = np.zeros(
                    (beads.q.size, beads.q.size), float)
            else:
                raise ValueError(
                    "Force constant matrix size does not match system size")
        self.dynmatrix = self.dynmatrix.reshape((beads.q.size, beads.q.size))
        self.atol = self.chop

        # Creates dublicate classes to easer computation of forces.
        self.dof = 3 * self.beads.natoms
        self.dbeads = self.beads.copy(nbeads=self.nparallel)
        self.dcell = self.cell.copy()
        self.dforces = self.forces.copy(self.dbeads, self.dcell)

        # Sets temperature.
        self.temp = self.ensemble.temp
        self.m = dstrip(self.beads.m).copy()

        # Initializes mass related arrays.
        self.m3 = dstrip(self.beads.m3[-1])
        self.im3 = np.divide(1., self.m3)
        self.sqm3 = np.sqrt(self.m3)
        self.isqm3 = np.sqrt(self.im3)

        # Initializes mass related diagonal matrices.
        self.M = np.diag(self.m3)
        self.iM = np.diag(self.im3)
        self.sqM = np.diag(self.sqm3)
        self.isqM = np.diag(self.isqm3)

        # Initializes variables specific to sc phonons.
        self.isc = 0
        self.imc = 0
        self.phononator = SCPhononator()
        self.phononator.bind(self)

        # a random shuffle to allow some randomness in the sobol-like PRNGs
        self.prng = prng
        self.random_shuffle = np.asarray(range(self.dof))

        # defines function to perform a Gaussian transformation.
        self.fginv = np.vectorize(gaussian_inv)

        #! TODO implement an option to give the file name to fetch the random samples. Also implement check of dimensionality, and raise an error if it runs out of random numbers
        # reads sobol points from file!
        if self.random_type == "file":
            self.random_sequence = np.loadtxt("SOBOL-RNG")
        elif self.random_type == "pseudo":
            self.random_sequence = self.prng.rng.rand(self.max_steps * self.max_iter, self.dof)
        elif self.random_type == "sobol":
            self.random_sequence = np.asarray([sobol(self.dof, i) for i in range(0,self.max_steps * self.max_iter + 1)])
        
        # Shuffles the 
        self.prng.rng.shuffle(self.random_shuffle)

    def step(self, step=None):
        if(self.isc == self.max_iter):
            softexit.trigger(" @SCP : Reached maximum iterations. Terminating the SCP calculation.")
        if(self.imc == 0):
            self.phononator.reset()
        elif(self.imc >= 1 and self.imc <= self.max_steps):
            self.phononator.step(step)
        elif(self.imc > self.max_steps):
            self.phononator.print_energetics()
            self.phononator.displace()


class DummyPhononator(dobject):
    """ No-op phononator """

    def __init__(self):
        pass

    def bind(self, dm):
        """ Reference all the variables for simpler access."""
        self.dm = dm

    def reset(self):
        """Dummy reset step which does nothing."""
        pass

    def step(self, step=None):
        """Dummy simulation step which does nothing."""
        pass

    def print_energetics(self):
        """Dummy print step which does nothing."""
        pass

    def displace(self):
        """Dummy evaluation step which does nothing."""
        pass


class SCPhononator(DummyPhononator):
    """ Self consistent phonon evaluator.
    """

    def bind(self, dm):
        """
        Reference all the variables for simpler access.
        """
        super(SCPhononator, self).bind(dm)
        self.v = np.zeros((self.dm.max_iter, self.dm.max_steps))
        self.x = np.zeros((self.dm.max_iter, self.dm.max_steps, self.dm.dof))
        self.q = np.zeros((self.dm.max_iter, 1, self.dm.dof))
        self.f = np.zeros((self.dm.max_iter, self.dm.max_steps, self.dm.dof))
        self.iD = np.zeros((self.dm.max_iter, self.dm.dof, self.dm.dof))

        # New variables added to dampen the displacements (between scp steps)
        self.dq_old = np.zeros(self.dm.dof)
        self.dq = np.zeros(self.dm.dof)
        self.dw_old = np.zeros(self.dm.dof)
        self.dw = np.zeros(self.dm.dof)
        self.w2 = np.zeros(self.dm.dof)
        self.w = np.zeros(self.dm.dof)
        self.wthreshold = self.dm.wthreshold
        self.precheck = self.dm.precheck
        self.checkweights = self.dm.checkweights

    def reset(self):
        """
        Resets the variables for a new round of phonon evaluation.
        """
        self.dm.w = np.zeros(self.dm.dof)
        self.dm.iw = np.zeros(self.dm.dof)
        self.dm.iw2 = np.zeros(self.dm.dof)
        self.dm.avgPot = 0.
        self.dm.avgForce = np.zeros((1, self.dm.dof))
        self.dm.avgHessian = np.zeros((self.dm.dof, self.dm.dof))
        self.apply_asr()
        self.apply_hpf()
        self.get_KnD()
        self.iD[self.dm.isc] = self.dm.iD.copy()
        self.q[self.dm.isc] = self.dm.beads.q.copy()
        self.dm.oldK = self.dm.K
        self.dm.imc = 1

        info("\n @SCP : Beginning SCP step no. %8d" % (self.dm.isc,), verbosity.medium)

        # prints the force constant matrix.
        outfile = self.dm.output_maker.get_output(self.dm.prefix + ".K." + str(self.dm.isc))
        np.savetxt(outfile,  self.dm.K)
        outfile.close()
        info(" @SCP : Saving the force constant matrix.", verbosity.medium)

        # prints the inverse displacement correlation matrix.
        outfile = self.dm.output_maker.get_output(self.dm.prefix + ".iD." + str(self.dm.isc))
        np.savetxt(outfile,  self.dm.iD)
        outfile.close()
        info(" @SCP : Saving the inverse displacement correlation matrix.", verbosity.medium)

        # prints the mean position.
        outfile = self.dm.output_maker.get_output(self.dm.prefix + ".q." + str(self.dm.isc))
        np.savetxt(outfile,  self.dm.beads.q)
        outfile.close()
        info(" @SCP : Saving the mean position.", verbosity.medium)

        # prints the frequencies. 
        outfile = self.dm.output_maker.get_output(self.dm.prefix + ".w." + str(self.dm.isc))
        np.savetxt(outfile,  self.dm.w)
        outfile.close()
        info(" @SCP : Saving the frequencies.", verbosity.medium)

        # prints the potential energy at the mean position. 
        outfile = self.dm.output_maker.get_output(self.dm.prefix + ".V0." + str(self.dm.isc))
        np.savetxt(outfile,  self.dm.forces.pots)
        outfile.close()
        info(" @SCP : Saving the minimum potential.", verbosity.medium)

        info(" @SCP : Generating %8d new confiurations to be sampled." % (self.dm.max_steps,), verbosity.medium)
        # Creates a list of configurations that are to be sampled.
        while self.dm.imc <= self.dm.max_steps:

            irng = (self.dm.isc) * self.dm.max_steps / \
                2 + (self.dm.imc + 1) / 2
            x = self.dm.fginv(self.dm.random_sequence[irng])

            # picks the elements of the vector in a random order.
            # this introduces a degree of randomness in the sobol-like PRNGs
            x = x[self.dm.random_shuffle]

            # Transforms the "normal" random number and stores it.
            x = np.dot(self.dm.isqM, np.dot(self.dm.sqtD, x))
            self.x[self.dm.isc, self.dm.imc - 1] = (self.dm.beads.q + x.T)[-1]
            self.dm.imc += 1

            # Performs an inversion to the displacement and samples another configuration.
            x = -x
            self.x[self.dm.isc, self.dm.imc - 1] = (self.dm.beads.q + x.T)[-1]
            self.dm.imc += 1

        # Resets the number of MC steps to 1.
        self.dm.imc = 1
        info(" @SCP : Performing %8d new force evaluations." % (self.dm.max_steps,), verbosity.medium)

    def step(self, step=None):
        """
        Executes one monte carlo step.
        """

        imcmin = self.dm.imc - 1
        imcmax = self.dm.imc - 1 + self.dm.nparallel

        x = self.x[self.dm.isc, imcmin: imcmax]
        self.dm.dbeads.q = x

        v = dstrip(self.dm.dforces.pots).copy()
        f = dstrip(self.dm.dforces.f).copy()

        self.v[self.dm.isc, imcmin : imcmax ] = v[:]
        self.f[self.dm.isc, imcmin : imcmax ] = f[:]

        self.dm.imc += self.dm.nparallel

    def print_energetics(self):
        """
        Prints the energetics of the sampled configurations.
        """

        # prints the sampled configurations.
        outfile = self.dm.output_maker.get_output(self.dm.prefix + ".x." +
                   str(self.dm.isc))
        np.savetxt(outfile,  self.x[self.dm.isc])
        outfile.close()
        info(" @SCP : Saving the sampled confiugrations.", verbosity.medium)

        # prints the potential energy of the sampled configurations.
        outfile = self.dm.output_maker.get_output(self.dm.prefix + ".v." +
                   str(self.dm.isc))
        np.savetxt(outfile,  self.v[self.dm.isc])
        outfile.close()
        info(" @SCP : Saving the potential energy of the sampled configurations.", verbosity.medium)

        # prints the sampled configurations.
        outfile = self.dm.output_maker.get_output(self.dm.prefix + ".f." +
                   str(self.dm.isc))
        np.savetxt(outfile,  self.f[self.dm.isc])
        outfile.close()
        info(" @SCP : Saving the forces of the sampled configurations.", verbosity.medium)

        self.dm.isc += 1
        self.dm.imc = 0

    def displace(self):
        """
        Displaces the center of the distribution towards the optimized limit.
        """

        # Conditionally, before performing the optimization checks if
        # the forces are statistically significant by checking if the error
        # is smaller than the forces. When sobol sequences are used, since 
        # the true error is not known, we use the "Monte Carlo standrard error"
        #  as a proxy. 

        if self.precheck:

            # Calculates the force, the error and the
            # batch weights.
            f, f_err, batch_w = self.weighted_force()
            fnm = np.dot(self.dm.V.T, f)

            #Assumes error = standard error.
            fnm_err = np.sqrt(np.dot(self.dm.V.T**2, f_err**2))
            fnm[self.z] = 0.0

            if np.all(np.abs(fnm) < fnm_err):
                info(" @SCP : All the average force components are statistically insignificant.", verbosity.medium)  
                info(" @SCP : Skipping the optimization step.", verbosity.medium)  
                return

        # Uses the displacement correction and a used defined scaling 
        # factor to move in the direction of the force.

        elif(self.dm.displace_mode == "sd"):

            # Calculates the force, the error and the
            # batch weights.
            f, f_err, batch_w = self.weighted_force()

            # Calculates the Hessian..
            k = self.weighted_hessian()

            # Displaces.
            self.dm.beads.q += np.dot(self.dm.D, f) * self.dm.tau

            # Calculates the new dynamical matrix.
            self.dm.dynmatrix = np.dot(self.dm.isqM, np.dot(
                (k + k.T) / 2.0, self.dm.isqM))

            # Applies acoustic sum rule to project out zero modes.
            # Applies a high pass filter to zero out low frequency modes.
            # Calculates the displacement correction matrix.
            self.apply_asr()
            self.apply_hpf()
            self.get_KnD()

            info(" @SCP : <f> =  %10.8f +/-  %10.8f: " % (f, f_err), verbosity.medium)
            info(" @SCP : OPTMODE = sd : Using the displacement correlation as a preconditioner.", verbosity.medium)

        # Uses the inverse Hessian to determine how much
        # to move in the direction of the force.

        if(self.dm.displace_mode == "ik"):

            # Calculates the force, the error and the
            # batch weights.
            f, f_err, batch_w = self.weighted_force()
            self.dm.beads.q += np.dot(self.dm.iK, f)

            # Calculates the new dynamical matrix.
            k = self.weighted_hessian()
            self.dm.dynmatrix = np.dot(self.dm.isqM, np.dot(
                (k + k.T) / 2.0, self.dm.isqM))

            # Applies acoustic sum rule to project out zero modes.
            # Applies a high pass filter to zero out low frequency modes.
            # Calculates the displacement correction matrix.
            self.apply_asr()
            self.apply_hpf()
            self.get_KnD()

            info(" @SCP : <f> =  %10.8f +/-  %10.8f: " % (f, f_err), verbosity.medium)
            info(" @SCP : OPTMODE = iK : Using the inverse Hessian as a preconditioner.",  verbosity.medium)

        # Same as iK but only moves along normal modes which have statistically
        # significant force components.

        elif(self.dm.displace_mode == "nmik"):

            # Calculates the force, the error and the
            # batch weights.
            f, f_err, batch_w = self.weighted_force()

            # Calculates the new dynamical matrix.
            k = self.weighted_hessian()
            self.dm.dynmatrix = np.dot(
                self.dm.isqM, np.dot((k + k.T) / 2.0, self.dm.isqM))

            # Applies acoustic sum rule to project out zero modes.
            # Applies a high pass filter to zero out low frequency modes.
            # Calculates the displacement correction matrix.
            self.apply_asr()
            self.apply_hpf()
            self.get_KnD()

            # Calculates the force along normal moces.
            # Zeros the forces along the known "zero modes".
            # Estimates the error.
            fnm = np.dot(self.dm.V.T, f)
            fnm[self.z] = 0.0
            fnm_err = np.sqrt(np.dot(self.dm.V.T**2, f_err**2))

            # Zeros the displacement along "insignificant" normal modes.
            iKfnm = self.dm.iw2 * fnm
            iKfnm[np.abs(fnm) < fnm_err] = 0.0
            dqnm = iKfnm
            dqnm[self.z] = 0.0

            self.dm.beads.q += np.dot(self.dm.V, dqnm)

            # Calculates the new forces.
            f, f_err, batch_w = self.weighted_force()

            info(" @SCP : <f> =  %10.8f +/-  %10.8f: : number of batch weights > %10.8f =  %8d / %8d" % (np.linalg.norm(f), np.linalg.norm(f_err), self.wthreshold, sum(batch_w > self.wthreshold), len(batch_w)), verbosity.medium)
            info(" @SCP : OPTMODE = nmiK : Using the inverse Hessian along statistically significant normal modes as a preconditioner.",  verbosity.medium)

        elif(self.dm.displace_mode == "rnmik"):

            scale_forces = 1.0 * self.dm.tau

            # Outer Optimization Loop
            while True:

                # Inner Optimization Loop
                w = 1.0

                while True:

                    # Calculates the force, the error and the
                    # batch weights.
                    f, f_err, batch_w = self.weighted_force()

                    # Saves the change in the weight associated with
                    # last batch.
                    w_old = w
                    w = batch_w[-1]
                    if np.absolute(w - w_old) < 1e-4:
                        scale_forces *= 1.1
                        #info(" @SCP: Increasing displacement by 10 %.")
                    elif np.absolute(w - w_old) > 5e-2:
                        scale_forces /= 1.1

                    # Zeros the displacement along "insignificant" normal modes.
                    iKfnm = self.dm.iw2 * fnm
                    iKfnm[np.abs(fnm) < fnm_err] = 0.0
                    dqnm = iKfnm
                    dqnm[self.z] = 0.0

                    # Breaks out if the forces are statistically insignificant
                    # or if the weights become large.
                    # Breaks if all the forces are statistically insignificant.
                    if np.all(np.abs(fnm) < fnm_err):
                        info(" @SCP : All forces are statistically insignificant.", verbosity.medium)
                        info(" @SCP : Exiting the inner optimization loop.", verbosity.medium)  
                        break
                    # or if the batch weight goes to shit.
                    elif batch_w[-1] < self.wthreshold:
                        info(" @SCP : Batch weight is small.", verbosity.medium)
                        info(" @SCP : Exiting the inner optimization loop.", verbosity.medium)  
                        break

                    self.dm.beads.q += np.dot(self.dm.V, dqnm)
                    info(" @SCP : <f> =  %10.8f +/-  %10.8f: : number of batch weights > %10.8f =  %8d / %8d : largest batch weight = %10.8e" % (np.linalg.norm(f), np.linalg.norm(f_err), self.wthreshold, sum(batch_w > self.wthreshold), len(batch_w), batch_w[-1]), verbosity.medium)

                # Calculates the hessian.
                # Applies acoustic sum rule to project out zero modes.
                # Applies a high pass filter to zero out low frequency modes.
                # Calculates the displacement correction matrix.
                k = self.weighted_hessian()
                self.dm.dynmatrix = np.dot(
                    self.dm.isqM, np.dot((k + k.T) / 2.0, self.dm.isqM))
                self.apply_asr()
                self.apply_hpf()
                self.get_KnD()

                # Stores the old forces.
                fnm_old, fnm_err_old = fnm, fnm_err

                # Recalculates the new forces in normal mode coordinates..
                f, f_err, batch_w = self.weighted_force()
                f_err = f_err
                fnm = np.dot(self.dm.V.T, f)
                fnm[self.z] = 0.0
                fnm_err = np.sqrt(np.dot(self.dm.V.T**2, f_err**2))

                # Breaks if all the forces are statistically insignificant.
                if np.all(np.abs(fnm) < fnm_err):
                    info(" @SCP : All forces are statistically insignificant.", verbosity.medium)
                    info(" @SCP : Skipping the optimization step.", verbosity.medium)  
                    break
                # or if the batch weights have gone to shit.
                elif np.max(batch_w) < self.wthreshold:
                    info(" @SCP : Batch weights are small..", verbosity.medium)
                    info(" @SCP : Skipping the optimization step.", verbosity.medium)  
                    break

                if np.all(np.abs(fnm_old - fnm) < fnm_err):
                    info(" @SCP : All forces are converged.", verbosity.medium)  
                    break

    def weighted_force(self):
        """
        Returns the reweighted force, the associated statistical error
        and the batch weights.
        """

        # Creates new variable names for easier referencing.
        qp, iDp, Kp, i =  self.dm.beads.q, self.dm.iD, self.dm.K, self.dm.isc - 1

        # Takes the set of forces calculated at the previous step for (self.q, self.iD)
        avg_f = dstrip(self.f[i]).copy()[-1] * 0.0
        var_f = dstrip(self.f[i]).copy()[-1] * 0.0
        norm = 0.0
        batch_w = np.zeros(self.dm.isc)

        for i in range(self.dm.isc):

            # Calculates the harmonic force.
            f = self.f[i]
            x = self.x[i]
            fh = -1.0 * np.dot(x - qp, Kp)

            # Calculates the weights to calculate average for the distribution for (qp, iDp).
            w, sw, rw = self.calculate_weights(i)
            batch_w[i] = sw

            # Simply multiplies the forces by the weights and averages.
            V1 = np.sum(rw)
            V2 = np.sum(rw**2)
            avg_fi = np.sum(rw * (f - fh), axis=0) / V1
            var_fi = np.sum(rw * (f - fh - avg_fi)**2, axis=0) / (V1 - V2 / V1)
            avg_f += sw * avg_fi
            var_f += sw**2 * var_fi
            norm += sw

        return avg_f / norm,  np.sqrt(var_f / norm**2 / self.dm.max_steps), batch_w

    def weighted_hessian(self):
        """
        Returns the reweighted Hessian.
        """

        # Creates new variable names for easier referencing.
        qp, iDp, Kp, i =  self.dm.beads.q, self.dm.iD, self.dm.K, self.dm.isc - 1

        # Takes the set of forces calculated at the previous step for (self.q, self.iD)
        avg_K = dstrip(self.dm.iD).copy() * 0.0
        norm = 0.0

        for i in range(self.dm.isc):
            #Draws the random configurations and forces.
            f = self.f[i]
            x = self.x[i]

            # Calculates the harmonic forces.
            fh = -1.0 * np.dot(x - qp, Kp)

            # Calculates the weights to calculate average for the distribution for (qp, iDp)
            w, sw, rw = self.calculate_weights(i)

            # Simply multiplies the forces by the weights and averages
            V1 = np.sum(rw)

            # Simply multiplies the forces by the weights and averages
            avg_dKi = -np.dot(iDp, np.dot(((f - fh) * rw).T, (x - qp[-1])).T) / V1
            avg_Ki = Kp + 0.50 * (avg_dKi + avg_dKi.T)
            avg_K += sw * avg_Ki
            norm += sw

        return avg_K / norm

    def calculate_weights(self, i):
        """
        Computes the weights to sample (verb) a distribution described at (qp, iDp) using
        samples (noun) generated at the i^th SCPhonons step.
        """

        # Creates new variable names for easier referencing.
        qp, iDp, Kp =  self.dm.beads.q, self.dm.iD, self.dm.K

        # Takes the set of positions calculated at the previous step for (self.q, self.iD)
        x = self.x[i].copy()

        # Stores self.q in a vector
        q0 = self.q[i]
        iD0 = self.iD[i]

        # Estimates the weights as the ratio of density matrix for (qp, iDp) to the density matrix for (self.q, self.iD)
        rw = np.exp(-(0.50 * np.dot(iDp, (qp - x).T).T * (qp - x)).sum(axis=1) +
                    (0.50 * np.dot(iD0, (x - q0).T).T * (x - q0)).sum(axis=1))

        if rw.sum() < 1e-24:
            w = rw * 0.0
        else:
            w = rw / rw.sum()

        w = w.reshape((self.dm.max_steps, 1))
        rw = rw.reshape((self.dm.max_steps, 1))
        return w, np.nan_to_num(np.exp(-np.var(np.log(w))))**self.dm.batch_weight_exponent, rw

    def get_KnD(self):
        """
        Computes the force constant, displacement displacement correlation matrix and related matrices.
        """

        td = np.zeros(self.dm.dof)
        tdm = np.zeros(self.dm.dof)
        itd = np.zeros(self.dm.dof)
        itdm = np.zeros(self.dm.dof)
        sqtdm = np.zeros(self.dm.dof)
        sqtd = np.zeros(self.dm.dof)

        if self.dm.mode == "qn":
            # Calculates the mass scaled displacements for all non-zero modes.
            td[self.nz] = np.nan_to_num(
                0.50 * self.dm.iw[self.nz] / np.tanh(0.50 * self.dm.w[self.nz] / self.dm.temp))
            td[self.z] = 0.0
            tdm[self.nz] = np.nan_to_num(
                0.50 * self.dm.iw[self.nz] / np.tanh(0.50 * self.dm.w[self.nz] / self.dm.temp))
            tdm[self.z] = 0.0
        elif self.dm.mode == "cl":
            td[self.nz] = (self.dm.iw[self.nz])**2 * self.dm.temp
            td[self.z] = 0.0
            tdm[self.nz] = (self.dm.iw[self.nz])**2 * self.dm.temp
            tdm[self.z] = 0.0

        # Calculates the inverse and the square of the mass scaled displacements.
        itd[self.nz] = np.divide(1.0, td[self.nz])
        itd[self.z] = 0.0
        itdm[self.nz] = np.divide(1.0, tdm[self.nz])
        itdm[self.z] = 0.0
        sqtd[self.nz] = np.sqrt(td[self.nz])
        sqtd[self.z] = 0.0
        sqtdm[self.nz] = np.sqrt(tdm[self.nz])
        sqtdm[self.z] = 0.0

        self.dm.itK = np.eye(self.dm.dof) * 0.0
        self.dm.tD = np.eye(self.dm.dof) * 0.0
        self.dm.tDm = np.eye(self.dm.dof) * 0.0
        self.dm.itD = np.eye(self.dm.dof) * 0.0
        self.dm.itDm = np.eye(self.dm.dof) * 0.0
        self.dm.sqtD = np.eye(self.dm.dof) * 0.0
        self.dm.sqtDm = np.eye(self.dm.dof) * 0.0

        for i in range(self.dm.dof):
            if self.z[i]:
                continue
            U = self.dm.U.T[i].reshape((1, self.dm.dof))
            m = np.dot(U.T, U)
            self.dm.itK += self.dm.iw2[i] * m
            self.dm.tD += td[i] * m
            self.dm.tDm += tdm[i] * m
            self.dm.itD += itd[i] * m
            self.dm.itDm += itdm[i] * m
            self.dm.sqtD += sqtd[i] * m
            self.dm.sqtDm += sqtd[i] * m

        self.dm.K = np.dot(self.dm.sqM, np.dot(self.dm.dynmatrix, self.dm.sqM))
        self.dm.iD = np.dot(self.dm.sqM, np.dot(self.dm.itD, self.dm.sqM))
        self.dm.iDm = np.dot(self.dm.sqM, np.dot(self.dm.itDm, self.dm.sqM))
        self.dm.D = np.dot(self.dm.isqM, np.dot(self.dm.tD, self.dm.isqM))
        self.dm.sqD = np.dot(self.dm.isqM, np.dot(self.dm.sqtD, self.dm.isqM))
        self.dm.iK = np.dot(self.dm.isqM, np.dot(self.dm.itK, self.dm.isqM))

    def apply_asr(self):
        """
        Removes the translations and/or rotations depending on the asr mode.
        """

        if(self.dm.asr == "internal"):

            self.dm.w2, self.dm.U = np.linalg.eigh(self.dm.dynmatrix)
            self.dm.V = self.dm.U.T[-(self.dm.dof - 5):]
            self.dm.v2 = self.dm.w2[-(self.dm.dof - 5):]
            self.dm.dynmatrix = np.dot(
                self.dm.V, np.dot(self.dm.dynmatrix, self.dm.V.T))

        if(self.dm.asr == "molecule"):

            # Computes the centre of mass.
            com = np.dot(np.transpose(self.dm.beads.q.reshape(
                (self.dm.beads.natoms, 3))), self.dm.m) / self.dm.m.sum()
            qminuscom = self.dm.beads.q.reshape(
                (self.dm.beads.natoms, 3)) - com
            # Computes the moment of inertia tensor.
            moi = np.zeros((3, 3), float)
            for k in range(self.dm.beads.natoms):
                moi -= np.dot(np.cross(qminuscom[k], np.identity(3)),
                              np.cross(qminuscom[k], np.identity(3))) * self.dm.m[k]

            U = (np.linalg.eig(moi))[1]
            R = np.dot(qminuscom, U)
            D = np.zeros((6, 3 * self.dm.beads.natoms), float)

            # Computes the vectors along translations and rotations.
            D[0] = np.tile([1, 0, 0], self.dm.beads.natoms) / self.dm.isqm3
            D[1] = np.tile([0, 1, 0], self.dm.beads.natoms) / self.dm.isqm3
            D[2] = np.tile([0, 0, 1], self.dm.beads.natoms) / self.dm.isqm3
            for i in range(3 * self.dm.beads.natoms):
                iatom = i / 3
                idof = np.mod(i, 3)
                D[3, i] = (R[iatom, 1] * U[idof, 2] - R[iatom, 2]
                           * U[idof, 1]) / self.dm.isqm3[i]
                D[4, i] = (R[iatom, 2] * U[idof, 0] - R[iatom, 0]
                           * U[idof, 2]) / self.dm.isqm3[i]
                D[5, i] = (R[iatom, 0] * U[idof, 1] - R[iatom, 1]
                           * U[idof, 0]) / self.dm.isqm3[i]

            # Computes unit vecs.
            for k in range(6):
                D[k] = D[k] / np.linalg.norm(D[k])

            # Computes the transformation matrix.
            transfmatrix = np.eye(3 * self.dm.beads.natoms) - np.dot(D.T, D)
            self.dm.dynmatrix = np.dot(
                transfmatrix.T, np.dot(self.dm.dynmatrix, transfmatrix))

        if(self.dm.asr == "crystal"):

            # Computes the centre of mass.
            com = np.dot(np.transpose(self.dm.beads.q.reshape(
                (self.dm.beads.natoms, 3))), self.dm.m) / self.dm.m.sum()
            qminuscom = self.dm.beads.q.reshape(
                (self.dm.beads.natoms, 3)) - com
            # Computes the moment of inertia tensor.
            moi = np.zeros((3, 3), float)
            for k in range(self.dm.beads.natoms):
                moi -= np.dot(np.cross(qminuscom[k], np.identity(3)),
                              np.cross(qminuscom[k], np.identity(3))) * self.dm.m[k]

            U = (np.linalg.eig(moi))[1]
            R = np.dot(qminuscom, U)
            D = np.zeros((3, 3 * self.dm.beads.natoms), float)

            # Computes the vectors along translations and rotations.
            D[0] = np.tile([1, 0, 0], self.dm.beads.natoms) / self.dm.isqm3
            D[1] = np.tile([0, 1, 0], self.dm.beads.natoms) / self.dm.isqm3
            D[2] = np.tile([0, 0, 1], self.dm.beads.natoms) / self.dm.isqm3

            # Computes unit vecs.
            for k in range(3):
                D[k] = D[k] / np.linalg.norm(D[k])

            # Computes the transformation matrix.
            transfmatrix = np.eye(3 * self.dm.beads.natoms) - np.dot(D.T, D)
            self.dm.dynmatrix = np.dot(
                transfmatrix.T, np.dot(self.dm.dynmatrix, transfmatrix))

    def apply_hpf(self):
        """
        Computes the square of frequencies, frequencies and normal modes of the system.
        Also replaces very low or imaginary frequencies by zero.
        """
        self.dm.w2, self.dm.U = np.linalg.eigh(self.dm.dynmatrix)
        self.dm.w2 = np.absolute(self.dm.w2)
        self.dm.V = np.dot(self.dm.isqM, self.dm.U.copy())
        self.dm.dynmatrix = np.tensordot(
            self.dm.w2 * self.dm.U, self.dm.U.T, axes=1)
        self.dm.w2, self.dm.U = np.linalg.eigh(self.dm.dynmatrix)
        self.dm.V = np.dot(self.dm.isqM, self.dm.U.copy())

        # Computes the tolerance and chops the frequencies.
        self.z = self.dm.w2 < self.dm.atol
        self.nz = np.logical_not(self.z)

        # Computes the square, inverse and inverse square of frequencies only for non-zero modes.
        self.dm.w2[self.z] = 0.0
        self.dm.w[self.nz] = np.sqrt(self.dm.w2[self.nz])
        self.dm.iw2[self.nz] = np.divide(1.0, self.dm.w2[self.nz])
        self.dm.iw[self.nz] = np.sqrt(self.dm.iw2[self.nz])

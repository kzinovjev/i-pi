#!/usr/bin/env python2
"""

Relies on the infrastructure of i-pi, so the ipi package should
be installed in the Python module directory, or the i-pi
main directory must be added to the PYTHONPATH environment variable.

Cuts short the output of a previous i-pi simulation, up to the
step indicated in the <step> field of the input file.
This is useful to restart a simulation that crashed.

It should be run in the same dyrectory as where i-pi was (or is being)
run, and simply fetches all information from the simulation input file.
One should also specify a directory name in which the trimmed files
will be output.

Syntax:
   trimsim.py inputfile.xml
"""
from __future__ import print_function
from __future__ import division


from builtins import range
from past.utils import old_div
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ipi.utils.io import read_file
from ipi.engine.outputs import *
from ipi.engine.properties import getkey
from ipi.inputs.simulation import InputSimulation
from ipi.utils.io.inputs import io_xml
from ipi.utils.units import unit_to_internal, unit_to_user


def kernel(x, mean=0, sigma=1):
    return np.exp(-(x - mean)**2 * (0.5 * sigma**2))


def histo(data, delta, k, mean, sigma):
    ly = delta * 0.0
    for x in data:
        ly += k(delta - x, mean, sigma)
    return ly


def get_np(path2iipi, bsize=20000, nskip=300, si=15.0, sf=-15.0, ns=10000):
    # opens & parses the i-pi input file
    ifile = open(path2iipi, "r")
    xmlrestart = io_xml.xml_parse_file(ifile)
    ifile.close()

    isimul = InputSimulation()
    isimul.parse(xmlrestart.fields[0][1])
    simul = isimul.fetch()

    # parses the temperature, the number of beads, the number of atoms, the number of target species and their masses.
    T = float(simul.syslist[0].ensemble.temp)
    P = simul.syslist[0].beads.nbeads
    natoms = simul.syslist[0].beads.natoms
    open_paths = simul.syslist[0].nm.open_paths[-1]
    m = simul.syslist[0].beads.m[open_paths]

    # initialises the data files.
    data_1 = np.zeros((bsize, 3), float)
    data_2 = np.zeros((bsize, 3), float)
    dq = np.zeros((bsize, 3), float)
    dqxgrid = np.linspace(si, sf, ns)
    dqygrid = np.linspace(si, sf, ns)
    dqzgrid = np.linspace(si, sf, ns)
    nplistx = []
    nplisty = []
    nplistz = []

    # Read the end to end distances from file
    data_path = '/home/cuzzocre/source/i-pi-mc/examples/lammps/ice-nst/P32-T269/endtoend.data'
    delta = np.loadtxt(data_path)
    step = np.shape(delta)[0]
    n_block = int(old_div(step, bsize))

    for x in range(n_block):
        dq = delta[x * bsize: (x + 1) * bsize]
        hx = histo(np.concatenate((dq.T[0], -dq.T[0])), dqxgrid, kernel, 0, np.sqrt(T * P * m))
        hy = histo(np.concatenate((dq.T[1], -dq.T[1])), dqygrid, kernel, 0, np.sqrt(T * P * m))
        hz = histo(np.concatenate((dq.T[2], -dq.T[2])), dqzgrid, kernel, 0, np.sqrt(T * P * m))

        # Defines the grid for momentum.
        pxi = old_div(-np.pi, (dqxgrid[1] - dqxgrid[0]))
        pxf = old_div(+np.pi, (dqxgrid[1] - dqxgrid[0]))
        pxstep = old_div(2 * np.pi, np.abs(dqxgrid[-1] - dqxgrid[0]))
        pxgrid = np.linspace(pxi, pxf, ns)

        pyi = old_div(-np.pi, (dqygrid[1] - dqygrid[0]))
        pyf = old_div(+np.pi, (dqygrid[1] - dqygrid[0]))
        pystep = old_div(2 * np.pi, np.abs(dqygrid[-1] - dqygrid[0]))
        pygrid = np.linspace(pyi, pyf, ns)

        pzi = old_div(-np.pi, (dqzgrid[1] - dqzgrid[0]))
        pzf = old_div(+np.pi, (dqzgrid[1] - dqzgrid[0]))
        pzstep = old_div(2 * np.pi, np.abs(dqzgrid[-1] - dqzgrid[0]))
        pzgrid = np.linspace(pzi, pzf, ns)

        # Computes the Fourier transform of the end to end vector.
        npx = np.abs(np.fft.fftshift(np.fft.fft(hx)))
        npy = np.abs(np.fft.fftshift(np.fft.fft(hy)))
        npz = np.abs(np.fft.fftshift(np.fft.fft(hz)))

        nplistx.append(npx)
        nplisty.append(npy)
        nplistz.append(npz)

    avgnpx = np.mean(np.asarray(nplistx), axis=0)
    avgnpy = np.mean(np.asarray(nplisty), axis=0)
    avgnpz = np.mean(np.asarray(nplistz), axis=0)
    normx = np.sum(avgnpx)
    normy = np.sum(avgnpy)
    normz = np.sum(avgnpz)
    errnpx = old_div(np.std(np.asarray(nplistx), axis=0), np.sqrt(n_block) / normx)
    avgnpx = old_div(avgnpx, normx)
    errnpy = old_div(np.std(np.asarray(nplisty), axis=0), np.sqrt(n_block) / normy)
    avgnpy = old_div(avgnpy, normy)
    errnpz = old_div(np.std(np.asarray(nplistz), axis=0), np.sqrt(n_block) / normz)
    avgnpz = old_div(avgnpz, normz)

    avgpsqnpx = old_div(pxgrid**2 * avgnpx, pxstep)
    errpsqnpx = old_div(pxgrid**2 * errnpx, pxstep)
    avgpsqnpy = old_div(pygrid**2 * avgnpy, pystep)
    errpsqnpy = old_div(pygrid**2 * errnpy, pystep)
    avgpsqnpz = old_div(pzgrid**2 * avgnpz, pzstep)
    errpsqnpz = old_div(pzgrid**2 * errnpz, pzstep)

    np.savetxt("np.data", np.c_[pxgrid, avgnpx, errnpx, avgnpy, errnpy, avgnpz, errnpz])
    np.savetxt("psq-np.data", np.c_[pxgrid, avgpsqnpx, errpsqnpx, avgpsqnpy, errpsqnpy, avgpsqnpz, errpsqnpz])

    psqmedx = 0.
    psqmed2x = 0.
    psqmedy = 0.
    psqmed2y = 0.
    psqmedz = 0.
    psqmed2z = 0.
    for i in range(n_block):
        psqmedx = psqmedx + old_div(np.dot(pxgrid**2, np.asarray(nplistx)[i, :]), normx)
        psqmed2x = psqmed2x + (old_div(np.dot(pxgrid**2, np.asarray(nplistx)[i, :]), normx))**2
        psqmedy = psqmedy + old_div(np.dot(pygrid**2, np.asarray(nplisty)[i, :]), normy)
        psqmed2y = psqmed2y + (old_div(np.dot(pygrid**2, np.asarray(nplisty)[i, :]), normy))**2
        psqmedz = psqmedz + old_div(np.dot(pzgrid**2, np.asarray(nplistz)[i, :]), normz)
        psqmed2z = psqmed2z + (old_div(np.dot(pzgrid**2, np.asarray(nplistz)[i, :]), normz))**2

    print('number of blocks', n_block)
    print('av_px^2', old_div(psqmedx, n_block), 'sigmax', old_div(np.sqrt((old_div(psqmed2x, n_block)) - (old_div(psqmedx, n_block))**2), np.sqrt(n_block)))
    print('av_py^2', old_div(psqmedy, n_block), 'sigmay', old_div(np.sqrt((old_div(psqmed2y, n_block)) - (old_div(psqmedy, n_block))**2), np.sqrt(n_block)))
    print('av_pz^2', old_div(psqmedz, n_block), 'sigmaz', old_div(np.sqrt((old_div(psqmed2z, n_block)) - (old_div(psqmedz, n_block))**2), np.sqrt(n_block)))

if __name__ == '__main__': get_np(*sys.argv[1:])

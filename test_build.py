import numpy as np
import sys
import os
import yaml
#import build_structure as model
from disk import disk
from grid import grid
import matplotlib.pyplot as plt
from fitsconversion import convert_to_fits


# parameter file and bookkeeping setup
modelname = 'testmodel'
if not os.path.exists(modelname):
    os.makedirs(modelname)


# set up the spatial grid and RADMC3D config files
grid = model.Grid(modelname)


# build a model structure 
diskmodel = model.DiskModel(modelname)
diskmodel.write_Model(grid)


# extract additional parameters for synthetic observations
conf = open(modelname+'.yaml')
pars = yaml.load(conf, Loader=yaml.FullLoader)
conf.close()


# raytracing
os.chdir(modelname)
os.system('rm -rf image.out')
os.system("radmc3d image incl %.2f posang %.2f " % \
          (pars["outputs"]["incl"], pars["outputs"]["PA"]) + \
          "npix %d iline %d " % \
          (pars["outputs"]["npix"], pars["setup"]["transition"]) + \
          "widthkms %.2f linenlam %d setthreads 4" % \
          (pars["outputs"]["widthkms"], pars["outputs"]["nchan"]))


# make a FITS file
outfile = modelname+'_'+pars["setup"]["molecule"]+'.fits'
os.system('rm '+outfile)
convert_to_fits('image.out', outfile, pars["outputs"]["dpc"],
                RA=pars["outputs"]["RA"], DEC=pars["outputs"]["DEC"])

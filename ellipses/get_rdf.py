from scipy.special import j1
from numpy import histogram, pi, power, rint, arange
from numpy.linalg import norm
from numpy import trapz, stack, savetxt
import sys
#wp: to access the analsysis tool folder
sys.path.insert(0, '/home/pineros/work/pca_tests/')
from analysis_tools.read_gsd import ReadGSD
from analysis_tools.rdf import RDF2D 

#wp:access GSD files from directory given in 'file_base' variable
file_base='/home/pineros/work/pca_tests/ellipsoids'
eta1=0.8000; eta2=0.85001; eta_step=0.025;
etas=arange(eta1,eta2,eta_step)


for eta in etas[1:]:
    filetraj = "{}/trajectory_{:.4f}.gsd".format(file_base, eta)
    #wp: frames is a directory with diameters, type and coordination entries
    frames = ReadGSD(filetraj, shuffle_data=True, randomize=False) 
    #wp: Get rdf
    dr=0.025
    r,gr = RDF2D(frames,dr); 
    #combine them as columns
    rdf=stack([r,gr],axis=1); 
    #write them to a text file
    filename="{}/rdf_{}.dat".format(file_base,eta)
    savetxt(filename,rdf); 
    print "rdf has been calculated and written to {}".format(filename) 

#-----end

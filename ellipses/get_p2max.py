import numpy as np
import sys
#wp: to access the analsysis tool folder
sys.path.insert(0, '/home/pineros/work/pca_tests/')
from analysis_tools.read_gsd import ReadGSD
from analysis_tools.p2max import P2Max 

#density range to compute
etas=np.arange(0.6100,0.90001,0.01); 

#store p2max for all density range
p2max_etas=etas*0.0;
index=0
#for eta in etas:
for eta in etas:
    #wp:access GSD files from directory given in 'file_base' variable
    #file_base='/home/pineros/work/pca_tests/ellipsoids/kappa_4.0'
    file_base='kappa_4.0_3N'
    filename = "{}/trajectory_{:.4f}.gsd".format(file_base, eta)
    print "Now doing p2max calc for eta {}".format(eta)
    #wp: frames is a directory with diameters, orientation type and coordination entries
    frames = ReadGSD(filename, shuffle_data=False, randomize=False)
    #frames2 = ReadGSD(filename, shuffle_data=True, randomize=False)
    #print "--------------------no shuffle"
    #print frames[0]['orientation'][0:5]
    #print frames[0]['coords'][0:5]
    #print "--------------------shuffle"
    #print frames2[0]['orientation'][0:5]
    #print frames2[0]['coords'][0:5]
    #wp: Calculate p2max 
    p2max = P2Max(frames);
    #print "p2max for eta {} is {}".format(eta,p2max)
    p2max_etas[index]=p2max
    index=index+1

#save as text file
filename_p2max='p2max.dat'
#using axis=-1 argument to stack as columns
p2max_data=np.stack((etas,p2max_etas),axis=-1)
np.savetxt(filename_p2max,p2max_data)

#werken:03/21/18-----end

import numpy as np
import numpy.linalg as npl 

'''Calculates the nematic order parameter of an ellipse as defined in Bautista et al JchemPhys 140,204502 given as:
	p2max = sqrt( < 1/N sum( cos(2*theta)) >^2 +< 1/N sum( sin(2*theta)) >^2 ) 
'''

#Must have an 'orientation' attribute defined in frames
def P2Max(frames): 
    M = len(frames)
    N = float(len(frames[0]['coords']))
   
    f_count=0; csum_f=0; ssum_f=0; 
    #loop over the frames
    for frame in frames:
        f_count=f_count+1;
        orientations =frame['orientation']; 
        ssum=0; csum=0;

        #loop over each particle for ensemble average
        for orientation in orientations:
           #orientation in 2D returns q(0)=cos(theta/2) and q(3)=sin(theta/2) in quaternion; rest are zeros
           cos_half=orientation[0]; sin_half=orientation[3]; 
           cos_two=-4.0*(cos_half*sin_half)**2 + (cos_half**2-sin_half**2)**2
           sin_two=4.0*cos_half*sin_half*(cos_half**2-sin_half**2)

           #add trig terms  
           csum=csum+cos_two
           ssum=ssum+sin_two

        #Add them for the frame average
        csum_f=csum_f+csum/N
        ssum_f=ssum_f+ssum/N
        
    #calculate final averages for cos and sin terms
    c_ave=csum_f/f_count; s_ave=ssum_f/f_count; 
    #compute p2max
    p2max=np.sqrt(c_ave**2+s_ave**2)

    return p2max
       
#Werken 03/21/18 

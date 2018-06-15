
# coding: utf-8

# Find roots or maxima of order parameters to determine isotropic/nematic transition
# ==================

# In[1]: 
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


# ### Use regular order parameter sign change to determine nematic to liquid transition

# In[30]: 

#***Computes nematic transition for a list of trajectories with different aspect ratio kappa
kappa_list=[3,4,6,9]
#kappa_list=[4]
nematic_list=[]
etas_list=[]

folder='300_features_30k_3N_fine'
tail_name='_30k_300_3N.dat'

for k in kappa_list:
    #***Import data as arrays
    #if k != 3:
    data=np.loadtxt('ellipsoids/analysis/'+folder+'/OPs_kappa'+str(k)+tail_name)
    #else:
    #    data=np.loadtxt('ellipsoids/analysis/'+folder+'/OPs_kappa'+str(k)+'_60k_100_0.01-0.05.dat')
    etas=data[::1,0]; OPs=data[::1,1:]; 
    
    
    #saves etas for std later
    etas_list.append(etas)

    #***Use cubic interpolation on order parameter and later find the root
    #pick order parameter and fit spline
    op_index=0;
    op=OPs[:,op_index]

    #returns object that contains tri-tuple of interpolated data (t,coefficients,k); s= smoothing 
    tck = interpolate.splrep(etas, op, s=0)
    
    #plots order parameter for reasons
    zero_line=etas*0.0;
    #plt.plot(etas,op,'-o')
    ##plt.plot(etas,zero_line,'--')
    #plt.rcParams["figure.figsize"] = [12,8];
    #plt.xlabel('$\eta$',fontsize=40);
    ##plt.ylabel(r'$\langle P_{1} \rangle$',fontsize=40);
    #plt.ylabel(r'$ P_{1} $',fontsize=40);
    #plt.tick_params(axis='both', labelsize=30)
    #plt.rcParams['xtick.direction'] = 'in'; plt.rcParams['ytick.direction'] = 'in';
    #plt.tight_layout()
    ##plt.savefig('figures/OP_kappa_'+str(k)+'.svg',dpi=fig.dpi)
    #
    #plt.show()
    

    #****Finds the root using built-in function
    #Will return *all* roots within the original interval
    root=interpolate.sproot(tck)
    print "The nematic order transition judged by the change of sign is {}".format(root[0])
    
    #***Saves in list
    nematic_list.append([k,root[0]])

    
#das it maen


# ### Alternatively, find maximum from the standard deviation (a type of susceptibility)
# In[29]: 
nematic_std_list=[]
k_index=0
tail_name='_30k_300_std_3N.dat'

for k in kappa_list:
    #***Import stds
    #if k != 3:
    data_std=np.loadtxt('ellipsoids/analysis/'+folder+'/OPs_kappa'+str(k)+tail_name)
    #else:
    #    data_std=np.loadtxt('ellipsoids/analysis/'+folder+'/OPs_kappa'+str(k)+'_60k_100_0.01-0.05_std.dat')
    OPs_std=data_std;
    
    #retrieve proper etas corresponding to kappa
    etas=etas_list[k_index]
    k_index+=1
    
    #****Interpolate std then use maximum on a finer grid
    #pick order parameter and fit spline
    coarse=1;
    etas_coarse=etas[::coarse]

    #First column contains the eta values
    op_index=0;
    op_std=OPs_std[::coarse,op_index]

    #provide existing data to interpolate
    tck_std = interpolate.splrep(etas_coarse, op_std, s=0)

    #pick new, finer range
    deta=0.001;
    etas_fine=np.arange(etas[0],etas[-1:],deta);
    #get interpolation: note that the splev attribute function is used; der=derivative
    op_std_fine=interpolate.splev(etas_fine, tck_std, der=0)
    
    #***Check plot visually  
    #plt.plot(etas_coarse,op_std,'-o')
    ##plt.plot(etas_coarse,op_std,'ko')
    ##plt.plot(etas_fine,op_std_fine, linewidth = 2.0,label='spline' )
    #
    #plt.rcParams["figure.figsize"] = [12,8]
    #plt.tick_params(axis='both', labelsize=30)
    #plt.rcParams['xtick.direction'] = 'in'; plt.rcParams['ytick.direction'] = 'in';
    #plt.legend(fontsize=25)
    #
    #plt.xlabel('$\eta$',fontsize=40);
    #plt.ylabel(r'$\sigma_{1}$',fontsize=40);
    #plt.tight_layout()
    #plt.savefig('figures/OP_kappa_'+str(k)+'_std.svg',dpi=fig.dpi)
    #
    #plt.show()
    
    #****Now to get maximum from interpolated data
    #max_index=np.argmax(op_std_fine);
    #max_eta=etas_fine[max_index]
    
    #***Alternatively, use unsplined data for comparison
    max_index=np.argmax(op_std);
    max_eta=etas_coarse[max_index]

    print "The nematic order transition judged by maximum susceptibility is {}".format(max_eta)
    
    #saves for later inspection
    nematic_std_list.append([k,max_eta])
    
#das it mein


# ### Plot nematic boundaries against known values by Xu et al

# In[19]: 
#***Model fitted to function phin=phi0/(k0+k) from xu et al J. Chem. Phys. 139, 024501 (2013)

phi0=6.37; k0=5.14;
kappas=np.arange(2.9,9.1,0.01);
phin=phi0/(k0+kappas);

#Data from PCA
nematic_array=np.asarray(nematic_list)
nematic_std_array=np.asarray(nematic_std_list)


# In[20]: 
#***Plot PCA boundary values against Xu model

#**Using order parameters
#plt1,=plt.plot(kappas,phin,linewidth = 2.0, label='Xu fit')
#plt2,=plt.plot(nematic_array[:,0],nematic_array[:,1],'ko',markersize=5, label='PCA')
#plt.legend(handles=[plt1, plt2])
#
##***plot formatting 
#plt.xlabel('Ellipse Aspect Ratio',fontsize=18);
#plt.ylabel('Fluid-Nematic Boundary (density)',fontsize=18);
#plt.title('PCA order parameter',fontsize=18)
#plt.rcParams["figure.figsize"] = [12,8]
#plt.tick_params(axis='both', labelsize=15)
#plt.ylim([0.4,0.8]);
#
##***Saves figure?
#fig=plt.figure(1)
#fig.savefig('figures/OP_nematic_boundary.pdf')


# In[22]: 
#***Plot PCA boundary values against Xu model 
#**Using order parameters
#plt1,=plt.plot(kappas,phin,linewidth = 2.0, label='Xu et al')
#plt2,=plt.plot(nematic_std_array[:,0],nematic_std_array[:,1],'ko',markersize=5, label='$max(\sigma_1$)')
#print(nematic_std_array[:,1])
#plt.legend(handles=[plt1, plt2],fontsize=25)
##plt.legend() #wp: also valid without explicitly referencing the handles
#
##***plot formatting 
#plt.xlabel('$\kappa$',fontsize=40);
#plt.ylabel('$\eta$',fontsize=40);
##plt.title('PCA order parameter',fontsize=24)
#plt.rcParams["figure.figsize"] = [12,8]
#plt.rcParams['xtick.direction'] = 'in'; plt.rcParams['ytick.direction'] = 'in';
#plt.tick_params(axis='both', labelsize=30)
##plt.minorticks_on()-
#plt.ylim([0.4,0.8]);
#
##***Saves figure?
#fig=plt.figure(1)
##fig.savefig('figures/OP_nematic_std_boundary.eps',dpi=fig.dpi)

#saves to text
xu_fit=np.vstack((kappas,phin));
np.savetxt('eta_max_nematic.dat',nematic_std_array);
np.savetxt('xu_fit_nematic.dat',xu_fit.T); 


# ### Positional feature vector comparison 
# In[10]: 
#***Gets phase boundaries from OP std above and turns it into a dictionary
nematic_dict={}
for k,pb in nematic_std_array:    
    nematic_dict[str(int(k))]=pb


# In[48]:


#***Computes nematic transition for a list of trajectories with different aspect ratio kappa
kappa_list=[4]

folder='300_features_30k_3N_fine'
tail_names={'4': '_30k_300_3N.dat', '9': None}
folder2='300_features_30k_3N'
tail_names2={'4': '_pos_30k_300_3N.dat', '9': None}
etas_list=[]; etas_list2=[]; 

for k in kappa_list:
    #***Import data as arrays
    strk=str(k)
    data=np.loadtxt('ellipsoids/analysis/'+folder+'/OPs_kappa'+strk+tail_names[strk])
    data2=np.loadtxt('ellipsoids/analysis/'+folder2+'/OPs_kappa'+strk+tail_names2[strk])
    
    etas=data[:,0]; OPs=data[:,1:]; etas2=data2[:,0]; OP2s=-data2[:,1:];  
    
    #saves etas for std later
    etas_list.append(etas); etas_list2.append(etas)

    #***Use cubic interpolation on order parameter and later find the root
    #pick order parameter and fit spline
    op_index=0;
    op=OPs[:,op_index]
    op2=OP2s[:,op_index]

    #adds phase boundaries
    #--nematic
    #blue='#1f77b4'; orange='#ff7f0e'; grey='#f0f0f0'
    #x1=nematic_dict[strk]; x2=x1; y1=0; y2=1
    ##plt.plot((x1, x2), (y1, y2), '--',color='k')
    #plt.axvspan(x1, x2, alpha=0.5, color=blue)
    ##--solid
    #x1=0.8+1./60.; x2=x1+1./60;
    #plt.axvspan(x1, x2, alpha=0.9, color=grey)
    
    #plots order parameter for reasons
    #plt.plot(etas,(op-op[0])/abs(np.max(op-op[0])),'-o',label="orientation")
    #plt.plot(etas2,(op2-op2[0])/abs(np.max(op2-op2[0])),'-o',label="position")
   
    #formatting
    #plt.rcParams["figure.figsize"] = [12,8]
    #plt.xlabel('$\eta$',fontsize=40);
    #plt.ylabel(r'$ P_{1}^*$',fontsize=40);
 
    #plt.rcParams['xtick.direction'] = 'in'; plt.rcParams['ytick.direction'] = 'in';
    #plt.tick_params(axis='both', labelsize=30)
    #plt.legend(fontsize=18,loc='upper left')
    #plt.tight_layout()
    #
    ##plt.savefig('figures/OP1_pos_vs_orientation_kappa_'+str(k)+'.svg',dpi=fig.dpi)
    #
    #plt.show() 

    #saves text file
    orient=np.vstack((etas,(op-op[0])/abs(np.max(op-op[0]))))
    pos=np.vstack((etas2,(op2-op2[0])/abs(np.max(op2-op2[0]))))
    np.savetxt('orientation_kappa_4_norm.dat',orient.T);
    np.savetxt('pos_kappa_4_norm.dat',pos.T);

#das it maen


# In[49]:


#***Computes nematic transition for a list of trajectories with different aspect ratio kappa
kappa_list=[4]

folder='300_features_30k_3N_fine'
tail_names={'4': '_30k_300_std_3N.dat', '9': None }
folder2='300_features_30k_3N'
tail_names2={'4': '_pos_30k_300_std_3N.dat', '9': None }
k_index=0;
coarse=1;

for k in kappa_list:
    #***Import data as arrays
    strk=str(k)
    data=np.loadtxt('ellipsoids/analysis/'+folder+'/OPs_kappa'+strk+tail_names[strk])
    data2=np.loadtxt('ellipsoids/analysis/'+folder2+'/OPs_kappa'+strk+tail_names2[strk])
    
    #retrieve proper etas corresponding to kappa
    etas=etas_list[k_index]
    etas=etas[::coarse]
    etas2=etas2[::coarse]
    k_index+=1
    
    OPs=data[:,:]; OP2s=data2[:,:];  

    #***Use cubic interpolation on order parameter and later find the root
    #pick order parameter and fit spline
    op_index=0;
    op=OPs[::coarse,op_index]
    op2=OP2s[::coarse,op_index]
    
    #adds phase boundaries
    #--nematic
    #blue='#1f77b4'; orange='#ff7f0e'; grey='#f0f0f0'
    #x1=nematic_dict[strk]; x2=x1; y1=0; y2=1
    ##plt.plot((x1, x2), (y1, y2), '--',color='k')
    #plt.axvspan(x1, x2, alpha=0.5, color=blue)
    ##--solid
    #x1=0.8+1./60.; x2=x1+1./60;
    #plt.axvspan(x1, x2, alpha=0.9, color=grey)
    #
    ##plots order parameter for reasons
    #plt.plot(etas,op/max(op),'-o',label="orientation")
    #plt.plot(etas2,op2/max(op2),'-^',label="position")
    ##plt.plot(etas,zero_line,'--')
    #plt.rcParams["figure.figsize"] = [12,8]
    #plt.rcParams['xtick.direction'] = 'in'; plt.rcParams['ytick.direction'] = 'in';
    #plt.xlabel('$\eta$',fontsize=40);
    #plt.ylabel(r'$\sigma_{1}/max(\sigma_{1}) $',fontsize=40);  
    #plt.tick_params(axis='both', labelsize=30)
    #plt.legend(fontsize=18,loc='upper left')
    #plt.tight_layout()
    #
    #plt.savefig('figures/OP_pos_vs_orientation_kappa_'+str(k)+'.svg',dpi=fig.dpi)
    #plt.show()

    #saves text file
    orient=np.vstack((etas,op/max(op)))
    pos=np.vstack((etas2,op2/max(op2)))
    np.savetxt('orientation_kappa_4_std_norm.dat',orient.T);
    np.savetxt('pos_kappa_4_std_norm.dat',pos.T);

#das it maen


# Unrelated to PCA, just the normal order parameter as reported by Bautista
# ======================= 
#{# In[53]: 
#{folder='ellipsoids/analysis/p2max/'
#{filename='kappa_4.0_3N_p2max.dat'
#{p2max=np.loadtxt(folder+filename)
#{
#{#plots order parameter for reasons
#{plt.plot(p2max[:,0],p2max[:,1],'-o')
#{
#{#plt.plot(etas,zero_line,'--')
#{plt.rcParams["figure.figsize"] = [12,8]
#{plt.rcParams['xtick.direction'] = 'in'; plt.rcParams['ytick.direction'] = 'in';
#{plt.xlabel('$\eta$',fontsize=40);
#{plt.ylabel(r'$P_{2}^{max}$',fontsize=40);  
#{plt.tick_params(axis='both', labelsize=30)
#{#plt.legend(fontsize=25)
#{    
#{plt.savefig('figures/p2max_kappa_4_'+str(k)+'3N.svg',dpi=fig.dpi)
#{plt.show()
#{
#{
#{# In[222]:
#{
#{
#{#****Comparing p2max to P1 from PCA
#{folder='100_features_20k_samps_extended'
#{tail_names={'4': '_20k_100_solid.dat', '9': '_20k_100_solid.dat'}
#{
#{k=4
#{#***Import data as arrays
#{strk=str(k)
#{data=np.loadtxt('ellipsoids/analysis/'+folder+'/OPs_kappa'+strk+tail_names[strk])
#{    
#{etas=data[:,0]; OPs=data[:,1:];  
#{    
#{op_index=0;
#{op=OPs[:,op_index]
#{
#{
#{plt.plot(p2max[:,0],p2max[:,1],'-^',label='$P_{2}^{max}$')
#{plt.plot(etas,(op-op[0])/abs(np.max(op-op[0])),'-o',label='<$P_1$>*')
#{plt.legend(fontsize=25)
#{

import hoomd
import hoomd.hpmc as hpmc
from numpy import pi, arange, sqrt
from mpi4py import MPI
import sys,getopt

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#---------Arg Input-----------
#wp: omit first index since it's just the filename
argv=sys.argv[1:]

try:
   #wp: arguments, and command-line values. If it has ":" value argument is required
   #wp: brackets indicate extra options (optional, not required)
   opts, args = getopt.getopt(argv,"hk:e:p:P:n:",["kappa=","eqtime=","prodtime=","period=","ncells="])
except getopt.GetoptError:
   print 'Need proper input. Check -h. Exitting...'
   sys.exit(2)

for opt, arg in opts:
   if opt == '-h':
      print 'I take k, eq (-e) time, prod (-p) time. Elz, take biz elzwhere' 
      sys.exit()
   elif opt in ("-k","--kappa"):
      kappa = float(arg) 
   elif opt in ("-e","--eqtime"):
      eqtime = int(arg) 
   elif opt in ("-p","--prodtime"):
      prodtime = int(arg) 
   elif opt in ("-P","--period"):
      Period = int(arg) 
   elif opt in ("-n","--ncells"):
      ny = int(arg) 

#---------HOOMD and run simulation-----------
hoomd.context.initialize("--mode=cpu")

#wp:Compression function
def Compress(eta_tgt, scale=0.999): 
    #perform stuff based on rank (wp: master core)
    if rank == 0:
        #get current state #wp: get integrator data for ellipse
        snap = system.take_snapshot(integrators=True)
        N = len(snap.particles.diameter)
	#wp: apparently N*area i.e. area of all objects. For ellipsoids, A=pi*a*b 
        #Vp = sum(pi*(snap.particles.diameter**2))
	a_ellipse=mc.shape_param['A'].a;
	b_ellipse=mc.shape_param['A'].b;
        Vp = N*(pi*(a_ellipse*b_ellipse))
        Vb = system.box.get_volume()
    else:
        snap = system.take_snapshot()
        N = None
        Vp = None
        Vb = None
    
    #broadcast from 0th rank #wp: to all other cores 
    N = comm.bcast(N, root=0)
    Vp = comm.bcast(Vp, root=0)
    Vb = comm.bcast(Vb, root=0)
    
    #assign variables
    #wp:current eta=N*pi*sigma^2/Vbox
    eta = Vp/Vb
    eta_init = eta
    
    if rank == 0: 
        print '\nStarting compression from eta={}\n'.format(eta)
    
    #calculate new quantities
    Vb_tgt = Vp/eta_tgt
    
    #box compression loop: #wp: slowly reduce volume while getting rid of overlaps
    while Vb > Vb_tgt:
        Vb = max(Vb*scale, Vb_tgt)
        eta = Vp/Vb
	#print "Currently compressing at {} eta\n".format(eta)
        new_box = system.box.set_volume(Vb)
        hoomd.update.box_resize(Lx=new_box.Lx, Ly=new_box.Ly, Lz=new_box.Lz, period=None)
        overlaps = mc.count_overlaps()
        
        #run until all overlaps are removed
        while overlaps > 0:
            hoomd.run(100, quiet=True)
            overlaps = mc.count_overlaps()
            
    if rank == 0: 
        print "Compressed to eta={}\n".format(eta)
#wp:End Compression function
    
#wp: aspect ratio of ellipsoid
if rank == 0:
	print "Simulation for ellipsis aspect ratio {}".format(kappa)
#wp: ellipsoid radius in x and y axis
a=0.5; b=kappa*a;
etas=arange(0.5400, 0.60001, 0.005)
#etas= arange(0.3000, 0.54501, 0.005)

#wp: eta = pi*sigma**2*rho, where rho=N/V and sigma=radius
#wp: for ellipsoid, eta= pi*kappa*a**2*density

#wp: rectangular box size factor assuming when reducing along the a_cell_x direction 
area_factor=0.5*kappa;
#wp: 2 particles per cell in hex 
Ncell=2.0;
#wp: scale cell by eta factor; Start eta lower than eta[0] to avoid overlapped conf
start_eta=etas[0]*0.99;
a_cell_scale = ((pi*kappa*a**2.0)/start_eta/area_factor*Ncell)**(1.0/2.0)
#print('the scaled axis is ', a_cell_scale);
#wp: number of cells to be used in each direction chosen such that it creates a square box relative to cell aspect ratio 
#ny=34 #wp: given in input
nx=int(round(ny*(kappa/2.0)))
n=[nx,ny];    
    
#wp: Choosing rectangular box for ellipsoid scaling for eta and area_factor
a_cellx=a_cell_scale;
a_celly=kappa/2.0*a_cellx;

#print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
#print "----------Lx and Ly are {} and {}".format(a_cellx*nx,a_celly*ny)
#print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

#wp: a3 must be [0.,0,1] for 2D lattices
user_cell_hex=hoomd.lattice.unitcell(
	N=2, dimensions=2,
	a1=[a_cellx,0.,0.],
	a2=[0.,a_celly,0.],
	a3=[0.,0.,1.],
	position=[[0.,0.,0.],[0.5*a_cellx,0.5*a_celly,0.]] );

#wp: Create user defined lattice system
system = hoomd.init.create_lattice(unitcell=user_cell_hex, n=n)

#wp: d = displacement size, a = rotation size, move ratio =0.5 by default
mc = hpmc.integrate.ellipsoid(d=0.1, a=0.1, seed=1234, move_ratio=0.5)
mc.shape_param.set('A', a=a, b=b, c=a);
#print('ellipsoids parameters (a,b,c) = ', mc.shape_param['A'].a, mc.shape_param['A'].b, mc.shape_param['A'].c)
    
#wp:originally 3M each
equil_steps = eqtime #500000
prod_steps = prodtime #500000 
#wp: originally 3M/1000
period = prod_steps/Period

#wp:---------MC for every density point
for eta in etas:
    print '\n{} from rank {}\n'.format(eta, rank)
    #compress and then equilibrate
    hoomd.run(1) #wp: To execute necessary hoomd system info before compression
    Compress(eta)
    hoomd.run(equil_steps) 

    #tuner = hoomd.hpmc.util.tune(obj=mc, tunables=['d', 'a'], max_val=[0.5, 2*pi/4.], target=0.2); 
    #for i in range(80):
    #    hoomd.run(100, quiet=True);
    #    tuner.update(); 

    #set up the file writer and run the production version
    d = hoomd.dump.gsd("trajectory_{:.4f}.gsd".format(eta), 
                           period=period, group=hoomd.group.all(), overwrite=True) 
    hoomd.run(prod_steps)
    #overlaps = mc.count_overlaps()
    #print "number of overlaps", overlaps
    #print "eta ", eta 

    #disable old file writer
    d.disable()    
    
    if rank == 0:
       print "Simulation over for eta {}".format(eta)

comm.Disconnect()

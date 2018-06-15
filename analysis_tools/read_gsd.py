import gsd.hoomd
from numpy import mean, array, unique, concatenate,array_split,pi,cos,sin
from numpy.random import shuffle, rand
from copy import deepcopy

def ReadGSD(filename, shuffle_data=True, randomize=False):
    frames = []
    #wp: open hoomd trajectory in read mode
    traj = gsd.hoomd.open(name=filename, mode='rb')
    
    #static quantities
    box = traj[0].configuration.box
    D = traj[0].configuration.dimensions
    #wp:number of particles
    N = len(traj[0].particles.position)

    #wp: tracks frames number    
    frame_num=0;
    #loop over the configurations and stor in new format #wp: for each frame
    for snap in traj:
        #dynamic quantities
        diameters = snap.particles.diameter
        coords = snap.particles.position
        #wp: checks to see if it has orientation
        orientation_flag=False
        if hasattr(snap.particles, 'orientation'): 
            orientation_flag=True 
            #wp: includes orientation 
	    orientations = snap.particles.orientation
        
        #check for a square or rect box 
        if (max(box[0:D]) - min(box[0:D]))/mean(box[0:D]) > 0.000000000001:
            rectbox=True
            L  = box[0];  Ly = box[1];
            #raise Exception('Not a rectangle or square!!!')
        else:
	    rectbox=False
            L = box[0]; 
        
        #get the particle types
        possible_types = snap.particles.types
        types = array([possible_types[type_id] for type_id in snap.particles.typeid])
        
        #replace with random positions if randomize is selected (for comparing to randomized PCA result and useful information content)
        if randomize:
            if rectbox: 
		coords = rand(N,D)
            	coords[:,0] = L*coords[:,0] - L/2.0
            	coords[:,1] = Ly*coords[:,1] - Ly/2.0
	    else:
            	coords = L*rand(N, D) - L/2.0

            if orientation_flag:
                #wp: generate random *angles* then convert to half ang orientations
                orientations=4.0*pi*rand(N,4)-2.0*pi
                orientations[:,0]=cos(orientations[:,1]/2.0)
                orientations[:,3]=sin(orientations[:,1]/2.0) 
                #wp:clears inner columns for 2D. 1:3 --> col1,col2
                orientations[:,1:3]=0.0
        
        #create our new data structure and shift to upper right quadrant 
        #wp:dictionary type
        #frames.append({'coords': (coords[:,0:D]+L/2.0), 'diameters': diameters, 'types': types, 'L': L, 'D': D})
        frames.append({'diameters': diameters, 'types': types, 'L': L, 'D': D})
        if orientation_flag:
            #wp: The property must be added into the dictionary of the proper frame
	    #wp: not as its own 'dictionary' that screws up the frame count
            frames[frame_num]['orientation'] = orientations
        if rectbox:
            frames[frame_num]['Ly'] = Ly 
            #wp: correct coords to + quadrant
            coords[:,0]+=L/2.0; coords[:,1]+=Ly/2.0;
            frames[frame_num]['coords']=coords[:,0:D]
        else:
            frames[frame_num]['coords']=coords[:,0:D]+L/2.0

        #wp:update frame number
        frame_num+=1

    #wp: end for trajectory snap
  
    #perform random shuffle of identical particles coordinates to help facilitate learning    
    if shuffle_data:
        shuffled_frames = []
        #wp: tracks frames number    
        frame_num=0;
	#wp: for the dictionary of a trajectory snapshot defined above
        for frame in frames:
            #extract local copies for organizational convenience
            coords = frame['coords']
            diameters = frame['diameters']
            types = frame['types']
            L = frame['L']
            D = frame['D'] 

            if rectbox:
                Ly=frame['Ly']

            if orientation_flag:
                orientations_f= frame['orientation']
                #concatenate them along the 'column' 1 direction so that orientations aren't shuffled independent of position etc 
                coords_orientations = concatenate((coords,orientations_f),axis=1) 
               
            #prepare for shuffle
            coords_shuffled = None
	    #wp:returns array, indeces, and frequency in which each input was repeated. Axis=None means the array is flattened
            unique_types, start, count = unique(types, return_index=True, return_inverse=False, return_counts=True, axis=None)
            start__end = zip(start, start+count)
            
            #check for errors
            if len(start__end) != len(unique_types):
                raise Exception('Bad data!!!')
            
            #do the shuffling #wp: for each unique type
            for start, end in start__end: 
                #wp: deepcopy creates a brand new structure and everything in it, not merely 'points' to the same object
                if orientation_flag:
                    grouped = deepcopy(coords_orientations[start:end])
                    shuffle(grouped) 
                else:
                    grouped = deepcopy(coords[start:end])
                    shuffle(grouped)

                if coords_shuffled is not None:
                    #wp:concatenates arrays along the 'row' or 0 dimension
		    #wp: i.e. it stacks them for each particle type
                    coords_shuffled = concatenate((coords_shuffled, grouped), axis=0)
                else:
                    coords_shuffled = deepcopy(grouped)

            #wp: appends basic info
            shuffled_frames.append({'diameters': array(diameters), 'types': array(types), 'L': L, 'D': D})

            if orientation_flag:
                #wp: once shuffled, break back into coords and orientations arrays; Note that [D] indicates the index at which the array is to be sliced for appropriate axis
                #wp: shuffle_split contains a list of arrays indexed as below
                shuffle_split=array_split(coords_shuffled,[D],axis=1); 
                #shuffled_frames.append({'coords': shuffle_split[0], 'orientation': shuffle_split[1]})
                shuffled_frames[frame_num]['coords']=shuffle_split[0]
                shuffled_frames[frame_num]['orientation']=shuffle_split[1]
            else: #wp: no orientations
                #shuffled_frames.append({'coords': array(coords_shuffled)})
                shuffled_frames[frame_num]['coords']=array(coords_shuffled)
        
            if rectbox:
                shuffled_frames[frame_num]['Ly'] = Ly
            
            #wp:update frame number
            frame_num+=1 

        #set the data
        frames = shuffled_frames
        shuffled_frames = []    
        
    return frames

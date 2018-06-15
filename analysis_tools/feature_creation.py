from copy import deepcopy
from numpy import power, rint, sqrt, sum, unique, repeat, arccos, dot, transpose, append, cos, sin, hstack, maximum, minimum, pi, array, asarray, mod, mean,std, zeros, unique,vstack
from numpy.linalg import norm
from numpy.random import shuffle
from sklearn import preprocessing
import warnings

#################################################################################################################################
#################################################################################################################################
#################################################################################################################################

#this generates NN features for the PCA analysis (or some other machine learning method as well)
def FrameToFeatures(frame, N_nn, method, particle_inc, nn_inc):
    #extract some relevant frames level details #wp: D=dimension
    coords = deepcopy(frame['coords'])
    D = float(frame['D'])
    N = float(len(coords))
    diameters = frame['diameters']
    diameters_scaled = frame['diameters']/frame['diameters'][0]
    #wp:
    if  'Ly' in frame:
	rectbox=True
        V = frame['L']*frame['Ly']
    else:
        rectbox=False
        V = power(frame['L'], D)

    normalizing_distance = power(V/N, 1.0/D)

    #wp
    if 'orientation' in method:
       orientations=frame['orientation'] 

    
    frame_features = []
    combined_data = zip(coords, diameters)
    #wp:for loop below obtains particle and diameter from `combined_data' in increments of 'particle_inc'
    for particle, diameter in combined_data[0::particle_inc]:
        #nearest neighbor coordinate wrapping
        #wp:subtracting an array from a scalar automatically does it for every member of the array i.e. returns the array subtraction
        Rpj_v = particle - coords
        #wp: periodic boundary conditions:
        #wp: 'rint' follows a 'round to even integer' rule--does this matter?
        if rectbox:
            Rpj_v[:,0] = Rpj_v[:,0] - rint(Rpj_v[:,0]/frame['L'])*frame['L']
            Rpj_v[:,1] = Rpj_v[:,1] - rint(Rpj_v[:,1]/frame['Ly'])*frame['Ly']
	else:
            Rpj_v = Rpj_v - rint(Rpj_v/frame['L'])*frame['L']

        #wp: r^2 for each coordinate?
        Rpj = (sqrt(sum(power(Rpj_v, 2.0), axis=1)))     
        
        #generate statistics for various nearest neighbors
        sorter = Rpj.argsort()
        #wp: sorts them in ascending order using the sorted index
        Rpj = Rpj[sorter[::1]] 
       
	#wp: return sorted orientation based on nearest neighbors
        if 'orientation' in method:
            orientations_sorted=orientations[sorter[::1]] 
            #wp: returns sin(t/2) from orientation
            #orientations_sorted=orientations_sorted[:,3]
            #wp: computes cos(theta)
            orientations_cos=orientations_sorted[:,0]**2-orientations_sorted[:,3]**2 
            orientations_sin=2.0*orientations_sorted[:,0]*orientations_sorted[:,3] 
	    #wp: returns |cos(alpha)| where alpha=theta1-theta2 i.e. relative orientns
            cs0=orientations_cos[0]; sn0=orientations_sin[0];
            csi=orientations_cos; sni=orientations_sin;
            orientations_sorted=abs(cs0*csi+sn0*sni)
            #orientations_sorted=(cs0*csi+sn0*sni)**2
	    #wp: returns |sin(alpha)| where alpha=theta1-theta2 i.e. relative orientns
            #orientations_sorted=abs(sni*cs0-csi*sn0)
            
	
        #for composition calculations
        if 'composition' in method:
            Dpj = diameters_scaled[sorter[::1]]
        
        #chosen axis for measuring angles based on nearest neighbor
        if 'angular' in method:
            #sort the particle-particle vectors according to distance
            Rpj_v = Rpj_v[sorter[::1]]
            
            #normalize all of the vectors
            Rpj_v = Rpj_v/maximum(norm(Rpj_v, axis=1), 1.0e-10)[:,None]
            
            #find the unit x and y axis to base all angular details on using the 1st and 2nd nearest neighbors
            x_axis = Rpj_v[1] #this specifies the x direction to measure the angular elevation from
            y_axis = Rpj_v[2] - array([dot(Rpj_v[2], Rpj_v[1]), 0.0]) #this specifies the upper two quadrants
            y_axis = y_axis/norm(y_axis)
            
            #compute the raw angle between 0-pi that has unresolved upper and lower quadrants
            x_axis_T = transpose(x_axis)
            y_axis_T = transpose(y_axis)
            Tpj = arccos(minimum(maximum(dot(Rpj_v, x_axis_T), -0.9999999), 0.9999999))
            
            #determine if the vector points into the upper quadrant defined by the second nearest neighbor
            LQ = arccos(minimum(maximum(dot(Rpj_v, y_axis_T), -0.9999999), 0.9999999)) > pi/2.0
            
            #adjust the angle to upper or lower quadrants 
            Tpj[LQ] = 2.0*pi - Tpj[LQ]
        
        #possible feature options
        feature_batch = []
        if 'distance' in method:
            #wp: scale distances by number density and adds them to new list 'feature_batch'; Note that the first distance=0 is not included
            feature_batch.extend((Rpj[1:N_nn+1]/normalizing_distance)[0::nn_inc])
	#wp:save sorted orientations based on nearest neighbor
        if 'orientation' in method:
            #wp: note that the orientation of particle itself is included
            #feature_batch.extend(orientations_sorted[1:N_nn+1][nn_inc::nn_inc])
            feature_batch.extend(orientations_sorted[1:N_nn+1][0::nn_inc])
            
        if 'composition' in method:
            feature_batch.extend(Dpj[1:N_nn+1][0::nn_inc])
        if 'angular' in method:
            feature_batch.extend(append(cos(Tpj[2:N_nn+1])[0::nn_inc], 
                                        sin(Tpj[2:N_nn+1])[0::nn_inc], axis=0))
            
        frame_features.append(feature_batch)

    return array(frame_features)

#this converts an entire trajectory into features
def TrajectoryToFeatures(frames, N_nn, method, particle_inc, nn_inc):
    #print filename
    features = []
    #wp: 'frames' is a dictionary containing coordinates, diameters, type, etc
    for frame in frames:        
        features_sp = FrameToFeatures(frame, N_nn, method, particle_inc, nn_inc)
        for feature in features_sp:
            features.append(feature)
    #wp: returns features for all frames
    return features

#################################################################################################################################
#################################################################################################################################
#################################################################################################################################

#this generates NN features for the PCA analysis (or some other machine learning method as well)
def FrameToFeaturesComposition(frame, probe_particle_indicies):
    #extract some relevant frames level details
    coords = deepcopy(frame['coords'])
    p_types = deepcopy(frame['types'])
    D = float(frame['D'])
    N = float(len(coords))
    V = power(frame['L'], D)
    normalizing_distance = power(V/N, 1.0/D)
    diameters = frame['diameters']
    diameters_scaled = frame['diameters']/frame['diameters'][0]
    
    #reduce the coords down to only those we care about
    coords = coords[probe_particle_indicies]

    #build up the features considering only the probes
    frame_features = []
    for particle in coords:
        #nearest neighbor coordinate wrapping
        Rpj_v = particle - coords
        Rpj_v = Rpj_v - rint(Rpj_v/frame['L'])*frame['L']
        Rpj = (sqrt(sum(power(Rpj_v, 2.0), axis=1)))     
        
        #extend the feature vector stacking the particle side by side
        frame_features.extend(Rpj/normalizing_distance)

    return array(frame_features)

#this converts an entire trajectory into features
def TrajectoryToFeaturesComposition(frames, probe_particle_indicies):
    features = []
    for frame in frames:        
        features.append(FrameToFeaturesComposition(frame, probe_particle_indicies))
    return features


#this generates NN features for the PCA analysis (or some other machine learning method as well)
def FrameToFeaturesBatch(frame, N_nn, N_batch, method, particle_inc, nn_inc):
    #extract some relevant frames level details
    coords = deepcopy(frame['coords'])
    D = float(frame['D'])
    N = float(len(coords)) 
    #wp:
    if  'Ly' in frame:
	rectbox=True
        V = frame['L']*frame['Ly']
    else:
        rectbox=False
        V = power(frame['L'], D)

    normalizing_distance = power(V/N, 1.0/D)

    #wp
    if 'orientation' in method:
       orientations=frame['orientation'] 
    
    #wp: determines how many samples per frame and overall feature vector
    aggregated_frame_features = []
    #wp: equiv to batches per frame
    available_number_samples=round(N/particle_inc)
    samples_per_frame=int(round(available_number_samples/float(N_batch))) 
    if samples_per_frame < 2:
       samples_per_frame=1 
       N_batch=int(available_number_samples)
       warnings.warn('The number of batches desired is exactly equal or longer than available samples per frame. Maximum of number of samples used instead.')

    Nint=int(N)
    particle_indeces=range(0,Nint,particle_inc)
    #loop over the batches
    for i in range(samples_per_frame): 
        #wp: rather than pick sequentially, do N_batch from N/part_inc squence eg fvec[0 10, 20], fvec[30,40,50] etc 
        coords_batch=[]
        for j in range(i*N_batch,i*N_batch+N_batch): 
	    coords_batch.append(coords[particle_indeces[j]])

        #wp:convert to array 
        coords_batch=asarray(coords_batch)

        #loop over the particles
        frame_features = []
        frame_features_pos = []
        for particle in coords_batch:
            #nearest neighbor coordinate wrapping
            Rpj_v = particle - coords
            if rectbox:
                Rpj_v[:,0] = Rpj_v[:,0] - rint(Rpj_v[:,0]/frame['L'])*frame['L']
                Rpj_v[:,1] = Rpj_v[:,1] - rint(Rpj_v[:,1]/frame['Ly'])*frame['Ly']
            else:
                Rpj_v = Rpj_v - rint(Rpj_v/frame['L'])*frame['L']

            Rpj = (sqrt(sum(power(Rpj_v, 2.0), axis=1)))     

            #sorting by the distance to enable the discovery of positional order
            sorter = Rpj.argsort()
            Rpj = Rpj[sorter[::1]]

            if 'orientation' in method:
                orientations_sorted=orientations[sorter[::1]] 
                #wp: computes cos(theta) given cos(t/2) and sin(t/2) from orientations
                orientations_cos=orientations_sorted[:,0]**2-orientations_sorted[:,3]**2 
                orientations_sin=2.0*orientations_sorted[:,0]*orientations_sorted[:,3] 
	        #wp: returns |cos(alpha)| where alpha=theta1-theta2 i.e. relative orientns
                cs0=orientations_cos[0]; sn0=orientations_sin[0];
                csi=orientations_cos; sni=orientations_sin;
                orientations_sorted=abs(cs0*csi+sn0*sni)
	        #wp: returns |sin(alpha)| where alpha=theta1-theta2 i.e. relative orientns
                #orientations_sorted=abs(sni*cs0-csi*sn0)

                #wp: append feature for first 'particle' until N_batch 
                #wp: This constitutes the first feaure vector
                frame_features.append((orientations_sorted[1:N_nn+1])[0::nn_inc])
                #wp:to sort based on distances
            	frame_features_pos.append((Rpj[1:N_nn+1])[0::nn_inc])
            else:
                #wp:Just return distances
            	frame_features.append((Rpj[1:N_nn+1])[0::nn_inc])
               

        #wp:Sort separate N_batches by lowest magnitude of feature component
        frame_features = array(frame_features)
        if 'orientation' in method:
            #when using double position sorting 
            #frame_features_pos = array(frame_features_pos)
            #sorter = frame_features_pos[:,0].argsort() 
            #wp: sorts by average orientation. Allegedly, those with better cross-box orientation are a better order indication to sort by than those with less long range order
            frame_features = array(frame_features)
            #sorter = mean(frame_features, axis=1).argsort() 
            sorter = std(frame_features, axis=1).argsort() 
        else:
            sorter = frame_features[:,0].argsort()
        #wp: specifically, sort by last element magnitude (furthest in box)
        frame_features = frame_features[sorter]
        #wp: Flattens all separate batches into a single feature vector
        aggregated_frame_features.append(array(frame_features).flatten()) 
    #wp:---end of 'batches per frame' for loop 

    return array(aggregated_frame_features)

#this converts an entire trajectory into features
def TrajectoryToFeaturesBatch(frames, N_nn, N_batch, method, particle_inc,nn_inc):
    #print filename
    features = []
    for frame in frames: 
        aggregated_frame_features = FrameToFeaturesBatch(frame, N_nn, N_batch, method, particle_inc, nn_inc)
        for frame_features in aggregated_frame_features:
            features.append(frame_features)
    return array(features)


#wp: Generates batch/nested features of a feature ensemble from the perspective of more than one member of the ensemble
def FrameToFeaturesBatchR(frame, N_nn, method, particle_inc, nn_inc, N_batch):
    #extract some relevant frames level details #wp: D=dimension
    coords = deepcopy(frame['coords'])
    D = float(frame['D'])
    N = float(len(coords))
    diameters = frame['diameters']
    diameters_scaled = frame['diameters']/frame['diameters'][0]
    #wp:
    if  'Ly' in frame:
	rectbox=True
        V = frame['L']*frame['Ly']
    else:
        rectbox=False
        V = power(frame['L'], D)

    normalizing_distance = power(V/N, 1.0/D)

    #wp:
    if 'orientation' in method:
       orientations=frame['orientation'] 

    
    frame_features = []
    aggregated_frame_features = []
    combined_data = zip(coords, diameters)
    #wp:for loop below obtains particle and diameter from `combined_data' in increments of 'particle_inc'
    for particle, diameter in combined_data[0::particle_inc]:
        #nearest neighbor coordinate wrapping
        #wp:subtracting an array from a scalar automatically does it for every member of the array i.e. returns the array subtraction
        Rpj_v = particle - coords
        #wp: periodic boundary conditions:
        #wp: 'rint' follows a 'round to even integer' rule--does this matter?
        if rectbox:
            Rpj_v[:,0] = Rpj_v[:,0] - rint(Rpj_v[:,0]/frame['L'])*frame['L']
            Rpj_v[:,1] = Rpj_v[:,1] - rint(Rpj_v[:,1]/frame['Ly'])*frame['Ly']
	else:
            Rpj_v = Rpj_v - rint(Rpj_v/frame['L'])*frame['L']

        #wp: r^2 for each coordinate?
        Rpj = (sqrt(sum(power(Rpj_v, 2.0), axis=1)))     
        
        #generate statistics for various nearest neighbors
        sorter = Rpj.argsort()
	#wp: save sorters in list
        sort_list=[]
        sort_list.append(sorter)
        #wp: sorts them in ascending order using the sorted index
        Rpj = Rpj[sorter[::1]] 

        #wp: Repeat process but for particles within [1:N_nn+1] given sorted distance 
        coords_temp=coords[sorter[::1]] 
        if N_nn % nn_inc == 0: 
            #to keep size cosnsitent even if non-multiple
            coords_temp2=coords_temp[0:N_nn+1][0::nn_inc] 
            coords_temp=vstack((coords_temp2,coords_temp[-1]))
	else: 
            coords_temp=coords_temp[0:N_nn+1][0::nn_inc] 

        #if [1:] removes 0th probe orientation since it is repeated
        coords_temp=coords_temp[0:] 
      
	#wp: determines which indeces to pick to have up to N_batches 
        Ncoords=len(coords_temp)-1
        Nstep=max(Ncoords/N_batch,1)
        BatchIndex=range(Ncoords,1,-Nstep)

	#wp: picks a coord from N_batches
        for batch in BatchIndex:
		Rpj_v = coords_temp[batch] - coords_temp
		#wp: periodic boundary conditions:
		if rectbox:
		    Rpj_v[:,0] = Rpj_v[:,0] - rint(Rpj_v[:,0]/frame['L'])*frame['L']
		    Rpj_v[:,1] = Rpj_v[:,1] - rint(Rpj_v[:,1]/frame['Ly'])*frame['Ly']
		else:
		    Rpj_v = Rpj_v - rint(Rpj_v/frame['L'])*frame['L']

		#wp: r^2 for each coordinate?
		Rpj = (sqrt(sum(power(Rpj_v, 2.0), axis=1)))     
		
		#wp: Find sorted distances
		sorter_i = Rpj.argsort() 
		sort_list.append(sorter_i)
       
        #possible feature options
        feature_batch = []

	#wp: return sorted orientation based on nearest neighbors
        if 'orientation' in method:
            #original reference point for all other particles
            orientations_sorted=orientations[sort_list[0]] 

            if N_nn % nn_inc == 0: 
                orientations_sorted2=orientations_sorted[0:N_nn+1][0::nn_inc] 
                #to keep size cosnsitent when multiple, add last val
                orientations_sorted=vstack((orientations_sorted2,orientations_sorted[-1]))
	    else:                     
                orientations_sorted=orientations_sorted[0:N_nn+1][0::nn_inc] 

            for i in range(len(sort_list)): 
                if i > 0:
                    #when [1:] removes 0th probe orientation since it is repeated 
                    orientations_sorted_i=orientations_sorted[0:]  
                    orientations_sorted_i=orientations_sorted_i[sort_list[i]]
                else:
                    orientations_sorted_i=orientations_sorted 

                #wp: computes cos(theta)
                orientations_cos=orientations_sorted_i[:,0]**2-orientations_sorted_i[:,3]**2 
                orientations_sin=2.0*orientations_sorted_i[:,0]*orientations_sorted_i[:,3] 
	        #wp: returns |cos(alpha)| where alpha=theta1-theta2 i.e. relative orientns
                cs0=orientations_cos[0]; sn0=orientations_sin[0];
                csi=orientations_cos; sni=orientations_sin;
                orientations_feature=abs(cs0*csi+sn0*sni)
                #orientations_sorted=(cs0*csi+sn0*sni)**2 
                
                feature_batch.append(orientations_feature[1:N_nn+1])
           
        if 'distance' in method:
            #wp: scale distances by number density and adds them to new list 'feature_batch'; Note that the first distance=0 is not included
            feature_batch.extend((Rpj[1:N_nn+1]/normalizing_distance)[0::nn_inc])


        #wp:Sort separate N_batches by lowest magnitude of feature component
        frame_features = array(feature_batch)
        #frame_features = feature_batch
        #print("inside batchR method: frame features ", frame_features)

        if 'orientation' in method:
            temp=zeros(len(frame_features))
            #done in for loop due to irregular array sizes in frame_feature list
            for i,tempf in enumerate(frame_features):
                #temp[i]=mean(tempf)
                temp[i]=std(tempf)

            sorter=temp.argsort()
       	    #resorts features 
	    #print frame_features
            frame_features = frame_features[sorter]

        #wp: Flattens all separate batches into a single feature vector
        #wp: can't use flatten as arrays are of different size; use hstack
	frame_features=hstack(frame_features)
        aggregated_frame_features.append(frame_features) 
    #wp:---end of 'particle' for loop 

    return array(aggregated_frame_features) 

#wp:alternative batching using "redundacy check" of the same list but different probe particles
def TrajectoryToFeaturesBatchR(frames, N_nn, method, particle_inc, nn_inc, N_batch):
    #print filename
    features = []
    for frame in frames: 
        aggregated_frame_features = FrameToFeaturesBatchR(frame, N_nn, method, particle_inc, nn_inc, N_batch)
        for frame_features in aggregated_frame_features:
            features.append(frame_features)
    return array(features)


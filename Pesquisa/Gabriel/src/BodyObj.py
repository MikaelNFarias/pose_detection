import numpy as np
from utils import *
import os
import sys
import pickle

import time

class BodyObj:

    def __init__(self,
                 obj_file: str,
                 control_points_file: str,
                 gender: str,
                 facets_file: str = 'facets_template_3DHBSh.npy',
                 debug: bool = False) -> None:

        self.obj_file = obj_file
        self.control_points_file = control_points_file
        self.gender = gender
        self.facets_file = facets_file
        self.debug = debug
    
    def _convert_cp(label = "female"):
        """
        Loads control points from a text file and saves them as a nested list using pickle.

        Args:
            label (str, optional): The label for the control points. Default is "female".

        Returns:
            list: A list of control points previously defined.
        """

        print('[1] starting to load cpoints from txt for %s'%(label))
        start = time.time()
        f = open(os.path.join(os.path.join(CP_FILES_DIR, f'control_points_{label}.txt')), 'r') 
        tmplist = []
        cp_list = []  

        for line in f:
            if '#' in line:
                if len(tmplist) != 0:
                    cp_list.append(tmplist)
                    tmplist = []
            elif len(line.split()) == 1:
                continue
            else:
                tmplist.append(list(map(float, line.strip().split()))[1]) 

        cp_list.append(tmplist)
        f.close()

        # Save the nested list using pickle
        with open(os.path.join(RESHAPER_FILES_DIR, f'cp_{label}.pkl'), 'wb') as f:
            pickle.dump(cp_list, f)

        return cp_list
    
    def convert_template(label = "female"):
        """
        Converts facet information from a .txt file ('facets_template_3DHBS.txt') to a .npy file ('facets_template_3DHBS.npy').

        Args:
            label (str, optional): The label for the template. Defaults to "female".

        Returns:
            np.ndarray: A list of facets previously defined.
        """
        print('[2] starting to load facets from txt for %s'%(label))
        start = time.time()
        facets = np.zeros((F_NUM, 3), dtype=int)
        f = open(os.path.join(RESHAPER_FILES_DIR, 'facets_template_3DHBSh.txt'), 'r')

        i = 0
        for line in f:
            if line[0] == 'f':
                tmp = list(map(int, line[1:].split()))
                facets[i, :] = tmp
                i += 1

        f.close()

        np.save(open(os.path.join(RESHAPER_FILES_DIR, 'facets_template_3DHBSh.npy'), 'wb'), facets)
        print('[2] finished loading facets from txt for %s in %fs' %(label, time.time() - start))

        return facets
    


    def obj2npy(label = "female", obj_file_dir = self.obj_file_dir): 
        """
        Loads data (vertices) from *.obj files in the database and returns a numpy array containing the vertices data.

        Args:
            label (str, optional): The label for the template. Defaults to "female".

        Returns:
            np.ndarray: A numpy array containing the vertices data.
        """


        print('[3] starting to load vertices from .obj files for %s'%(label))
        start = time.time() 
        obj_file = os.path.join(OBJ_FILES_ANSURI)   
        obj_file_dir = os.path.join(OBJ_FILES_ANSURI, label)
        file_list = sorted(os.listdir(obj_file_dir))
    
        # load original data
        vertices = []
        for i, obj in enumerate(file_list):
            sys.stdout.write('\r>>  converting %s body %d'%(label, i + 1))
            sys.stdout.flush()
            f = open(os.path.join(obj_file_dir, obj), 'r')
            for line in f:
                if line[0] == '#':
                    continue
                elif "v " in line:
                    line.replace('\n', ' ')
                    tmp = list(map(float, line[1:].split()))
                    # append vertices from every obj files
                    vertices.append(tmp)
                else:
                    break 
                
            f.close()
        vertices = np.array(vertices, dtype=np.float64).reshape(len(file_list), V_NUM, 3)
    
    # Normalize data
        for i in range(len(file_list)):
                # mean value of each of the 3 columns
                v_mean = np.mean(vertices[i,:,:], axis=0)
                vertices[i,:,:] -= v_mean   
    
        np.save(open(os.path.join(RESHAPER_FILES_DIR, f"vertices_{label}.npy"), "wb"), vertices)
        
        print('\n[3] finished loading vertices from .obj files for %s in %fs' %(label, time.time() - start))
        
        return vertices 

    def infer():

        ...
        
import trimesh
import numpy as np
from scipy.spatial import ConvexHull
import os
def load_mesh(file_path):
    # Load the 3D mesh from a file
    mesh = trimesh.load(file_path)
    return mesh

def get_height(mesh):
    min_z = np.min(mesh.vertices[:, 2])
    max_z = np.max(mesh.vertices[:, 2])
    height = max_z - min_z
    return height * 100  # Convert to cm

def get_girth_at_height(mesh, relative_height):
    min_z = np.min(mesh.vertices[:, 2])
    max_z = np.max(mesh.vertices[:, 2])
    target_height = min_z + relative_height * (max_z - min_z)
    
    threshold = 0.01 * (max_z - min_z)
    vertices_at_height = mesh.vertices[np.abs(mesh.vertices[:, 2] - target_height) < threshold]
    
    if len(vertices_at_height) == 0:
        return 0
    
    vertices_2d = vertices_at_height[:, :2]
    hull = ConvexHull(vertices_2d)
    perimeter = hull.area  # Perimeter approximation
    return perimeter * 100  # Convert to cm

def get_length_between_heights(mesh, start_height, end_height):
    min_z = np.min(mesh.vertices[:, 2])
    max_z = np.max(mesh.vertices[:, 2])
    start_z = min_z + start_height * (max_z - min_z)
    end_z = min_z + end_height * (max_z - min_z)
    length = abs(end_z - start_z)
    return length * 100  # Convert to cm

def get_forearm_length(mesh):
    return get_length_between_heights(mesh, 0.3, 0.4)

def get_crotch_height(mesh):
    return get_length_between_heights(mesh, 0.0, 0.2)

def get_thigh_length(mesh):
    return get_length_between_heights(mesh, 0.2, 0.4)

def get_chest_depth(mesh):
    min_z = np.min(mesh.vertices[:, 2])
    max_z = np.max(mesh.vertices[:, 2])
    chest_height = min_z + 0.6 * (max_z - min_z)
    
    threshold = 0.01 * (max_z - min_z)
    chest_vertices = mesh.vertices[np.abs(mesh.vertices[:, 2] - chest_height) < threshold]
    
    if len(chest_vertices) == 0:
        return 0
    
    min_y = np.min(chest_vertices[:, 1])
    max_y = np.max(chest_vertices[:, 1])
    depth = max_y - min_y
    return depth * 100  # Convert to cm

def main():
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
    DATA_DIR = os.path.join(CURRENT_DIR, 'data')
    FILE_PATH = os.path.join(DATA_DIR, 'ANSURI_fem_0005.obj')
    #file_path = 'path_to_your_3d_mesh_file.obj'  # Replace with your file path
    mesh = load_mesh(FILE_PATH)
    
    stature_cm = get_height(mesh)
    neck_base_girth = get_girth_at_height(mesh, 0.85)
    chest_girth = get_girth_at_height(mesh, 0.6)
    waist_girth = get_girth_at_height(mesh, 0.5)
    hips_buttock_girth = get_girth_at_height(mesh, 0.4)
    shoulder_girth = get_girth_at_height(mesh, 0.75)
    thigh_girth = get_girth_at_height(mesh, 0.3)
    thigh_low_girth = get_girth_at_height(mesh, 0.25)
    calf_girth = get_girth_at_height(mesh, 0.15)
    ankle_girth = get_girth_at_height(mesh, 0.05)
    forearm_girth = get_girth_at_height(mesh, 0.35)
    wrist_girth = get_girth_at_height(mesh, 0.05)
    shoulder_length = get_length_between_heights(mesh, 0.7, 0.75)
    sleeveoutseam_length = get_length_between_heights(mesh, 0.75, 0.85)
    forearm_length = get_forearm_length(mesh)
    crotchheight_length = get_crotch_height(mesh)
    waistback_length = get_length_between_heights(mesh, 0.4, 0.5)
    thigh_length = get_thigh_length(mesh)
    chest_depth_length = get_chest_depth(mesh)
    head_girth = get_girth_at_height(mesh, 0.95)  # Assuming the head girth is at 95% height

    print(f'Stature (Height): {stature_cm:.2f} cm')
    print(f'Neck Base Girth: {neck_base_girth:.2f} cm')
    print(f'Chest Girth: {chest_girth:.2f} cm')
    print(f'Waist Girth: {waist_girth:.2f} cm')
    print(f'Hips/Buttock Girth: {hips_buttock_girth:.2f} cm')
    print(f'Shoulder Girth: {shoulder_girth:.2f} cm')
    print(f'Thigh Girth: {thigh_girth:.2f} cm')
    print(f'Thigh Low Girth: {thigh_low_girth:.2f} cm')
    print(f'Calf Girth: {calf_girth:.2f} cm')
    print(f'Ankle Girth: {ankle_girth:.2f} cm')
    print(f'Forearm Girth: {forearm_girth:.2f} cm')
    print(f'Wrist Girth: {wrist_girth:.2f} cm')
    print(f'Shoulder Length: {shoulder_length:.2f} cm')
    print(f'Sleeve Outseam Length: {sleeveoutseam_length:.2f} cm')
    print(f'Forearm Length: {forearm_length:.2f} cm')
    print(f'Crotch Height Length: {crotchheight_length:.2f} cm')
    print(f'Waist Back Length: {waistback_length:.2f} cm')
    print(f'Thigh Length: {thigh_length:.2f} cm')
    print(f'Chest Depth Length: {chest_depth_length:.2f} cm')
    print(f'Head Girth: {head_girth:.2f} cm')

if __name__ == '__main__':
    main()

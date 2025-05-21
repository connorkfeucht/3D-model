import h5py # reading .hdf5 files
import sys # for command line arguments
import os # for changing directories
import glob # filename pattern matching
import numpy as np # array manipulation
import pyvista as pv # high level wrapper around VTK for building and rendering 3d meshes

def plot_mesh(mesh, output1, output2, orig_dir, target_dir): # takes single mesh, writes to 2 different pngs from different angles.
    os.chdir(orig_dir)
    output_dir = os.path.join(orig_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    # Render & save screenshot
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, color="white")
    plotter.set_background("black")
    plotter.camera_position = "iso" # can be "xy" "zy" or a point
    plotter.line_smoothing = True
    # might want to use image_scale at some point to make images smaller or larger
    plotter.show(screenshot=output1)

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, color="white")
    plotter.set_background("black")
    plotter.camera_position = "xy"
    plotter.line_smoothing = True
    plotter.show(screenshot=output2)

    os.chdir("..")
    os.chdir(target_dir)


def parse_mesh(filename, output1, output2, orig_dir, target_dir): # reads one .hdf5 file, builds the mesh and calls plot_mesh
    # Open and locate the mesh groups 
    with h5py.File(filename, "r") as f:
        mesh_root = f["parts"]["part_001"]["mesh"] # contains subgroups 000, 001, ... which each contain one small mesh

        # Load each sub-mesh into a PolyData and collect them
        polys = []
        for mesh_id in sorted(mesh_root.keys(), key=int): # get names like 000, 001, ... and sort them numerically
            grp = mesh_root[mesh_id]
            pts = grp["points"][...]       # shape (N,3), absolute coords of each vertex
            tris = grp["triangle"][...]    # shape (M,3), how those vertices connect to form triangles, each value is a vertex's index in the pts array. 

            # VTK wants faces in "flat" format: [3, i0, i1, i2,  3, j0, j1, j2, ...]. prefixing each triangle with a 3.
            faces = np.hstack([
                np.concatenate([[3], tri.astype(np.int64)])
                for tri in tris
            ]) # loop over each triangle, prepend the count 3, cast to int64, then horizontally stack them all into one flat array

            poly = pv.PolyData(pts, faces) # builds PolyData objects, which are the small meshes in parts 000, 001, ...
            polys.append(poly)

    # Merge all sub-meshes into one
    mesh = polys[0]
    for poly in polys[1:]:
        mesh = mesh.merge(poly)
    
    plot_mesh(mesh, output1, output2, orig_dir, target_dir)

# Using PyVista to render images of .hdf5 3D models
def main(argc, argv):
    if argc != 2:
        print("Please specify a path to the directory which the input files are stored as an argument. ie: /path/to/my/hdf5_folder")
        return
    
    orig_dir = os.getcwd() # saves original directory
    target_dir = argv[1]
    if not os.path.isdir(target_dir):
        print(f"Error: “{target_dir}” is not a directory.")
        sys.exit(1)

    os.chdir(target_dir)
    for hdf5_file in glob.glob("*.hdf5"): # runs parse_mesh on all of the .hdf5 files in the target directory
        name = os.path.splitext(hdf5_file)[0]
        output1 = f"{name}1.png"
        output2 = f"{name}2.png"
        parse_mesh(hdf5_file, output1, output2, orig_dir, target_dir)


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)


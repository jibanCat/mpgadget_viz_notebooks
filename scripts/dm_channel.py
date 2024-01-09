from bigfile import BigFile

import numpy as np
from matplotlib import pyplot as plt


# A customized tool for visaulize SPH simulations from Gadget 
# https://github.com/rainwoodman/gaepsi
from gaepsi2 import painter
from gaepsi2 import color
from gaepsi2 import camera
import psutil
import os
import math

def memory_usage_psutil():
    # returns memory usage in GB
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / float(2 ** 30)  # convert bytes to gigabytes
    return mem

print(f"Initial Memory Usage: {memory_usage_psutil()} GB")

import warnings
warnings.filterwarnings("ignore")

plt.rc("xtick", labelsize=15)  # fontsize of the tick labels
plt.rc("ytick", labelsize=15)
plt.rcParams["axes.linewidth"] = 1.5  # set the value globally
plt.rcParams["font.family"] = "serif"

# Import the required matplotlib components for color mapping.
import matplotlib as mpl


# This class helps with creating and using color maps from Matplotlib.
class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        # Store the name of the color map.
        self.cmap_name = cmap_name
        # Get the color map from matplotlib's available color maps.
        self.cmap = plt.get_cmap(cmap_name)
        # Normalize the color map between the start and stop values.
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        # Create a ScalarMappable object which will map values to colors.
        self.scalarMap = mpl.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    # Method to get the RGB color for a given value.
    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)

def cut_tube(part, tube, partition=10):
    """
    This function reads a subset of a particle file to conserve memory usage.

    Parameters:
    part (Bigfile): The input particle file. This file can be opened using part.open("Position")[:].
                    It is read from Bigfile.File("path/to/PART").
    tube (dict): Parameters for the tube, including resolution, wt, Lx, Ly, Lz, and imsize.
    partition (int): The number of partitions to divide the file into for memory-efficient reading.
                     This is helpful when dealing with large particle position files.

    Returns:
    tuple: A tuple containing the full mask and the processed particle positions (ppos).
    """

    # Initialize an empty list to store particle positions.
    ppos_list = []

    # Get the total number of positions to determine batch sizes for partitioning.
    size = part.open("Position").size

    # Create a mask array initialized with zeros. This mask will be used later.
    full_mask = np.zeros((size,), dtype=np.bool_)

    # Calculate the batch size for each partition.
    batch = size // partition

    # Process each partition.
    for i in range(partition):
        # Define start and end indices for the current batch.
        istart = i * batch
        iend = (i + 1) * batch

        # Load a chunk of positions, converting data type to float64 for memory efficiency.
        ppos = np.float64(part.open("Position")[istart:iend])

        # Adjust positions relative to the tube's center.
        ppos -= tube["center"]
        boxsize = tube["Lbox"]

        # Apply periodic boundary conditions to ensure particles wrap around the box correctly.
        ppos[ppos < -boxsize / 2] += boxsize
        ppos[ppos > boxsize / 2] -= boxsize

        # Create a mask to filter particles based on their positions within the tube dimensions.
        mask = np.abs(ppos[:, 0]) < tube["Lx"] * 0.5
        mask &= np.abs(ppos[:, 1]) < tube["Ly"] * 0.5
        mask &= np.abs(ppos[:, 2]) < tube["Lz"] * 0.5

        # Update the full mask with the current batch's mask.
        full_mask[istart:iend] = mask

        # Append the filtered positions to the list and free memory.
        ppos_list.append(ppos[mask])
        del ppos, mask

    # Concatenate all filtered positions into a single array.
    ppos = np.concatenate(ppos_list)

    return full_mask, ppos

def cut_tube_mpi(part, tube, rank, num_rank, partition=10):
    """
    This function reads a subset of a particle file to conserve memory usage.

    Parameters:
    part (Bigfile): The input particle file. This file can be opened using part.open("Position")[:].
                    It is read from Bigfile.File("path/to/PART").
    tube (dict): Parameters for the tube, including resolution, wt, Lx, Ly, Lz, and imsize.
    partition (int): The number of partitions to divide the file into for memory-efficient reading.
                     This is helpful when dealing with large particle position files.

    Returns:
    tuple: A tuple containing the full mask and the processed particle positions (ppos).
    """

    # Initialize an empty list to store particle positions.
    ppos_list = []

    # Get the total number of positions to determine batch sizes for partitioning.
    size = part.open("Position").size

    portion_size = math.ceil(size / num_rank)
    offset = rank * portion_size
    portion_size_last = portion_size

    # Create a mask array initialized with zeros. This mask will be used later.
    full_mask = np.zeros((portion_size,), dtype=np.bool_)

    # Adjust portion size for the last rank
    if rank == num_rank - 1 and size % num_rank != 0:
        portion_size_last = size - offset
        full_mask = np.zeros((portion_size_last,), dtype=np.bool_)

    # Calculate the batch size for each partition.
    batch = math.ceil(portion_size / partition)

    # Process each partition.
    for i in range(partition):
        # Define start and end indices for the current batch.
        istart = i * batch + offset
        iend = min((i + 1) * batch + offset, offset + portion_size_last)
        print(f"rank {rank}, istart: {istart}, iend: {iend}")
        # Load a chunk of positions, converting data type to float64 for memory efficiency.
        ppos = np.float64(part.open("Position")[istart:iend])

        # Adjust positions relative to the tube's center.
        ppos -= tube["center"]
        boxsize = tube["Lbox"]

        # Apply periodic boundary conditions to ensure particles wrap around the box correctly.
        ppos[ppos < -boxsize / 2] += boxsize
        ppos[ppos > boxsize / 2] -= boxsize

        # Create a mask to filter particles based on their positions within the tube dimensions.
        mask = np.abs(ppos[:, 0]) < tube["Lx"] * 0.5
        mask &= np.abs(ppos[:, 1]) < tube["Ly"] * 0.5
        mask &= np.abs(ppos[:, 2]) < tube["Lz"] * 0.5

        # Update the full mask with the current batch's mask.
        istart_mask = i * batch
        iend_mask = min((i + 1) * batch, portion_size_last)
        full_mask[istart_mask:iend_mask] = mask

        # Append the filtered positions to the list and free memory.
        ppos_list.append(ppos[mask])
        del ppos, mask

    # Concatenate all filtered positions into a single array.
    ppos = np.concatenate(ppos_list)

    return full_mask, ppos

def getcorner(tube, theta, ct_scale, r, z):
    """
    Calculates the list of linked corner positions in image size coordinates based on the given parameters.

    Parameters:
    tube (dict): Contains tube parameters including Lx, Ly, Lz, and imsize.
    theta (float): Angle in radians, used for calculating corner positions.
    ct_scale (float): Scale factor for the camera transformation.
    r (float): Radius used for the position calculation.
    z (float): Z-coordinate for the position calculation.

    Returns:
    list: A list of linked corner positions in image size coordinates.
    """
    # Normalize the angle to ensure it is within 0 to 2π range.
    theta = np.mod(theta, 2 * np.pi)

    # Extract and scale the dimensions of the tube.
    Lx, Ly, Lz = tube["Lx"] * 1.0, tube["Ly"] * 1.0, tube["Lz"] * 1.0
    # Determine the largest dimension for further calculations.
    lgt = 0.5 * np.max([Lx, Ly, Lz])
    # Image size from the tube parameters.
    imsize = tube["imsize"]

    # Calculate x and y coordinates based on the radius and angle.
    x, y = r * np.cos(theta), r * np.sin(theta)
    # Calculate the camera transformation scale.
    ct = lgt * ct_scale

    # Set up the orthogonal projection matrix for the camera.
    mpers = camera.ortho(-ct, ct, (-ct, ct, -ct, ct))
    # Set up the model-view matrix to look at the target from a specific position.
    mmv = camera.lookat((x * lgt, y * lgt, z * lgt), (0, 0, 0), (0, 0, 1))

    # Define the coordinates of the corners of a box.
    corner = np.array(
        [
            # Corner coordinates are defined relative to the center.
            [-Lx * 0.5, -Ly * 0.5, -Lz * 0.5],  # 0
            [-Lx * 0.5, -Ly * 0.5, +Lz * 0.5],  # 1
            [-Lx * 0.5, +Ly * 0.5, -Lz * 0.5],  # 2
            [-Lx * 0.5, +Ly * 0.5, +Lz * 0.5],  # 3
            [+Lx * 0.5, -Ly * 0.5, -Lz * 0.5],  # 4
            [+Lx * 0.5, -Ly * 0.5, +Lz * 0.5],  # 5
            [+Lx * 0.5, +Ly * 0.5, -Lz * 0.5],  # 6
            [+Lx * 0.5, +Ly * 0.5, +Lz * 0.5],
        ]
    )  # 7

    # Apply the camera matrices to transform the corner coordinates into 2D.
    corner2d = camera.apply(camera.matrix(mpers, mmv), corner)
    # Transform the 2D coordinates for rendering on the device (e.g., screen).
    cornerdev = camera.todevice(corner2d, extent=(imsize, imsize))

    # Determine the dashed corner based on the theta value.
    if 0 <= theta < np.pi / 2:
        dash_corner = cornerdev[0]
    elif np.pi / 2 <= theta < np.pi:
        dash_corner = cornerdev[4]
    elif np.pi <= theta < 1.5 * np.pi:
        dash_corner = cornerdev[6]
    elif 1.5 * np.pi <= theta < 2 * np.pi:
        dash_corner = cornerdev[2]

    # Initialize an empty list to store the 2D corner coordinates.
    cn2d = []
    for i in range(0, len(cornerdev)):
        p = cornerdev[i]
        # Calculate the difference relative to other corners.
        dr = corner - corner[i]
        # Find the nearest corner for creating an edge.
        msk = np.count_nonzero(dr, axis=-1) == 1
        for o in cornerdev[msk]:
            # Create and append the line coordinates to the list.
            x, y = np.array([p[0], o[0]]), np.array([p[1], o[1]])
            # Calculate distances to determine if the line should be dashed or solid.
            r2 = np.linalg.norm(o - dash_corner, axis=0)
            r3 = np.linalg.norm(p - dash_corner, axis=0)
            if r2 < 1 or r3 < 1:
                cn2d.append([x, y, 0])  # Dashed line
            else:
                cn2d.append([x, y, 1])  # Solid line
    return cn2d

def get_dmonly_channel(
    filename,
    tube,
    theta,
    ct_scale,
    r,
    z,
    use_input_pos_mask: bool = False,
    pos: np.ndarray = None,
):
    """
    Calculate density image channel based on given file.

    Parameters:
    filename (str): Particle file, assumed Gas particle.
    tube (dict): Tube parameters including resolution, wt, Lx, Ly, Lz, and imsize.
    theta (float): Angle in radians.
    ct_scale (float): Control the distance from the camera to the box.
    r (float): Radial coordinate.
    z (float): Z-coordinate.
    use_input_pos_mask (bool): Flag to use input position and mask (optimization).
    pos (np.ndarray): Optional array of positions.

    Returns:
    list: List containing temperature and density image channels.
    """

    # Load particle data from file.
    part = BigFile(filename)

    # If using input position mask, skip cutting tube to save time.
    if use_input_pos_mask:
        print("Use input pos and mask instead (saving iteration time)")
    else:
        print("Start cutting ...")
        # Cut the tube from the particle field to get the mask and positions.
        mask, pos = cut_tube(part, tube)

    # ------- Image Channel Processing -------

    # Set the smoothing length and weight for each particle.
    sml = np.ones(len(pos)) * tube["resolution"]
    weight = np.ones(len(pos)) * tube["wt"]

    # Determine the maximum dimension from the tube for camera positioning.
    Lx, Ly, Lz = tube["Lx"] * 1.0, tube["Ly"] * 1.0, tube["Lz"] * 1.0
    lgt = 0.5 * np.max([Lx, Ly, Lz])
    # Image size from the tube parameters.
    imsize = tube["imsize"]

    # Calculate camera position based on input parameters.
    x, y = r * np.cos(theta), r * np.sin(theta)
    ct = lgt * ct_scale

    # Set up the camera's projection matrix.
    print("Define the projection matrix for the camera ...")
    mpers = camera.ortho(-ct, ct, (-ct, ct, -ct, ct))  # Orthographic projection matrix.

    # Set up the camera's model-view matrix.
    print("Define the modelview matrix for the camera ...")
    mmv = camera.lookat(
        (x * lgt, y * lgt, z * lgt), (0, 0, 0), (0, 0, 1)
    )  # Camera's viewpoint.

    # Transform data coordinates to clip coordinates using camera matrices.
    print(
        "Apply the camera matrix to the data coordinates to obtain position in clip coordinates ..."
    )
    gas2d = camera.apply(camera.matrix(mpers, mmv), pos)
    print(f"Channel: Memory Usage: {memory_usage_psutil()} GB")

    # Convert clip coordinates to device coordinates for rendering.
    print("Convert the clipping coordinates to device coordinates ...")
    gasdev = camera.todevice(gas2d, extent=(imsize, imsize))

    print(f"Channel: Memory Usage: {memory_usage_psutil()} GB") 
    # Use a painting algorithm to create the image channels from the device coordinates.
    channels = painter.paint(gasdev, sml, [weight], (imsize, imsize), np=None) # none: all available processors
    print(f"Channel paint: Memory Usage: {memory_usage_psutil()} GB") 

    # Normalize and adjust image channels if necessary.
    # channels[1] /= channels[0]

    # Transpose the channels to correct the orientation.
    channels[0] = np.transpose(channels[0])
    # channels[1] = np.transpose(channels[1])

    return channels


def get_channel_portion(file, theta, ct_scale, r, z, Lbox, Lx, Ly, Lz, imsize, res, wt, rank, num_rank):
    tube = {
    "Lbox": Lbox,  # The side length of the entire simulation box in kpc. Equivalent to 60 cMpc/h.
    "center": np.array(
        [0, 0, 0]
    ),  # The center point of the subset box in the 3D image, here set to the origin.
    "Lx": Lx,  # The x-dimension size of the subset of the simulation to visualize, in kpc. Also 60 cMpc/h.
    "Ly": Ly,  # The y-dimension size, similar to 'Lx'.
    "Lz": Lz,  # The z-dimension size, similar to 'Lx'.
    "imsize": imsize,  # The size of the interpolation grid. Determines the resolution of the output image.
    "resolution": res,  # The resolution of the visualization. Related to the SPH (Smoothed Particle Hydrodynamics) smoothing length.
    "wt": wt,  # Normalization weighting factor for SPH interpolation. Used to adjust the influence of each particle.
}
    
    part = BigFile(file)
    mask_l, ppos_l = cut_tube_mpi(part, tube, rank, num_rank)
    channel = get_dmonly_channel(
    file,
    tube,
    theta,
    ct_scale,
    r,
    z,
    use_input_pos_mask = True,
    pos = ppos_l,
)   
    return channel

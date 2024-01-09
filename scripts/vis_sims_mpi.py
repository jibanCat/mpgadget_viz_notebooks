from mpi4py import MPI
import numpy as np
import argparse

from matplotlib import pyplot as plt

import argparse
import os

from gaepsi2 import color

from dm_channel import MplColorHelper, getcorner, get_channel_portion


def plot_img(summed_channel, figsize, tube, theta, ct_scale, r, z, out_name):
    imsize = tube['imsize']
        # Processing image data for visualization.
    xx = summed_channel[0][summed_channel[0] > 0]
        # Auto-calculate f0 range if not provided.
    f0_low = np.log10(np.percentile(xx, 25))
    f0_high = np.log10(np.percentile(xx, 90))

    channels1 = np.ones([imsize, imsize]) * 1e5
    img_t = dmmap(
        color.NL(channels1, range=(4.4, 5.5)),
        color.NL(summed_channel, range=(f0_low, f0_high)),
        )

    f, ax = plt.subplots(1, 1, figsize=figsize)

    ax.imshow(img_t, origin="lower", extent=(0, imsize, 0, imsize))

    box = getcorner(tube, theta, ct_scale, r, z)
    for d in box:
        x, y, m = d
                        # Choose linestyle based on the value of 'm'.
        if m == 0:
            ax.plot(x, y, c="grey", alpha=0.5, lw=1, linestyle="--")
        if m == 1:
            ax.plot(x, y, c="C0", lw=1.5, alpha=0.7)

    ax.set_xlim(0, imsize)
    ax.set_ylim(0, imsize)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_name, bbox_inches="tight", dpi=200)

if __name__ == "__main__":

        # Generate an array of 100 linearly spaced values between 0 and 1.
    y = np.linspace(0, 1, 100)
    # Create an instance of MplColorHelper with the "BuPu" colormap, ranging from 0 to 1.
    COL = MplColorHelper("BuPu", 0, 1)

    # Apply the color map to the array of values 'y' to get the corresponding RGB values.
    x = COL.get_rgb(y)
    # Create a colormap object using the RGB values.
    dmmap = color.Colormap(x[:, 0:3])

    #Parse command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--file", type=str)
    parser.add_argument("-l","--Lbox",default=1000000, type=float)
    parser.add_argument("-x","--Lx",default=None, type=float)
    parser.add_argument("-y","--Ly",default=None, type=float)
    parser.add_argument("-z","--Lz",default=None, type=float)
    parser.add_argument("-s","--figsize",default=8., type=float)
    parser.add_argument("-o","--out_name",default='vis.png',type=str)
    parser.add_argument("-i","--imsize",default=1000, type=int)
    parser.add_argument("-r","--resolution",default=5., type=float)
    parser.add_argument("-w","--wt",default=1/750, type=float)
    parser.add_argument("-c","--channel_file",default=None, type=str)
    parser.add_argument("-v","--save_channels",default=True, type=bool)

#Parse command arguments and check that all provided options are available
    cmd_args = parser.parse_args()

# e.g., 
# ibrun python vis_sims_mpi.py -f "/scratch1/01317/yyang440/cosmo_11p_sims/cosmo_11p_Box250_Part750_1040/output/PART_068/1/" -l 250000 -s 2. -o 'vis_Box250_Part750_1040_redshift0.png'

# input
    file = cmd_args.file
# "/scratch1/01317/yyang440/cosmo_11p_sims/cosmo_11p_Box1000_Part3000_0536/output/PART_067/1/"
    Lbox = cmd_args.Lbox # 250000
    figsize = (cmd_args.figsize,cmd_args.figsize)
    out_name = cmd_args.out_name
# 'vis_Box1000_Part3000_0536_redshift0.png'
    imsize = cmd_args.imsize
    res = cmd_args.resolution
    save_channels = cmd_args.save_channels

    Lx = cmd_args.Lx or Lbox
    Ly = cmd_args.Ly or Lbox
    Lz = cmd_args.Lz or Lbox

    wt = cmd_args.wt
    channel_file = cmd_args.channel_file
# Define a dictionary 'tube' containing parameters for visualizing a subset of the simulation.
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

    ct_scale=1.4
    theta=5.0
    z=0.5
    r=2

    if channel_file:
        channel = np.load(channel_file)
        tube['imsize'] = channel.shape[1]
        plot_img(channel, figsize, tube, theta, ct_scale, r, z, out_name)
        exit()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_rank = comm.Get_size()

    # Each rank generates a (2, 2, 1) shaped array
    local_channel = get_channel_portion(file, theta, ct_scale, r, z, Lbox, Lx, Ly, Lz, imsize, res, wt, rank, num_rank)
    out, ext = os.path.splitext(out_name)
    if save_channels:
        np.save(f"{out}_rank{rank}.npy", local_channel)
    # Prepare an array to gather all arrays at the root

    # Sum the arrays across all ranks
    summed_channel = np.empty((1, imsize, imsize), dtype='float32') if rank == 0 else None
    comm.Reduce(local_channel, summed_channel, op=MPI.SUM, root=0)

    if rank == 0:
        print("\nSummed channel:\n", summed_channel)
        np.save(f"channel_summed.npy", summed_channel)

        plot_img(summed_channel, figsize, tube, theta, ct_scale, r, z, out_name)



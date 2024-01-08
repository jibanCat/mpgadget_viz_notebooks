# MP-Gadget Visualization Notebooks

This repository hosts a straightforward visualization tutorial for MP-Gadget simulations using Gaepsi2, a versatile visualization tool. You can explore Gaepsi2 in detail [here](https://github.com/rainwoodman/gaepsi2).

![Visualization Example 1](https://github.com/jibanCat/mpgadget_viz_notebooks/assets/23435784/cae36339-65c8-4cec-aa19-e678e7fdb112)
![Visualization Example 2](https://github.com/jibanCat/mpgadget_viz_notebooks/assets/23435784/183ad60a-e6f1-4af4-994e-4db597d821bd)

The tutorial is an adaptation of Yueying Ni's original notebook, modified to suit the MP-Gadget framework.

## MP-Gadget Simulations

MP-Gadget is a versatile and scalable cosmological simulation code. You can find more information and download the simulations from their GitHub page: [MP-Gadget Repository](http://github.com/MP-Gadget/MP-Gadget/).

For accessing the simulation output, navigate to the `output` directory. Particle position data can be found in `output/PART_XXX/`, where `XXX` is the snapshot number. These snapshots represent various stages of cosmic evolution, with higher numbers indicating more recent phases of the Universe. To correlate snapshot numbers with redshift values, consult the `output/Snapshots.txt` file, which includes a detailed conversion table.

## Example Data for Tutorial

To facilitate learning, I have provided a sample simulation dataset. This dataset features a simulation with \(128^3\) particles (both dark matter and gas particles) within a 60 cMpc/h box. The dataset includes only the `PART` folder for the final snapshot and is available on Google Drive: [Sample Simulation Data](https://drive.google.com/drive/folders/1ygmwjg_TT9qAgArnIUP1ZquuinlcyOCf?usp=share_link).

## Parallel Visualization on Distributed Memory Systems

Gaepsi2 offers multiprocessing support via *sharedmem*, but it's restricted to shared-memory architectures (single compute cluster nodes), posing challenges for visualizing large-scale simulations. To address this, we've included scripts for multi-node visualization using *mpi4py* in the 'scripts' directory. An example job submission script, `submit.sh`, is also provided to facilitate easy deployment on distributed systems.

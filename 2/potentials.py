import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

from octree import build_octree, Octree_Node, get_node_at_level, fill_massmap_from_octree

Np = np.int64(16) ** 3
#Np = np.int64(256) ** 3  # number of particles
mp = np.float32(3.64453e10)  # particle mass in Msun; all 32-bit to save memory
G = np.float32(4.3009e-9)  # gravitational constant in Mpc*(km/s)^2/Msun
h = np.float32(
    0.3755
)  # Hubble parameter (this is a Einstein-de Sitter universe with Omega_m=1)
L = np.float32(250.0)  # side length of periodic cubic simulation volume
scale_factor = np.float32(0.1)  # scale factor a
redshift = 1.0 / scale_factor - 1
rho_mean = (
    Np * mp / L**3
)  # mean density in Msun/Mpc^3 (comoving, matches 3*H_0^2/(8*pi*G))


def main() -> None:
    output_dir = "Plots"
    os.makedirs(output_dir, exist_ok=True)

    # Question 2: Calculating potentials

    # in the run.sh file there is no command that downloads this dataset,
    # so to test my code i will just randomly populate particles.
    # with h5py.File("/disks/cosmodm/DMO_a0.1_256.hdf5", "r") as handle:
    #     pos = handle["Position"][...]  # particle positions, shape (Np,3), comoving
    #     # vel=handle["Velocity"][...] #particle velocities, shape (Np,3), comoving <-- not used, but if you're interested
    
    pos = np.random.uniform(
        low=0.0,
        high=L,
        size=(Np, 3)
    )
    
    # Question 2a: using Barnes-Hut [note: not actually calculating a potential, unless you do the bonus question]

    root = build_octree(
        particles=np.arange(Np),
        particle_positions=pos,
        center_position=np.array([0.5*L, 0.5*L, 0.5*L], dtype=np.float32),
        index=np.array([0, 0, 0]),
        box_size=L,
        depth=0,
        max_depth=7,
    )
    
    # Plotting the mass distribution for a slice

    for level in [1, 3, 5, 7]:  # feel free to change any of this code
        pixels = 2**level
        massmap = np.zeros((4, pixels, pixels), dtype=np.float32)

        fill_massmap_from_octree(
            root=root,
            level=level,
            massmap=massmap,
        )

        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        pcm = ax[0, 0].pcolormesh(
            np.arange(pixels), np.arange(pixels), massmap[0, :, :]
        )
        ax[0, 0].set(ylabel="z index", title="x index = 0")
        fig.colorbar(pcm, ax=ax[0, 0], label="Total mass inside node")

        pcm = ax[0, 1].pcolormesh(
            np.arange(pixels), np.arange(pixels), massmap[1, :, :]
        )
        ax[0, 1].set(title="x index = 1")
        fig.colorbar(pcm, ax=ax[0, 1], label="Total mass inside node")

        pcm = ax[1, 0].pcolormesh(
            np.arange(pixels), np.arange(pixels), massmap[2, :, :]
        )
        ax[1, 0].set(ylabel="z index", xlabel="y index", title="x index = 2")
        fig.colorbar(pcm, ax=ax[1, 0], label="Total mass inside node")

        pcm = ax[1, 1].pcolormesh(
            np.arange(pixels), np.arange(pixels), massmap[3, :, :]
        )
        ax[1, 1].set(xlabel="y index", title="x index = 3")
        fig.colorbar(pcm, ax=ax[1, 1], label="Total mass inside node")

        ax[0, 0].set_aspect("equal", "box")
        ax[0, 1].set_aspect("equal", "box")
        ax[1, 0].set_aspect("equal", "box")
        ax[1, 1].set_aspect("equal", "box")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"fig2a_level{level}.png"), dpi=300)
        plt.close()

    # Question 2b: using the FFT

    Ngrid = np.int64(128)
    densgrid = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.float32)
    potential = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.float32)
    # TO DO: assign particle masses to densgrid, convert to density, and calculate potentials from it

    # Plotting four slices of a grid

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    pcm = ax[0, 0].pcolormesh(np.arange(Ngrid), np.arange(Ngrid), potential[0, :, :])
    # ax[0,0].set(ylabel='...', title='...')
    fig.colorbar(pcm, ax=ax[0, 0], label="Potential")
    pcm = ax[0, 1].pcolormesh(np.arange(Ngrid), np.arange(Ngrid), potential[16, :, :])
    # ax[0,1].set(title='...')
    fig.colorbar(pcm, ax=ax[0, 1], label="Potential")
    pcm = ax[1, 0].pcolormesh(np.arange(Ngrid), np.arange(Ngrid), potential[32, :, :])
    # ax[1,0].set(ylabel='...', xlabel='...', title='...')
    fig.colorbar(pcm, ax=ax[1, 0], label="Potential")
    pcm = ax[1, 1].pcolormesh(np.arange(Ngrid), np.arange(Ngrid), potential[64, :, :])
    # ax[1,1].set(xlabel='...', title='...')
    fig.colorbar(pcm, ax=ax[1, 1], label="Potential")
    ax[0, 0].set_aspect("equal", "box")
    ax[0, 1].set_aspect("equal", "box")
    ax[1, 0].set_aspect("equal", "box")
    ax[1, 1].set_aspect("equal", "box")
    plt.savefig(os.path.join(output_dir, "fig2b.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()

import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

Np = np.int64(256) ** 3  # number of particles
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


class Octree_Node:
    """
    A node in a 3D octree.

    Parameters
    ----------
    index : ndarray
        index of the node, shape (3,).
        From the index of the node the center_position is calculated.
        the center position can be retrieved using the get_center_position method.
    depth : int
        Depth of the node. The root has depth 0
    index : tuple
        Integer index of the node at its depth
    CoM : ndarray
        Centre-of-mass position, shape (3,)
    children : ndarray or None
        Child nodes, shape (2, 2, 2), or None for leaf nodes
    particles : ndarray or None
        Particle indices inside the node
        This is only needed for leaf nodes 
    """

    def __init__(
        self,
        index:np.ndarray,
        depth:int,
        CoM:np.ndarray,
        children=None,
        particles=None,
    ): # note: the fewer of these you keep, the better! these are just examples
        self.particles = particles
        self.depth = depth
        self.index = index
        self.CoM = CoM
        self.children = children

        if particles is None:
            self.mass = np.float32(0.0)
        else:
            self.mass = np.float32(len(particles)) * mp
            
    def get_center_position(self, box_dimensions:np.ndarray):
        square_dimensions = (box_dimensions / (self.depth+1))
        grid_point =  self.index*square_dimensions
        offset = 0.5*square_dimensions
        return grid_point + offset


def build_octree(
    particles,
    particle_positions,
    center_position,
    index,
    box_size,
    depth=0,
    max_depth=7,
):
    """
    Build an octree recursively.

    Parameters
    ----------
    particles : ndarray
        Contains the indices of the particles inside the current node, shape (N_node,)
    particle_positions : ndarray
        Particle positions inside the current node, shape (N_node, 3)
    center_position : ndarray
        Geometrical center of the current node, shape (3,)
    index : tuple
        Integer index of this node at the current depth, (ix, iy, iz)
    box_size : float
        Side length of the current node
    depth : int
        Current octree depth
    max_depth : int
        Maximum tree depth

    Returns
    -------
    node : Octree_Node
        Current octree node
    """

    # If node contains no particles
    # mass should be zero, CoM can be set equal to center_position
    # and children should be None
    if len(particles) == 0:
        return Octree_Node(
            particles=None,
            depth=depth,
            index=index,
            CoM=center_position,
            children=None,
        )
        
    # if this is a leaf node, return the node without children
    # compute the center_of_mass, since every particle has the same 
    # mass, we can simply take the average particle position:
    CoM = np.sum(particle_positions, axis=0)/len(particle_positions)
    
    # if max depth is reached return the node without creating children,
    # set particles equal to particle_positions.
    if depth == max_depth:
        return Octree_Node(
            particles=particles,
            depth=depth,
            index=index,
            CoM=CoM, # should calculate center of mass
            children=None,
        )
    # else: continue building the tree by recursively calling build on children

    # instantiate an empty array in which we will store the children
    children = np.empty((2, 2, 2), dtype=object)

    # precompute the children box sizes:
    child_box_size = 0.5*box_size
    
    # split the node into 8 child nodes  
    for offset_x in [0, 1]:
        for offset_y in [0, 1]:
            for offset_z in [0, 1]:
                # If the parent index is (ix, iy, iz), then the child index is
                # (2*ix + offset_x, 2*iy + offset_y, 2*iz + offset_z)
                offset = np.array([offset_x, offset_y, offset_z])
                child_index = 2*index + offset
                
                # mask for particles in this octant:
                # note that we can use the boolean property of the offsets (0, 1)
                child_particles = (
                    ((particle_positions[:,0] > center_position[0]) == offset[0]) &
                    ((particle_positions[:,1] > center_position[1]) == offset[1]) &
                    ((particle_positions[:,2] > center_position[2]) == offset[2])
                )
                child_particle_positions = particle_positions[child_particles]
                
                # child center is half a child box size separated from parent center
                child_center_position = center_position + 0.5*(2*offset - 1)*child_box_size
                
                # recursively call build on the children
                children[offset_x, offset_y, offset_z] = build_octree(
                    particles = child_particles,
                    particle_positions = child_particle_positions,
                    center_position = child_center_position,
                    index = child_index,
                    box_size = child_box_size,
                    depth = depth+1,
                    max_depth = max_depth,
                )
    
    return Octree_Node(
        particles=particles,
        index=index,
        depth=depth,
        CoM=CoM,
        children=children,
    )



def get_node_at_level(
    node,
    target_level,
    target_index,
):
    """
    Traverse the octree and return a node at a given level and index

    Parameters
    ----------
    node : Octree_Node
        Current node
    target_level : int
        Level to reach
    target_index : tuple
        Index of the target node at target_level, (ix, iy, iz)

    Returns
    -------
    node : Octree_Node or None
        Node at the requested level and index
    """

    # if the current node is already at the requested level,
    # check whether the index matches and return the node

    # work out which child recursively

    return None


def fill_massmap_from_octree(
    root,
    level,
    massmap,
):
    """
    Fill the four requested x-slices from the octree

    Parameters
    ----------
    root : Octree_Node
        Root of the octree
    level : int
        Octree level to plot
    massmap : ndarray
        Mass map, shape (4, pixels, pixels)

    Notes
    -----
    massmap[0,:,:] corresponds to x-index 0
    massmap[1,:,:] corresponds to x-index 1
    massmap[2,:,:] corresponds to x-index 2
    massmap[3,:,:] corresponds to x-index 3
    """

    pixels = 2**level

    # loop over the first four x index slices
    # and fill the corresponding y,z mass maps

    for ix in range(4):
        for iy in range(pixels):
            for iz in range(pixels):

                node = get_node_at_level(
                    node=root,
                    target_level=level,
                    target_index=(ix, iy, iz),
                )

                if node is not None:
                    massmap[ix, iy, iz] = node.mass

    return


def main() -> None:
    output_dir = "Plots"
    os.makedirs(output_dir, exist_ok=True)

    # Question 2: Calculating potentials

    with h5py.File("/disks/cosmodm/DMO_a0.1_256.hdf5", "r") as handle:
        pos = handle["Position"][...]  # particle positions, shape (Np,3), comoving
        # vel=handle["Velocity"][...] #particle velocities, shape (Np,3), comoving <-- not used, but if you're interested
    # Question 2a: using Barnes-Hut [note: not actually calculating a potential, unless you do the bonus question]

    # TO DO: build an octree

    root = build_octree(
        particle_positions=pos,
        center_position=np.array([L / 2, L / 2, L / 2], dtype=np.float32),
        index=(0, 0, 0),
        box_size=L,
        depth=0,
        max_depth=7,
    )
    # Plotting the mass distribution for a slice

    for level in [3, 5, 7]:  # feel free to change any of this code
        pixels = 2**level
        massmap = np.zeros((4, pixels, pixels), dtype=np.float32)
        # TO DO: traverse the octree, fill map massmap[0,:,:] with the masses of nodes at depth 3 and x_index=x_0,
        #        massmap[1,:,:] with the masses of nodes at depth 3 and x_index=x_1, etc; then plot these slices;
        #        then do the same for levels 5 and 7

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

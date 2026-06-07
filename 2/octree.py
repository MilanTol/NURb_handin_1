import numpy as np

Np = np.int64(256) ** 3  # number of particles
mp = np.float32(3.64453e10)  # particle mass in Msun; all 32-bit to save memory


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
        index: np.ndarray,
        depth: int,
        CoM: np.ndarray,
        children=None,
        particles=None,
    ):  # note: the fewer of these you keep, the better! these are just examples
        self.particles = particles
        self.depth = depth
        self.index = index
        self.CoM = CoM
        self.children = children

        if particles is None:
            self.mass = np.float32(0.0)
        else:
            self.mass = np.float32(len(particles)) * mp

    def get_center_position(self, box_dimensions: np.ndarray):
        square_dimensions = box_dimensions / (self.depth + 1)
        grid_point = self.index * square_dimensions
        offset = 0.5 * square_dimensions
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
    CoM = np.sum(particle_positions, axis=0) / len(particles)

    # if max depth is reached return the node without creating children,
    # set particles equal to particle_positions.
    if depth == max_depth:
        return Octree_Node(
            particles=particles,
            depth=depth,
            index=index,
            CoM=CoM,  # should calculate center of mass
            children=None,
        )
    # else: continue building the tree by recursively calling build on children

    # instantiate an empty array in which we will store the children
    children = np.empty((2, 2, 2), dtype=object)

    # precompute the children box sizes:
    child_box_size = 0.5 * box_size

    # split the node into 8 child nodes
    for offset_x in [0, 1]:
        for offset_y in [0, 1]:
            for offset_z in [0, 1]:
                # If the parent index is (ix, iy, iz), then the child index is
                # (2*ix + offset_x, 2*iy + offset_y, 2*iz + offset_z)
                offset = np.array([offset_x, offset_y, offset_z])
                child_index = 2 * index + offset

                # mask for particles in this octant:
                # note that we can use the boolean property of the offsets (0, 1)
                child_particles = (
                    ((particle_positions[:, 0] > center_position[0]) == offset[0])
                    & ((particle_positions[:, 1] > center_position[1]) == offset[1])
                    & ((particle_positions[:, 2] > center_position[2]) == offset[2])
                )
                child_particle_positions = particle_positions[child_particles]

                # child center is half a child box size separated from parent center
                child_center_position = (
                    center_position + 0.5 * (2 * offset - 1) * child_box_size
                )

                # recursively call build on the children
                children[offset_x, offset_y, offset_z] = build_octree(
                    particles=child_particles,
                    particle_positions=child_particle_positions,
                    center_position=child_center_position,
                    index=child_index,
                    box_size=child_box_size,
                    depth=depth + 1,
                    max_depth=max_depth,
                )

    return Octree_Node(
        particles=particles,
        index=index,
        depth=depth,
        CoM=CoM,
        children=children,
    )

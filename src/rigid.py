"""Rigid-body utilities: quaternions, rotations, and rigid-body moves."""

import numpy as np
from .utils import minimum_image


def quaternion_normalize(q):
    """Normalize a quaternion to unit length.
    
    Args:
        q: Quaternion, shape (4,)
        
    Returns:
        Normalized quaternion, shape (4,)
    """
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        raise ValueError("Quaternion norm is too small")
    return q / norm


def quaternion_multiply(q1, q2):
    """Multiply two quaternions: q1 * q2.
    
    Quaternion multiplication: (w1, x1, y1, z1) * (w2, x2, y2, z2)
    
    Args:
        q1: First quaternion, shape (4,)
        q2: Second quaternion, shape (4,)
        
    Returns:
        Product quaternion, shape (4,)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])


def quaternion_conjugate(q):
    """Compute conjugate of quaternion: (w, x, y, z) -> (w, -x, -y, -z).
    
    Args:
        q: Quaternion, shape (4,)
        
    Returns:
        Conjugate quaternion, shape (4,)
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_rotate_vector(q, v):
    """Rotate a vector by a quaternion.
    
    Rotation: v' = q * v * q* where v is treated as pure quaternion (0, vx, vy, vz)
    
    Args:
        q: Unit quaternion, shape (4,)
        v: Vector to rotate, shape (3,)
        
    Returns:
        Rotated vector, shape (3,)
    """
    # Treat v as pure quaternion (0, vx, vy, vz)
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    
    # Compute q * v * q*
    q_conj = quaternion_conjugate(q)
    temp = quaternion_multiply(q, v_quat)
    result_quat = quaternion_multiply(temp, q_conj)
    
    # Extract vector part (last 3 components)
    return result_quat[1:4]


def uniform_random_orientation(rng):
    """Sample a uniform random orientation on SO(3) using quaternions.
    
    Uses the method: sample q = (w, x, y, z) where w, x, y, z are independent
    standard normal, then normalize. This gives uniform distribution on unit
    quaternions, which maps 2-to-1 to SO(3).
    
    Args:
        rng: Random number generator (numpy.random.Generator)
        
    Returns:
        Unit quaternion representing the orientation, shape (4,)
    """
    q = rng.normal(size=4)
    return quaternion_normalize(q)


def apply_rigid_transform(R, q, s_body, L):
    """Apply rigid transform: map body-frame site coordinates to lab coordinates.
    
    r_a = R + rotate(q, s_a) with periodic wrap
    
    Args:
        R: Center of mass position in lab frame, shape (3,)
        q: Orientation quaternion, shape (4,)
        s_body: Body-frame site coordinates, shape (N_sites, 3)
        L: Box length for periodic wrapping
        
    Returns:
        Lab-frame site coordinates, shape (N_sites, 3), wrapped to [0, L)
    """
    # Rotate each body-frame site
    r_lab = np.zeros_like(s_body)
    for i, s in enumerate(s_body):
        r_lab[i] = R + quaternion_rotate_vector(q, s)
    
    # Apply periodic wrapping
    if L is None:
        return r_lab
    return r_lab % L


def rigid_body_move_proposal(R, q, max_disp, max_angle, rng):
    """Propose a rigid-body move: translation + rotation.
    
    Translation: uniform in cube of side 2*max_disp centered at R
    Rotation: small random rotation with max angle max_angle about random axis
    
    Args:
        R: Current center of mass position, shape (3,)
        q: Current orientation quaternion, shape (4,)
        max_disp: Maximum displacement (half side of translation cube)
        max_angle: Maximum rotation angle in radians
        rng: Random number generator (numpy.random.Generator)
        
    Returns:
        Tuple (R_new, q_new):
        - R_new: New center of mass position, shape (3,)
        - q_new: New orientation quaternion, shape (4,)
    """
    # Translation: uniform in cube [-max_disp, max_disp]^3
    disp = (rng.random(3) * 2 - 1) * max_disp
    R_new = R + disp
    
    # Rotation: small random rotation
    # Generate random axis (unit vector)
    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    
    # Generate random angle in [0, max_angle]
    angle = rng.random() * max_angle
    
    # Create rotation quaternion: q_rot = (cos(θ/2), sin(θ/2) * axis)
    half_angle = 0.5 * angle
    q_rot = np.array([
        np.cos(half_angle),
        np.sin(half_angle) * axis[0],
        np.sin(half_angle) * axis[1],
        np.sin(half_angle) * axis[2]
    ])
    
    # Apply rotation: q_new = q_rot * q
    q_new = quaternion_multiply(q_rot, q)
    q_new = quaternion_normalize(q_new)
    
    return R_new, q_new


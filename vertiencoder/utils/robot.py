import numpy as np
import torch
import rospy
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from nav_msgs.msg import Path, Odometry
import tf.transformations
import tf
import warnings


# General functions
def angle_to_quaternion(angle):
    """Convert an angle in radians into a quaternion _message_."""
    if not isinstance(angle, np.ndarray):
        angle = np.array(angle, dtype=np.float32)
    if angle.shape[-1] != 3:
        raise ValueError(f"Input must have last dim equal to 3 got {angle.shape[-1]}")
    if len(angle.shape) > 2:
        raise ValueError(f"Input tensor must be 1D or 2D got {len(angle.shape)}")
    if len(angle.shape) == 1:
        return Quaternion(
            *tf.transformations.quaternion_from_euler(
                angle[0], angle[1], angle[2], axes="sxyz"
            )
        )
    if len(angle.shape) == 2:
        return [
            Quaternion(
                *tf.transformations.quaternion_from_euler(
                    ang_[0], ang_[1], ang_[2], axes="sxyz"
                )
            )
            for ang_ in angle
        ]


def quaternion_to_angle(q):
    """Convert a quaternion into an angle in radians."""
    if not isinstance(q, Quaternion) and not isinstance(q, list):
        raise ValueError(f"Input must be of type Quaternion or list of Quaternions")
    if isinstance(q, list):
        return np.array(
            [
                tf.transformations.euler_from_quaternion(
                    (quat.x, quat.y, quat.z, quat.w), axes="sxyz"
                )
                for quat in q
            ],
            dtype=np.float32,
        ).tolist()
    if isinstance(q, Quaternion):
        return list(
            tf.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w), axes="sxyz")
        )


def clamp_angle(angles):
    if not isinstance(angles, np.ndarray):
        t = type(angles)
        angles = np.array(angles, dtype=np.float32)

    angles += np.pi
    angles %= 2 * np.pi
    angles -= np.pi
    if t == list:
        return angles.tolist()
    else:
        return t(angles)


def map_range(value, from_min, from_max, to_min, to_max):
    # Calculate the range of the input value
    from_range = from_max - from_min

    # Calculate the range of the output value
    to_range = to_max - to_min

    # Scale the input value to the output range
    mapped_value = (value - from_min) * (to_range / from_range) + to_min

    return mapped_value


def get_euclidean_dist(start_pose, goal_pose):
    if not isinstance(start_pose, torch.Tensor):
        start_pose = torch.tensor(start_pose, dtype=torch.float32)
    if not isinstance(goal_pose, torch.Tensor):
        goal_pose = torch.tensor(goal_pose, dtype=torch.float32)
    if len(start_pose.shape) == 1:
        start_pose = start_pose.unsqueeze(0)
    if len(goal_pose.shape) == 1:
        goal_pose = goal_pose.unsqueeze(0)
    if start_pose.shape[-1] != 3 and start_pose.shape[-1] != 6:
        raise ValueError(
            f"Input tensors must have last dim equal to 3 for SE2 and 6 for SE3 got {start_pose.shape[-1]} on start_pose"
        )
    if goal_pose.shape[-1] != 3 and goal_pose.shape[-1] != 6:
        raise ValueError(
            f"Input tensors must have last dim equal to 3 for SE2 and 6 for SE3 got {goal_pose.shape[-1]} on goal_pose"
        )
    if start_pose.shape[-1] != goal_pose.shape[-1]:
        warnings.warn(
            "start_pose and goal_pose are not in same SE space", category=UserWarning
        )

    diff = goal_pose[:, :2] - start_pose[:, :2]
    distance = torch.norm(diff, dim=1).squeeze()
    if len(distance.shape) == 0:
        return distance.item()

    return distance


def odometry_to_particle(odom_msg):
    if not isinstance(odom_msg, Odometry):
        raise ValueError("Input must be of type Odometry")
    x, y, z = (
        odom_msg.pose.pose.position.x,
        odom_msg.pose.pose.position.y,
        odom_msg.pose.pose.position.z,
    )
    roll, pitch, yaw = quaternion_to_angle(odom_msg.pose.pose.orientation)
    return [x, y, z, roll, pitch, yaw]


# ROS related functions
def particle_to_posestamped(particle, frame_id):
    if not isinstance(particle, np.ndarray):
        particle = np.array(particle, dtype=np.float32)

    pose = PoseStamped()
    pose.header = make_header(frame_id)
    pose.pose.position.x = particle[0]
    pose.pose.position.y = particle[1]
    pose.pose.position.z = particle[2]
    pose.pose.orientation = angle_to_quaternion(particle[3:])
    return pose


def particle_to_pose(particle):
    if not isinstance(particle, np.ndarray):
        particle = np.array(particle, np.float32)

    pose = Pose()
    pose.position.x = particle[0]
    pose.position.y = particle[1]
    pose.position.z = particle[2]
    pose.orientation = angle_to_quaternion(particle[3:])
    return pose


def particles_to_poses(particles):
    return list(map(particle_to_pose, particles))


def particles_to_poses_stamped(particles, frame_id):
    return list(map(particle_to_posestamped, particles, [frame_id] * len(particles)))


def make_header(frame_id, stamp=None):
    if stamp == None:
        stamp = rospy.Time.now()
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return header


def visualize(
    publisher,
    poses,
    frame_id="odom",
):
    if publisher.get_num_connections() > 0:
        path_msg = Path()
        path_msg.header = make_header(frame_id)
        path_msg.poses = particles_to_poses_stamped(poses, frame_id)
        publisher.publish(path_msg)
        print("path published")


# Model and transformation related functions
def ackermann(throttle, steering, wheel_base=0.324, dt=0.1):
    if not isinstance(throttle, torch.Tensor):
        throttle = torch.tensor(throttle, dtype=torch.float32)
    if not isinstance(steering, torch.Tensor):
        steering = torch.tensor(steering, dtype=torch.float32)
    if not isinstance(wheel_base, torch.Tensor):
        wheel_base = torch.tensor(wheel_base, dtype=torch.float32)
    if not isinstance(dt, torch.Tensor):
        dt = torch.tensor(dt, dtype=torch.float32)
    if throttle.shape != steering.shape:
        raise ValueError("throttle and steering must have the same shape")
    if len(throttle.shape) == 0:
        throttle = throttle.unsqueeze(0)

    deltaPose = torch.zeros(throttle.shape[0], 6, dtype=torch.float32)

    dtheta = (throttle / wheel_base) * torch.tan(steering) * dt
    dx = throttle * torch.cos(dtheta) * dt
    dy = throttle * torch.sin(dtheta) * dt

    deltaPose[:, 0], deltaPose[:, 1], deltaPose[:, 5] = dx, dy, dtheta

    return deltaPose.squeeze()


def euler_to_rotation_matrix(euler_angles):
    """Convert Euler angles to a rotation matrix"""
    # Compute sin and cos for Euler angles
    cos = torch.cos(euler_angles)
    sin = torch.sin(euler_angles)
    zero = torch.zeros_like(euler_angles[:, 0])
    one = torch.ones_like(euler_angles[:, 0])
    # Constructing rotation matrices (assuming 'xyz' convention for Euler angles)
    R_x = torch.stack(
        [one, zero, zero, zero, cos[:, 0], -sin[:, 0], zero, sin[:, 0], cos[:, 0]],
        dim=1,
    ).view(-1, 3, 3)
    R_y = torch.stack(
        [cos[:, 1], zero, sin[:, 1], zero, one, zero, -sin[:, 1], zero, cos[:, 1]],
        dim=1,
    ).view(-1, 3, 3)
    R_z = torch.stack(
        [cos[:, 2], -sin[:, 2], zero, sin[:, 2], cos[:, 2], zero, zero, zero, one],
        dim=1,
    ).view(-1, 3, 3)

    return torch.matmul(torch.matmul(R_z, R_y), R_x)


def extract_euler_angles_from_se3_batch(tf3_matx):
    # Validate input shape
    if tf3_matx.shape[1:] != (4, 4):
        raise ValueError("Input tensor must have shape (batch, 4, 4)")

    # Extract rotation matrices
    rotation_matrices = tf3_matx[:, :3, :3]

    # Initialize tensor to hold Euler angles
    batch_size = tf3_matx.shape[0]
    euler_angles = torch.zeros(
        (batch_size, 3), device=tf3_matx.device, dtype=tf3_matx.dtype
    )

    # Compute Euler angles
    euler_angles[:, 0] = torch.atan2(
        rotation_matrices[:, 2, 1], rotation_matrices[:, 2, 2]
    )  # Roll
    euler_angles[:, 1] = torch.atan2(
        -rotation_matrices[:, 2, 0],
        torch.sqrt(rotation_matrices[:, 2, 1] ** 2 + rotation_matrices[:, 2, 2] ** 2),
    )  # Pitch
    euler_angles[:, 2] = torch.atan2(
        rotation_matrices[:, 1, 0], rotation_matrices[:, 0, 0]
    )  # Yaw

    return euler_angles


def to_robot_torch(Robot_frame, P_relative):
    SE3 = True

    if not isinstance(Robot_frame, torch.Tensor):
        Robot_frame = torch.tensor(Robot_frame, dtype=torch.float32)

    if not isinstance(P_relative, torch.Tensor):
        P_relative = torch.tensor(P_relative, dtype=torch.float32)

    if len(Robot_frame.shape) == 1:
        Robot_frame = Robot_frame.unsqueeze(0)

    if len(P_relative.shape) == 1:
        P_relative = P_relative.unsqueeze(0)

    if len(Robot_frame.shape) > 2 or len(P_relative.shape) > 2:
        raise ValueError(
            f"Input must be 1D for  unbatched and 2D for batched got input dimensions {Robot_frame.shape} and {P_relative.shape}"
        )

    if Robot_frame.shape != P_relative.shape:
        raise ValueError("Input tensors must have same shape")

    if Robot_frame.shape[-1] != 6 and Robot_frame.shape[-1] != 3:
        raise ValueError(
            f"Input tensors must have last dim equal to 6 for SE3 and 3 for SE2 got {Robot_frame.shape[-1]}"
        )

    if Robot_frame.shape[-1] == 3:
        SE3 = False
        Robot_frame_ = torch.zeros(
            (Robot_frame.shape[0], 6),
            device=Robot_frame.device,
            dtype=Robot_frame.dtype,
        )
        Robot_frame_[:, [0, 1, 5]] = Robot_frame
        Robot_frame = Robot_frame_
        P_relative_ = torch.zeros(
            (P_relative.shape[0], 6), device=P_relative.device, dtype=P_relative.dtype
        )
        P_relative_[:, [0, 1, 5]] = P_relative
        P_relative = P_relative_

    """ Assemble a batch of SE3 homogeneous matrices from a batch of 6DOF poses """
    batch_size = Robot_frame.shape[0]
    ones = torch.ones_like(P_relative[:, 0])
    transform = torch.zeros_like(Robot_frame)
    T1 = torch.zeros(
        (batch_size, 4, 4), device=Robot_frame.device, dtype=Robot_frame.dtype
    )
    T2 = torch.zeros(
        (batch_size, 4, 4), device=P_relative.device, dtype=P_relative.dtype
    )

    T1[:, :3, :3] = euler_to_rotation_matrix(Robot_frame[:, 3:])
    T2[:, :3, :3] = euler_to_rotation_matrix(P_relative[:, 3:])
    T1[:, :3, 3] = Robot_frame[:, :3]
    T2[:, :3, 3] = P_relative[:, :3]
    T1[:, 3, 3] = 1
    T2[:, 3, 3] = 1

    T1_inv = torch.inverse(T1)
    tf3_mat = torch.matmul(T2, T1_inv)

    transform[:, :3] = torch.matmul(
        T1_inv, torch.cat((P_relative[:, :3], ones.unsqueeze(-1)), dim=1).unsqueeze(2)
    ).squeeze(dim=2)[:, :3]
    transform[:, 3:] = extract_euler_angles_from_se3_batch(tf3_mat)

    if not SE3:
        transform = transform[:, [0, 1, 5]]

    return transform


def to_world_torch(Robot_frame, P_relative):
    SE3 = True

    if not isinstance(Robot_frame, torch.Tensor):
        Robot_frame = torch.tensor(Robot_frame, dtype=torch.float32)
    if not isinstance(P_relative, torch.Tensor):
        P_relative = torch.tensor(P_relative, dtype=torch.float32)

    if len(Robot_frame.shape) == 1:
        Robot_frame = Robot_frame.unsqueeze(0)

    if len(P_relative.shape) == 1:
        P_relative = P_relative.unsqueeze(0)

    if len(Robot_frame.shape) > 2 or len(P_relative.shape) > 2:
        raise ValueError(
            f"Input must be 1D for  unbatched and 2D for batched got input dimensions {Robot_frame.shape} and {P_relative.shape}"
        )

    if Robot_frame.shape != P_relative.shape:
        raise ValueError("Input tensors must have same shape")

    if Robot_frame.shape[-1] != 6 and Robot_frame.shape[-1] != 3:
        raise ValueError(
            f"Input tensors must have last dim equal to 6 for SE3 and 3 for SE2 got {Robot_frame.shape[-1]}"
        )

    if Robot_frame.shape[-1] == 3:
        SE3 = False
        Robot_frame_ = torch.zeros(
            (Robot_frame.shape[0], 6),
            device=Robot_frame.device,
            dtype=Robot_frame.dtype,
        )
        Robot_frame_[:, [0, 1, 5]] = Robot_frame
        Robot_frame = Robot_frame_
        P_relative_ = torch.zeros(
            (P_relative.shape[0], 6), device=P_relative.device, dtype=P_relative.dtype
        )
        P_relative_[:, [0, 1, 5]] = P_relative
        P_relative = P_relative_

    """ Assemble a batch of SE3 homogeneous matrices from a batch of 6DOF poses """
    batch_size = Robot_frame.shape[0]
    ones = torch.ones_like(P_relative[:, 0])
    transform = torch.zeros_like(Robot_frame)
    T1 = torch.zeros(
        (batch_size, 4, 4), device=Robot_frame.device, dtype=Robot_frame.dtype
    )
    T2 = torch.zeros(
        (batch_size, 4, 4), device=P_relative.device, dtype=P_relative.dtype
    )

    R1 = euler_to_rotation_matrix(Robot_frame[:, 3:])
    R2 = euler_to_rotation_matrix(P_relative[:, 3:])

    T1[:, :3, :3] = R1
    T2[:, :3, :3] = R2
    T1[:, :3, 3] = Robot_frame[:, :3]
    T2[:, :3, 3] = P_relative[:, :3]
    T1[:, 3, 3] = 1
    T2[:, 3, 3] = 1

    T_tf = torch.matmul(T2, T1)
    transform[:, :3] = torch.matmul(
        T1, torch.cat((P_relative[:, :3], ones.unsqueeze(-1)), dim=1).unsqueeze(2)
    ).squeeze(dim=2)[:, :3]
    transform[:, 3:] = extract_euler_angles_from_se3_batch(T_tf)

    if not SE3:
        transform = transform[:, [0, 1, 5]]

    return transform

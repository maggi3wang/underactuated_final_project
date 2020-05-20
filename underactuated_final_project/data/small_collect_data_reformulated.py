import argparse
import os
import tkinter as tk
import numpy as np

from pydrake.common.eigen_geometry import Quaternion
from pydrake.examples.manipulation_station import ManipulationStation
from pydrake.geometry import ConnectDrakeVisualizer

from pydrake.manipulation.simple_ui import SchunkWsgButtons
from pydrake.manipulation.planner import (
    DifferentialInverseKinematicsParameters)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import BasicVector, DiagramBuilder, LeafSystem
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.primitives import FirstOrderLowPassFilter

# DifferentialIK from https://github.com/RobotLocomotion/drake/blob/fcccd9341b508c86027afa4462738d067d52454d/examples/manipulation_station/differential_ik.py
from underactuated_final_project.differential_ik import DifferentialIK

class EndEffectorTeleop(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self.DeclareVectorOutputPort("rpy_xyz", BasicVector(6),
                                     self.DoCalcOutput)

        self.roll = self.pitch = self.yaw = 0
        self.x = self.y = self.z = 0

        self.window = tk.Tk()

    def SetPose(self, pose):
        tf = RigidTransform(pose)
        self.SetRPY(RollPitchYaw(tf.rotation()))
        self.SetXYZ(tf.translation())

    def SetRPY(self, rpy):
        self.roll = rpy.roll_angle()
        self.pitch = rpy.pitch_angle()
        self.yaw = rpy.yaw_angle()

    def SetXYZ(self, xyz):
        self.x = xyz[0]
        self.y = xyz[1]
        self.z = xyz[2]

    def DoCalcOutput(self, context, output):
        output.SetAtIndex(0, self.roll)
        output.SetAtIndex(1, self.pitch)
        output.SetAtIndex(2, self.yaw)
        output.SetAtIndex(3, self.x)
        output.SetAtIndex(4, self.y)
        output.SetAtIndex(5, self.z)

def create_se3_matrix(vel):
    vel_se3 = np.zeros((4, 4))
    vel_se3[1, 2] = -vel[0]
    vel_se3[2, 1] = vel[0]

    vel_se3[2, 0] = -vel[1]
    vel_se3[0, 2] = vel[1]

    vel_se3[0, 1] = -vel[2]
    vel_se3[1, 0] = vel[2]

    vel_se3[0:3, 3] = vel[3:6]
    return vel_se3

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def normalize(x):
    return x / np.linalg.norm(x)

def compute_V(theta, differential_ik, X_G):
    X_H = differential_ik.ForwardKinematics(theta).matrix()
    V = np.linalg.norm(X_H - X_G)**2
    return V

def main():
    np.set_printoptions(precision=3, suppress=True)

    # Desired rate relative to real time
    target_realtime_rate = 1.0

    # Time constant for the first order low pass filter applied to the teleop commands
    filter_time_const = 0.1

    # Further limit iiwa joint velocities
    velocity_limit_factor = 0.1

    builder = DiagramBuilder()
    station = builder.AddSystem(ManipulationStation())
    station.SetupManipulationClassStation()
    mbp = station.get_multibody_plant()

    station.AddManipulandFromFile(
        "drake/examples/manipulation_station/models/061_foam_brick.sdf",
        RigidTransform(RotationMatrix.Identity(), [0.7, -0.25, 0]))

    station.Finalize()

    ConnectDrakeVisualizer(builder, station.get_scene_graph(),
                           station.GetOutputPort("pose_bundle"))

    robot = station.get_controller_plant()
    num_joints = robot.num_positions()
    params = DifferentialInverseKinematicsParameters(num_joints,
                                                     robot.num_velocities())

    time_step = 0.005
    params.set_timestep(time_step)
    iiwa14_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])

    factor = velocity_limit_factor
    params.set_joint_velocity_limits((-factor*iiwa14_velocity_limits,
                                      factor*iiwa14_velocity_limits))
    differential_ik = builder.AddSystem(DifferentialIK(
        robot, robot.GetFrameByName("iiwa_link_7"), params, time_step))

    builder.Connect(differential_ik.GetOutputPort("joint_position_desired"),
                    station.GetInputPort("iiwa_position"))

    teleop = builder.AddSystem(EndEffectorTeleop())

    filter_ = builder.AddSystem(
        FirstOrderLowPassFilter(time_constant=filter_time_const, size=6))

    builder.Connect(teleop.get_output_port(0), filter_.get_input_port(0))
    builder.Connect(filter_.get_output_port(0),
                    differential_ik.GetInputPort("rpy_xyz_desired"))

    teleop.window.withdraw()
    schunk_wsg_buttons = SchunkWsgButtons(window=teleop.window)
    wsg_buttons = builder.AddSystem(schunk_wsg_buttons)
    builder.Connect(wsg_buttons.GetOutputPort("position"), station.GetInputPort("wsg_position"))
    builder.Connect(wsg_buttons.GetOutputPort("force_limit"),
                    station.GetInputPort("wsg_force_limit"))

    diagram = builder.Build()
    simulator = Simulator(diagram)

    diagram_context = diagram.CreateDefaultContext()

    # This is important to avoid duplicate publishes to the hardware interface:
    simulator.set_publish_every_time_step(False)

    station_context = diagram.GetMutableSubsystemContext(
        station, simulator.get_mutable_context())

    mbp_context = station.GetMutableSubsystemContext(station, station_context)

    station.GetInputPort("iiwa_feedforward_torque").FixValue(
        station_context, np.zeros(station.num_iiwa_joints()))

    simulator.AdvanceTo(1e-6)
    q0 = station.GetOutputPort("iiwa_position_measured").Eval(station_context)
    differential_ik.parameters.set_nominal_joint_position(q0)

    teleop.SetPose(differential_ik.ForwardKinematics(q0))
    filter_.set_initial_output_value(
        diagram.GetMutableSubsystemContext(
            filter_, simulator.get_mutable_context()),
        teleop.get_output_port(0).Eval(diagram.GetMutableSubsystemContext(
            teleop, simulator.get_mutable_context())))
    differential_ik.SetPositions(diagram.GetMutableSubsystemContext(
        differential_ik, simulator.get_mutable_context()), q0)

    #some_q = [-0., 0.1, 0.3, -1, -0.3, 1., 0.]
    #some_q = [-0., 0.0, 0.5, -1, -0.3, 0., 0.]
    #differential_ik.SetPositions(diagram.GetMutableSubsystemContext(
    #    differential_ik, simulator.get_mutable_context()), some_q)

    t = 1
    simulator.AdvanceTo(t)

    # Joint angle inputs
    theta = station.GetIiwaPosition(station_context)
    print('theta', theta)

    q_prev = station.GetIiwaPosition(station_context)
    prev_V = np.inf

    while t < 131 + time_step:
        print('\n')

        # Goal in world frame
        goal_pose = mbp.GetPositions(mbp_context).copy()
        print('goal_pose', goal_pose)
        X_G = np.identity(4)
        X_G[0:3, 0:3] = RotationMatrix(quaternion=Quaternion(normalize(goal_pose[0:4]))).matrix()
        X_G[0:3, 3] = goal_pose[4:7]
        X_G[2, 3] += 0.22

        R_y = np.identity(4)
        R_y[0, 0] = -1
        R_y[2, 2] = -1

        X_G = np.dot(X_G, R_y)      # Rotate X_G around y-axis by 180 deg
        R_G = X_G[0:3, 0:3]
        p_G = X_G[0:3, 3]
        print(R_G, p_G)
        print('X_G: \n{}'.format(X_G))

        # End effector in world frame
        X_H = differential_ik.ForwardKinematics(station.GetIiwaPosition(station_context)).matrix()
        R_H = X_H[0:3, 0:3]
        p_H = X_H[0:3, 3]

        print('X_H: \n{}'.format(X_H))

        X_HG = np.linalg.solve(X_H, X_G)
        print('X_HG: {}'.format(X_HG))

        # Calculate Lyapunov function
        V = np.linalg.norm(X_HG - np.identity(4))**2
        print('V: {}'.format(V))

        # Calculate Lyapunov function
        V = np.linalg.norm(X_H - X_G)**2
        assert(abs(6 - 2 * np.trace(np.dot(R_G.transpose(), R_H)) + 
            np.dot((p_H - p_G).transpose(), (p_H - p_G)) - V) < 0.00001)
        print('V: {}'.format(V))

        # assert(V <= prev_V + 0.0001)   # This assertion is failing even though it shouldn't!
        prev_V = V

        # Calculate numerical gradient
        delta_theta = 1e-7

        pdV_numerical = np.zeros(7)
        for i in range(7):
            theta_upper_perturbed = station.GetIiwaPosition(station_context)
            theta_upper_perturbed[i] = theta_upper_perturbed[i] + delta_theta

            theta_lower_perturbed = station.GetIiwaPosition(station_context)
            theta_lower_perturbed[i] = theta_lower_perturbed[i] - delta_theta

            pdV_numerical_1 = compute_V(theta_upper_perturbed, differential_ik, X_G)
            pdV_numerical_2 = compute_V(theta_lower_perturbed, differential_ik, X_G)

            pdV_numerical[i] = (pdV_numerical_1 - pdV_numerical_2) / (2.0 * delta_theta)

        print('pdV_numerical: {}'.format(pdV_numerical))

        # Calculate Lyapunov derivatives

        # Calculate Jacobian
        jacobian = robot.CalcJacobianSpatialVelocity(
            context=differential_ik.GetRobotContext(), with_respect_to=JacobianWrtVariable.kQDot,
            frame_B=robot.GetFrameByName("iiwa_link_7"),    # frame on which point Bi is fixed
            p_BP=[0.0, 0.0, 0.0],                           # position vec from frame_B's origin to points Bi
            frame_A=robot.world_frame(),                    # frame that measures v_ABi
            frame_E=robot.world_frame())                    # frame that v_ABi is expressed on input 
                                                            # and frame that Jacobian J_V_ABi is expressed on output
        print('jacobian: \n{}'.format(jacobian))
        pdV = np.zeros(num_joints)
        ang_jacobian = jacobian[0:3, 0:7]
        trans_jacobian = jacobian[3:6, 0:7]

        for i in range(7):
            rot_jacobian = skew(ang_jacobian[0:3, i]) @ R_H
            pdV[i] = -2 * np.trace(R_G @ rot_jacobian) + 30 * (p_H - p_G) @ trans_jacobian[0:3, i]

        print('pdV: {}'.format(pdV))
        mu = max(1.0/(V*20), 0.1)
        print('mu', mu)
        V_dot = -mu * np.linalg.norm(pdV)**2
        print('V_dot: {}'.format(V_dot))

        theta_dot = -mu * pdV
        print('theta_dot: \n{}'.format(theta_dot))

        q_prev = station.GetIiwaPosition(station_context)
        print('q_prev: {}'.format(q_prev))
        q = q_prev + time_step * theta_dot
        print('q: {}'.format(q))
        differential_ik.SetPositions(diagram.GetMutableSubsystemContext(
            differential_ik, simulator.get_mutable_context()), q)

        # Why doesn't this velocity match theta_dot?
        print('iiwa vel: {}'.format(station.GetIiwaVelocity(station_context)))
        simulator.set_target_realtime_rate(target_realtime_rate)
        simulator.AdvanceTo(t)
        t += time_step

        if abs(X_HG[0, 3]) < 0.03 and abs(X_HG[1, 3]) < 0.03 and abs(X_HG[2, 3]) < 0.03:
            print('close enough to grasp, {}, {}, {}'.format(X_HG[0, 3], X_HG[1, 3], X_HG[2, 3]))
            break

if __name__ == "__main__":
    main()
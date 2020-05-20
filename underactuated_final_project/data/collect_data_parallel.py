import argparse
import os
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pandas as pd
import random
import tkinter as tk
import gc
import sys
import time

from pydrake.common.eigen_geometry import Quaternion
from pydrake.examples.manipulation_station import ManipulationStation
from pydrake.geometry import ConnectDrakeVisualizer
from pydrake.geometry.render import DepthCameraProperties, MakeRenderEngineVtk, RenderEngineVtkParams

from pydrake.manipulation.simple_ui import SchunkWsgButtons
from pydrake.manipulation.planner import DifferentialInverseKinematicsParameters
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.systems.sensors import RgbdSensor, Image as PydrakeImage, PixelType, PixelFormat
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import (AbstractValue, BasicVector, DiagramBuilder, LeafSystem)
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.primitives import FirstOrderLowPassFilter

from underactuated_final_project.differential_ik import DifferentialIK


class EndEffectorTeleop(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self.DeclareVectorOutputPort("rpy_xyz", BasicVector(6),
                                     self.DoCalcOutput)
        self.DeclareVectorOutputPort("position", BasicVector(1),
                                     self.CalcPositionOutput)
        self.DeclareVectorOutputPort("force_limit", BasicVector(1),
                                     self.CalcForceLimitOutput)

        self.DeclarePeriodicPublish(0.01, 0.0)

        self.roll = self.pitch = self.yaw = 0
        self.x = self.y = self.z = 0
        self.gripper_max = 0.107
        self.gripper_min = 0.01
        self.gripper_goal = self.gripper_max

        self.window = tk.Tk()

        def update(scale, value):
            return lambda event: scale.set(scale.get() + value)

        # Delta displacements for motion via keyboard teleop.
        rotation_delta = 0.05  # rad
        position_delta = 0.01  # m

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

    def CalcPositionOutput(self, context, output):
        output.SetAtIndex(0, self.gripper_goal)

    def CalcForceLimitOutput(self, context, output):
        self._force_limit = 40
        output.SetAtIndex(0, self._force_limit)

    def DoCalcOutput(self, context, output):
        output.SetAtIndex(0, self.roll)
        output.SetAtIndex(1, self.pitch)
        output.SetAtIndex(2, self.yaw)
        output.SetAtIndex(3, self.x)
        output.SetAtIndex(4, self.y)
        output.SetAtIndex(5, self.z)


class RgbImageVisualizer(LeafSystem):
    def __init__(self, draw_timestep=0.25):
        LeafSystem.__init__(self)
        self.set_name('image viz')
        self.timestep = draw_timestep
        self.DeclarePeriodicPublish(draw_timestep, 0.0)
        
        self.rgb_image_input_port = \
            self.DeclareAbstractInputPort("rgb_image_input_port",
                AbstractValue.Make(PydrakeImage[PixelType.kRgba8U](224, 224, 3)))

        self.color_image = None

    def DoPublish(self, context, event):
        """
        Update color_image and label_image for saving
        """
        self.color_image = self.EvalAbstractInput(context, 0).get_value()

    def save_image(self, filename):
        """
        Save images to a file
        """
        color_fig = plt.imshow(self.color_image.data)
        plt.axis('off')
        color_fig.axes.get_xaxis().set_visible(False)
        color_fig.axes.get_yaxis().set_visible(False)
        plt.savefig(filename + '.png', bbox_inches='tight', pad_inches=0)


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

def run_trial(trial_num, write_q):
    start = time.process_time()
    t = 0

    # Time constant for the first order low pass filter applied to the teleop commands
    filter_time_const = 0.1

    # Further limit iiwa joint velocities
    velocity_limit_factor = 0.1

    builder = DiagramBuilder()
    station = builder.AddSystem(ManipulationStation())

    # Initializes the chosen station type.
    station.SetupIiwaOnTableStation()

    mbp = station.get_multibody_plant()
    parser = Parser(mbp)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../mug_clean/brick.sdf')
    mug = parser.AddModelFromFile(model_path)

    # Add camera
    renderer_params = RenderEngineVtkParams()
    station.get_scene_graph().AddRenderer("renderer", MakeRenderEngineVtk(renderer_params))

    kFocalY = 300.0
    kHeight = 224
    kWidth = 224
    fov_y = math.atan(kHeight / 2. / kFocalY) * 2
    parent_frame_id = station.get_scene_graph().world_frame_id()
    camera_properties = DepthCameraProperties(kWidth, kHeight, fov_y, "renderer", z_near=0.1, z_far=2.0)
    #camera_tf = RigidTransform(rpy=RollPitchYaw([2*np.pi/4, np.pi, -2*np.pi/4]), p=[2.4, 0, 0.5])
    camera_tf = RigidTransform(rpy=RollPitchYaw([2*np.pi/4-0.5, np.pi, -2*np.pi/4]), p=[1.5, 0.0, 0.6])
    station.RegisterRgbdSensor("0", mbp.world_frame(), camera_tf, camera_properties)

    rgb_image_visualizer = RgbImageVisualizer()
    camera_viz = builder.AddSystem(rgb_image_visualizer)

    station.Finalize()

    ConnectDrakeVisualizer(builder, station.get_scene_graph(),
                           station.GetOutputPort("pose_bundle"))

    for name in station.get_camera_names():     # TODO if time: add additional camera
        builder.Connect(
            station.GetOutputPort("camera_" + name + "_rgb_image"),
            camera_viz.get_input_port(0))

    robot = station.get_controller_plant()
    num_joints = robot.num_positions()
    params = DifferentialInverseKinematicsParameters(num_joints,
                                                     robot.num_velocities())

    time_step = 0.005
    params.set_timestep(time_step)
    # True velocity limits for the IIWA14 (in rad, rounded down to the first
    # decimal)
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
    simulator.set_target_realtime_rate(0)

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

    # Set random mug position
    mug_rot = RotationMatrix(RollPitchYaw(0, 0, random.uniform(-np.pi/2.0, np.pi/2.0)))
    mug_trans = [random.uniform(0.4, 0.7), random.uniform(-0.2, 0.2), 0.01]

    X_TObject = RigidTransform(mug_rot, mug_trans)
    # print('X_TObject', X_TObject.matrix())

    mbp.SetFreeBodyPose(
        mbp_context,
        mbp.GetBodyByName("base_link", mug), #mbp.GetBodyByName("mug_body_link", mug),
        X_TObject)

    mug_q0 = mbp.GetPositions(mbp_context).copy()
    mbp.SetPositions(mbp_context, mug_q0)
    # print('mug_q0', mug_q0)

    t += 1
    simulator.AdvanceTo(t)

    prev_V = np.inf
    close_enough_to_grasp = False

    filenames = []
    data_to_write = ''

    num = 0
    start_t = t
    image_num = 0

    while t - start_t < 40:
        # print('\n')

        # Goal in world frame
        mug_pose = mbp.GetPositions(mbp_context).copy()    # Quaternion to rotation matrix
        # print('mug_pose', mug_pose)
        X_G = np.identity(4)
        X_G[0:3, 0:3] = RotationMatrix(quaternion=Quaternion(normalize(mug_pose[0:4]))).matrix()
        X_G[0:3, 3] = mug_pose[4:7]
        if (X_G[2, 3] < -0.01):
            break

        X_G[2, 3] += 0.26

        R_y = np.identity(4)
        R_y[0, 0] = -1
        R_y[2, 2] = -1

        X_G = np.dot(X_G, R_y)      # Rotate X_G around y-axis by 180 deg
        R_G = X_G[0:3, 0:3]
        p_G = X_G[0:3, 3]
        # print('X_G: \n{}'.format(X_G))

        # End effector in world frame
        X_H = differential_ik.ForwardKinematics(station.GetIiwaPosition(station_context)).matrix()
        R_H = X_H[0:3, 0:3]
        p_H = X_H[0:3, 3]
        # print('X_H: \n{}'.format(X_H))

        # Goal pose relative to the end effector
        X_HG = np.linalg.solve(X_H, X_G)
        # print('X_HG: \n{}'.format(X_HG))

        # Calculate Lyapunov function
        V = np.linalg.norm(X_HG - np.identity(4))**2
        # print('V: {}'.format(V))

        prev_V = V

        # Calculate Lyapunov derivatives

        # Calculate Jacobian
        jacobian = robot.CalcJacobianSpatialVelocity(
            context=differential_ik.GetRobotContext(), with_respect_to=JacobianWrtVariable.kQDot,
            frame_B=robot.GetFrameByName("iiwa_link_7"),    # frame on which point Bi is fixed
            p_BP=[0.0, 0.0, 0.0],                           # position vec from frame_B's origin to points Bi
            frame_A=robot.world_frame(),                    # frame that measures v_ABi
            frame_E=robot.world_frame())                    # frame that v_ABi is expressed on input 
                                                            # and frame that Jacobian J_V_ABi is expressed on output
        # print('jacobian: \n{}'.format(jacobian))
        pdV = np.zeros(num_joints)
        ang_jacobian = jacobian[0:3, 0:7]
        trans_jacobian = jacobian[3:6, 0:7]

        for i in range(7):
            rot_jacobian = skew(ang_jacobian[0:3, i]) @ R_H
            #pdV[i] = -2 * np.trace(R_G @ rot_jacobian) + 30 * (p_H - p_G) @ trans_jacobian[0:3, i]
            pdV[i] = -2 * np.trace(R_G @ rot_jacobian) + 2 * (p_H - p_G) @ trans_jacobian[0:3, i]

        # print('pdV: {}'.format(pdV))
        mu = 1.0/(V*22)
        #mu = max(0.1, 1.0/(V*27))
        # print('mu', mu)
        V_dot = -mu * np.linalg.norm(pdV)**2
        # print('V_dot: {}'.format(V_dot))

        theta_dot = -mu * pdV
        # print('theta_dot: \n{}'.format(theta_dot))

        q_prev = station.GetIiwaPosition(station_context)
        # print('q_prev: {}'.format(q_prev))
        q = q_prev + time_step * theta_dot
        # print('q: {}'.format(q))
        differential_ik.SetPositions(diagram.GetMutableSubsystemContext(
            differential_ik, simulator.get_mutable_context()), q)

        if num % 50 == 0:
            # Save image
            filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'all/images/{:05d}_{:03d}'.format(trial_num, image_num))
            rgb_image_visualizer.save_image(filename)
            filenames.append(filename)

            # Write to file
            data_to_write += ('{}, {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, '
                '{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f},\n'.format(
                trial_num, image_num, V, 
                q_prev[0], q_prev[1], q_prev[2], q_prev[3], q_prev[4], q_prev[5], q_prev[6], 
                pdV[0], pdV[1], pdV[2], pdV[3], pdV[4], pdV[5], pdV[6]))

            image_num += 1
        num += 1

        # why doesn't this velocity match theta_dot?
        # print('iiwa vel: {}'.format(station.GetIiwaVelocity(station_context)))
        # print(np.linalg.norm(station.GetIiwaVelocity(station_context) - theta_dot))
        simulator.set_target_realtime_rate(0)
        t += time_step
        simulator.AdvanceTo(t)

        if V < 0.05 and abs(X_HG[0, 3]) < 0.01 and abs(X_HG[1, 3]) < 0.01 and abs(X_HG[2, 3]) < 0.01:
            print('close enough to grasp, {}, {}, {}'.format(X_HG[0, 3], X_HG[1, 3], X_HG[2, 3]))
            close_enough_to_grasp = True
            break

    if close_enough_to_grasp:
        print('closing wsg {}'.format(trial_num))
    else:
        print('failed {}'.format(trial_num))
        # Erase previous data and write over images
        data_to_write = ''

        for filename in filenames:
            os.remove('{}.png'.format(filename))

    # print(data_to_write)
    write_q.put(data_to_write)
    return data_to_write

def listener(q):
    # Listens for messages on the q, writes to file.
    datafile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'all/data.csv')

    with open(datafile, 'w') as f:
        while 1:
            m = q.get()
            if m == 'kill':
                break
            f.write(str(m))
            f.flush()

def main():
    np.set_printoptions(precision=4, suppress=True)
    random.seed(3)
    # test was 2, train was 3

    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 2)

    # Put listener to work
    watcher = pool.apply_async(listener, (q,))

    # Fire off workers
    jobs = []
    for i in range(140):
        job = pool.apply_async(run_trial, (i, q))
        jobs.append(job)

    # Collect results from workers through the pool result queue
    for job in jobs:
        job.get()

    # Kill the listener
    q.put('kill')
    pool.close()
    pool.join()

    # while image_num < 100000:
    #     image_num = run_trial(image_num)
    #     gc.collect()

if __name__ == "__main__":
    main()
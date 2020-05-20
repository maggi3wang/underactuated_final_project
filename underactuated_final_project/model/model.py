import copy
import numpy as np
import os
import pandas as pd
import time
import cv2
from PIL import Image

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

import pycuda.driver as cuda
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import transforms

import matplotlib.pyplot as plt
from .net import Net

class RGBImageAndJointAngleDataset(Dataset):
    def __init__(self, image_offset, images_dir, csv_file, transform, robot, num_joints, differential_ik, delta_theta):
        self.image_offset = image_offset
        self.df = pd.read_csv(csv_file, header=0)
        self.images_dir = images_dir
        self.transform = transform
        self.robot = robot
        self.num_joints = num_joints
        self.differential_ik = differential_ik
        self.delta_theta = delta_theta

        # Preprocess df. Sort df by first number, and then sort by second number. Could be more efficient...
        self.df.sort_values(by=['a', 'b'], inplace = True)
        print(self.df)

    def __len__(self):
        #return len(self.df)
        return len(os.listdir(self.images_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_filename = '{:06d}.png'.format(idx + self.image_offset) # this might be wrong
        img_name = os.path.join(self.images_dir, img_filename)
        image = Image.open(img_name)
        rgb_image = image.convert('RGB')

        joint_angles = self.df.iloc[idx].values[3:10]
        joint_descriptors = [np.array(self.joint_angle_decoder(joint_angles))]
        for i in range(self.num_joints):
            unit = np.zeros(self.num_joints)
            unit[i] = 1
            joint_descriptors.append(np.array(self.joint_angle_decoder(joint_angles + unit[i]*self.delta_theta)))

        sample = {
            'image': self.transform(rgb_image),
            #'joint_angles': torch.from_numpy(joint_angles),                # only actual joint angles
            'joint_descriptor': torch.from_numpy(joint_descriptors[0]),
            'joint_descriptor_perturb_1': torch.from_numpy(joint_descriptors[1]), # atrocious, to clean
            'joint_descriptor_perturb_2': torch.from_numpy(joint_descriptors[2]),
            'joint_descriptor_perturb_3': torch.from_numpy(joint_descriptors[3]),
            'joint_descriptor_perturb_4': torch.from_numpy(joint_descriptors[4]),
            'joint_descriptor_perturb_5': torch.from_numpy(joint_descriptors[5]),
            'joint_descriptor_perturb_6': torch.from_numpy(joint_descriptors[6]),
            'joint_descriptor_perturb_7': torch.from_numpy(joint_descriptors[7]),
            'clf': self.df.iloc[idx].values[2],
            'clf_partials': torch.from_numpy(self.df.iloc[idx].values[10:17])
        }
        return sample

    def joint_angle_decoder(self, joint_angles):
        joint_descriptor = []
        for i in range(self.num_joints):
            link_name = "iiwa_link_{}".format(i+1)
            X_joint = self.differential_ik.ForwardKinematics(joint_angles, self.robot.GetFrameByName(link_name)).matrix()
            rot = X_joint[0:3, 0:3]
            trans = X_joint[0:3, 3]
            joint_descriptor.extend(np.append(rot.flatten(), trans))
        assert(len(joint_descriptor) == self.num_joints*12)
        return joint_descriptor


class Model():
    def __init__(self, train_dir, test_dir, models_dir, batch_size=32, lr=0.001):
        # Directories
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.models_dir = models_dir

        # Model params
        self.batch_size = batch_size
        self.learning_rate = lr

        self.cuda_avail = torch.cuda.is_available()

        # Drake stuff
        builder = DiagramBuilder()
        station = builder.AddSystem(ManipulationStation())

        # Initializes the chosen station type.
        station.SetupIiwaOnTableStation()
        station.Finalize()
        self.robot = station.get_controller_plant()
        self.num_joints = self.robot.num_positions()
        params = DifferentialInverseKinematicsParameters(self.num_joints, self.robot.num_velocities())
        time_step = 0.005
        params.set_timestep(time_step)
        self.differential_ik = DifferentialIK(self.robot, self.robot.GetFrameByName("iiwa_link_7"), params, time_step)

        # Create model and run setup

        self.model = Net()
        self.delta_theta = 0.05  # For siamese regression, finite difference
        self.set_up_data()
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001)

    def set_up_data(self):
        # RGB images (224x224x3)
        training_transform = transforms.Compose([
            transforms.Resize(224),
            #transforms.RandomResizedCrop(180),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        train_dataset = RGBImageAndJointAngleDataset(
            image_offset=0,
            images_dir=os.path.join(self.train_dir, 'images'),
            csv_file=os.path.join(self.train_dir, 'data.csv'),
            transform=training_transform, 
            robot=self.robot,
            num_joints=self.num_joints,
            differential_ik=self.differential_ik,
            delta_theta=self.delta_theta)

        test_dataset = RGBImageAndJointAngleDataset(
            #image_offset=5501,
            #image_offset=1401,
            image_offset=0,
            images_dir=os.path.join(self.test_dir, 'images'),
            csv_file=os.path.join(self.test_dir, 'data.csv'),
            transform=test_transform, 
            robot=self.robot,
            num_joints=self.num_joints,
            differential_ik=self.differential_ik,
            delta_theta=self.delta_theta)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

        self.train_set_size = len(train_dataset)
        self.test_set_size = len(test_dataset)

        if self.cuda_avail:
            cuda.init()
            torch.cuda.set_device(0)
            print(cuda.Device(torch.cuda.current_device()).name())
            self.model.cuda()

    def direct_regression_loss(self, alpha_weight, beta_weight, 
            clf, clf_partial, clf_target, clf_partial_target):
        num_samples = len(clf_target)
        regression_loss = alpha_weight/num_samples * torch.sum(torch.norm(clf - clf_target) ** 2)
        #print('clf', clf)
        #print('clf_target', clf_target)
        for i in range(self.num_joints):
            #print('partial {}, target {}'.format(clf_partial[:, i], clf_partial_target[:, i]))
            regression_loss += beta_weight/num_samples * torch.sum(torch.norm((clf_partial[:, i] - clf_partial_target[:, i])) ** 2)
        return regression_loss

    def siamese_regression_loss(self, clf, clf_partial):
        num_samples = int(len(clf)/(self.num_joints + 1))
        siamese_regression_loss = 0
        for i in range(self.num_joints):
            siamese_regression_loss += ((1/(7*num_samples) * 
                torch.sum(clf_partial[0:num_samples, i] - 
                (clf[(i+1)*num_samples:(i+2)*num_samples] - clf[0:num_samples])/self.delta_theta))**2)
        return siamese_regression_loss

    def train(self, num_epochs=80, siamese=True):
        since = time.time()

        train_loss_history = []
        test_loss_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        lowest_loss = np.inf
        epoch_of_lowest_loss = 0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and test phase
            #for phase in ['train', 'test']:
            for phase in ['train', 'test']:
                dataloader = None
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                    dataloader = self.train_loader
                else:
                    self.model.eval()   # Set model to evaluate mode
                    dataloader = self.test_loader

                running_loss = 0.0

                # Iterate over data.
                for sample in dataloader:
                    image = sample.get('image').type(torch.cuda.FloatTensor)
                    #joint_descriptors = sample.get('joint_descriptor').type(torch.cuda.FloatTensor)
                    clf = sample.get('clf').type(torch.cuda.FloatTensor)
                    clf_partials = sample.get('clf_partials').type(torch.cuda.FloatTensor)

                    joint_descriptors_lst = [sample.get('joint_descriptor').type(torch.cuda.FloatTensor)]
                    for i in range(1, 7+1):
                        joint_descriptors_lst.append(sample.get('joint_descriptor_perturb_{}'.format(i)).type(torch.cuda.FloatTensor))

                    joint_descriptors = torch.cat(joint_descriptors_lst, dim=0)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        images = [image] * (self.num_joints + 1)
                        images = torch.cat(images, dim=0)

                        outputs = self.model(images, joint_descriptors)
                        num_samples = int(len(outputs)/(self.num_joints + 1))
                        direct_regression_loss = self.direct_regression_loss(1, 1, 
                            outputs[0:num_samples, 0], outputs[0:num_samples, 1:], clf, clf_partials)
                        siamese_regression_loss = self.siamese_regression_loss(
                            outputs[:, 0], outputs[:, 1:])
                        loss = direct_regression_loss + 0.1 * siamese_regression_loss

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * clf.size(0)
                    
                epoch_loss = running_loss / len(dataloader.dataset)
                
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

                # deep copy the model
                if phase == 'test' and epoch_loss < lowest_loss:
                    lowest_loss = epoch_loss
                    epoch_of_lowest_loss = epoch
                    best_model_wts = copy.deepcopy(self.model.state_dict())

                if phase == 'test':
                    test_loss_history.append(epoch_loss)

                    #if epoch == num_epochs - 1:
                    #    best_model_wts = copy.deepcopy(self.model.state_dict())
                else:
                    train_loss_history.append(epoch_loss)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Lowest loss: {:4f}'.format(lowest_loss))

        # load best model weights
        #self.model.load_state_dict(best_model_wts)
        self.model.load_state_dict(best_model_wts)

        model_checkpoint_filename = "{:04d}.pth.tar".format(epoch_of_lowest_loss)
        filename = os.path.join(self.models_dir, model_checkpoint_filename)

        state = {'epoch': epoch_of_lowest_loss, 'state_dict': self.model.state_dict()}
        torch.save(state, filename)

        plt.plot(train_loss_history)
        plt.plot(test_loss_history)
        plt.show()

        return self.model, test_loss_history


from bdb import effective
from ctypes import sizeof
import imp
from tokenize import Double
from turtle import distance, pos
from geometry_msgs.msg import Pose, PoseArray, Quaternion
from pf_localisation import sensor_model
from . pf_base import PFLocaliserBase
import math
import rospy

from . util import rotateQuaternion, getHeading
from random import random
import random as r
import numpy as np

from time import time


class PFLocaliser(PFLocaliserBase):
    
    # Static variables declalation and initialize
    omegaFast = 0
    omegaSlow = 0
    
    def __init__(self):
        # ----- Call the superclass constructor
        super(PFLocaliser, self).__init__()
        
        # ----- Set motion model parameters
        self.ODOM_ROTATION_NOISE = 0.02 # Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 0.2 # Odometry model x axis (forward) noise
        self.ODOM_DRIFT_NOISE = 0.2 # Odometry model y axis (side-to-side) noise
 
        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict
        
    def initialise_particle_cloud(self, initialpose):
        """
        Set particle cloud to initialpose plus noise

        Called whenever an initialpose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        self.particlecloud can be initialised here. Initial pose of the robot
        is also set here.
        
        :Args:
            | initialpose: the initial pose estimate
        :Return:
            | (geometry_msgs.msg.PoseArray) poses of the particles
        """

        # Set fixed number of particles here
        particlesNum = 500
        # To return
        poseArray = PoseArray()
        # For faster calculation
        rotationNoise = 2 * math.pi * self.ODOM_ROTATION_NOISE

        # Spawn particles with noise
        for i in range(particlesNum):
            pose = Pose()
            pose.position.x = initialpose.pose.pose.position.x + r.gauss(0, 1) * self.ODOM_TRANSLATION_NOISE
            pose.position.y = initialpose.pose.pose.position.y + r.gauss(0, 1) * self.ODOM_DRIFT_NOISE
            pose.position.z = 0
            pose.orientation = rotateQuaternion(initialpose.pose.pose.orientation, r.gauss(0, 1) * rotationNoise)
            poseArray.poses.append(pose)

        return poseArray

    def update_particle_cloud(self, scan):
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

         """

        # rospy.loginfo("updating...")

        # Declare global variables
        global indexsOfRandomParticles

        # Initialize
        weights = []
        particlesNum = len(self.particlecloud.poses)
        particlesNumNeeded = particlesNum
        indexsOfRandomParticles = []
        MIN_NUM_PARCTICLE = 100
        MAX_NUM_PARCTICLE = 1000
        
        # Set long term and short term weights
        # alphaFast >> alphaSlow
        alphaFast = 0.2
        alphaSlow = 0.01

        # Set self.ParticleCloud to poseArray at the end
        poseArray = []
        uPoseArray = PoseArray()

        # For faster calculation
        rotationNoise = 2 * math.pi * self.ODOM_ROTATION_NOISE

        # Get list of weights
        for i in range(particlesNum):
            weights.append(self.sensor_model.get_weight(scan, self.particlecloud.poses[i]))

        # Normallise to make sum = 1
        sumOfWeights = sum(weights)
        for i in range(particlesNum):
            weights[i] = weights[i]/sumOfWeights

        # Calculate Omega
        omegaAverage = sumOfWeights/particlesNum
        self.omegaFast = self.omegaFast + alphaFast * (omegaAverage - self.omegaFast)
        self.omegaSlow = self.omegaSlow + alphaSlow * (omegaAverage - self.omegaSlow)
        
        particlesNumNeeded = particlesNum + math.floor(500/omegaAverage) - 100
        if particlesNumNeeded > MAX_NUM_PARCTICLE:
            particlesNumNeeded = MAX_NUM_PARCTICLE
        elif particlesNumNeeded < MIN_NUM_PARCTICLE:
            particlesNumNeeded = MIN_NUM_PARCTICLE
        
        # For faster calculation
        particlesNumNeededInv = 1/particlesNumNeeded

        # Cumulative sum
        cumSumWeights = np.cumsum(weights)

        # Systemetic resample initialize
        threshhold = []
        threshhold.append(r.uniform(0, particlesNumNeededInv))
        particleIndex = 0

        # Resample
        for j in range(particlesNumNeeded):
            while threshhold[j] > cumSumWeights[particleIndex]:
                particleIndex += 1
            poseArray.append(self.particlecloud.poses[particleIndex])
            threshhold.append(threshhold[j] + particlesNumNeededInv)

        # Add samples with noise or random poses
        for i in range(particlesNumNeeded):
            pose = Pose()
            if random() < max(0.0, 1.0 - self.omegaFast/self.omegaSlow):
                # Random sample is added here
                pose.position.x = 30 * random() - 15
                pose.position.y = 30 * random() - 15
                pose.orientation.z = random()
                pose.orientation.w = random()
                indexsOfRandomParticles.append(i)
            else:
                # Noise sample is added here
                tem = poseArray[i]
                pose.position.x = tem.position.x + r.gauss(0, 0.02)
                pose.position.y = tem.position.y + r.gauss(0, 0.02)
                pose.position.z = 0
                pose.orientation = rotateQuaternion(tem.orientation, r.gauss(0, 1) * rotationNoise)
            uPoseArray.poses.append(pose)

        # End here
        self.particlecloud = uPoseArray
        
    def estimate_pose(self):
        """
        This should calculate and return an updated robot pose estimate based
        on the particle cloud (self.particlecloud)
        
        Create new estimated pose, given particle cloud
        E.g. just average the location and orientation values of each of
        the particles and return this.
        
        Better approximations could be made by doing some simple clustering,
        e.g. taking the average location of half the particles after 
        throwing away any which are outliers

        :Return:
            | (geometry_msgs.msg.Pose) robot's estimated pose.
         """

        # rospy.loginfo("estimating...")
        
        # To return
        pose = Pose()

        # Initialize
        pose.position.x = 0
        pose.position.y = 0
        pose.position.z = 0 
        pose.orientation.x = 0 
        pose.orientation.y = 0 
        pose.orientation.z = 0
        pose.orientation.w = 0

        # For faster calculation
        particlesNum = len(self.particlecloud.poses)
        randomParticleNum = len(indexsOfRandomParticles)
        effectiveParticleNum = particlesNum - randomParticleNum

        if effectiveParticleNum == 0:
            return self.particlecloud.poses[1]

        if randomParticleNum == 0:
            effectiveParticlecloud = self.particlecloud.poses
        else:
            # Copy self.particlecloud.pose without randomParticles
            effectiveParticlecloud = self.particlecloud.poses.copy()

            # Remove random particles
            for i in indexsOfRandomParticles[::-1]:
                del effectiveParticlecloud[i]

        # Sum
        # Only 4 variables since in 2D
        for p in effectiveParticlecloud:
            pose.position.x += p.position.x
            pose.position.y += p.position.y
            pose.orientation.z += p.orientation.z
            pose.orientation.w += p.orientation.w

        # Average
        pose.position.x /= effectiveParticleNum
        pose.position.y /= effectiveParticleNum
        pose.orientation.z /= effectiveParticleNum
        pose.orientation.w /= effectiveParticleNum

        # Return
        return pose

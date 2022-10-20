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
from random import random, gauss
import random as r
import numpy as np
import time, statistics

from time import time


class PFLocaliser(PFLocaliserBase):
    
    # Static variables declalation and initialize
    omegaFast = 0
    omegaSlow = 0
    init_pose = PoseArray()

    
    def __init__(self):
        # ----- Call the superclass constructor
        super(PFLocaliser, self).__init__()
        
        # ----- Set motion model parameters
        self.ODOM_ROTATION_NOISE = 0.02 # Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 0.2 # Odometry model x axis (forward) noise
        self.ODOM_DRIFT_NOISE = 0.2 # Odometry model y axis (side-to-side) noise
 
        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict
        
        self.number_of_particles = 300 	  #Number of particles

        global init_pose
        
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

        self.init_pose = initialpose

        return poseArray

    def update_particle_cloud(self, scan):
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

         """
        
        
        #Method 1 - Fast slow-based algorithm

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
                pose.position.x = r.gauss(np.random.triangular(0, 0, 30), 0.1)
                pose.position.y = r.gauss(np.random.triangular(0, 0, 30), 0.1)
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
        
        
        '''
        #Method 2 - Weight-based algorithm

        particles_num = self.number_of_particles
        weights = []
        
        #Compute the likelihood weights of a set of particle clouds
        for pose in self.particlecloud.poses:
            weights.append(self.sensor_model.get_weight(scan, pose))

        normalised_weights = self.normalise(weights)
        
        cdf = [normalised_weights[0]]
        
        for i in range(1,len(normalised_weights)):
            cdf.append(cdf[i-1] + normalised_weights[i])
            
        if statistics.mean(weights) > 4:
            particles_num = 300
        else:
            particles_num = 400
            
        threshold = random()* math.pow(particles_num,-1)
        new_particle_cloud = PoseArray()
        
        i = 1
        for j in range(0, particles_num):
            while(threshold > cdf[i]):
                i += 1
                # make sigma prop to the inverse of the weight,

            if weights[i] < 4:
                #robot kidnapped/particle is far away from robot position
                sigma = 3
            else:
                sigma = 1.1/weights[i]
            
            new_particle_cloud.poses.append(self.initialise_pose(self.particlecloud.poses[i],sigma))

            threshold = threshold + math.pow(particles_num,-1)

        self.particlecloud = new_particle_cloud
        
    def normalise(self, w):
        w = w.copy()
        total = 0
        for i in range(len(w)):
            total += w[i]
        
        for j in range(len(w)):
            w[j] = w[j]/ total
        return w

        
    def initialise_pose(self, input_pose, sigma):
        pose = Pose()
        pose.position.x = gauss(input_pose.position.x, sigma)
        pose.position.y = gauss(input_pose.position.y, sigma)
        pose.orientation.z = gauss(input_pose.orientation.z, sigma)
        pose.orientation.w = gauss(input_pose.orientation.w, sigma)
        
        return pose

    
    '''

    '''
    #Method 3 - Continuous-based algorithm

    numofParticles = len(self.particlecloud.poses)
    uPose = []
    uPoseArray = PoseArray()

    #Get weight
    weights = []
    for i in range(len(self.particlecloud.poses)):
        weights.append(self.sensor_model.get_weight(scan, self.particlecloud.poses[i]))

    #Normalize weight
    nWeights = []
    for i in range(len(weights)):
        nWeights.append(weights[i] / (sum(weights)))

    #Resampling
    num = len(self.particlecloud.poses) * 2
    cumSum = np.cumsum(nWeights)
    i, n = 0, 0
    uni = uniform(0, 1/m)

    while n < num:
        if (uni <= cumSum[i]):
            uPose.append(self.particlecloud.poses[i])
            n += 1
            uni /= 1 / m
        else:
            i += 1

    # rospy.loginfo(len(uPose))

    while len(uPose) > len(self.particlecloud.poses):
        uPose.pop(r.randrange(len(uPose)))

    # rospy.loginfo(len(uPose))
    # rospy.loginfo(len(self.particlecloud.poses))

    for i in range(0, len(self.particlecloud.poses)):
        pose = Pose()
        pose.position.x = gauss(uPose[i].position.x, 0.1)
        pose.position.y = gauss(uPose[i].position.y, 0.1)
        pose.orientation = rotateQuaternion(uPose[i].orientation, r.gauss(0, 1) * 0.02)
        uPoseArray.poses.append(pose)

    for i in range(0, int(len(uPoseArray.poses) * 0.05)):
        rPose = Pose()
        rPose.position.x = gauss(np.random.triangular(0, 0, 30), 0.1)
        rPose.position.y = gauss(np.random.triangular(0, 0, 30), 0.1)
        rPose.orientation = rotateQuaternion(Quaternion(0, 0, 0, 0), math.radians(np.random.uniform(0, 360)))
        uPoseArray.poses.append(rPose)

    self.particlecloud = uPoseArray
    '''

        
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
        # Get number of particles
        particlesNum = len(self.particlecloud.poses)

        # Get mean and std of position of x and y
        mPosiX = 0
        sPosiX = 0
        posiX = []
        for i in range(0, particlesNum):
            posiX.append(self.particlecloud.poses[i].position.x)
        mPosiX = np.mean(posiX)
        sPosiX = np.std(posiX)

        mPosiY = 0
        sPosiY = 0
        posiY = []
        for i in range(0, particlesNum):
            posiY.append(self.particlecloud.poses[i].position.y)
        mPosiY = np.mean(posiY)
        sPosiY = np.std(posiY)

        #rospy.loginfo(mPosiX)
        #rospy.loginfo(sPosiX)
        
        # Calc std score
        sStdScoreX = 0
        sStdScoreY = 0
        for i in range(0, particlesNum):
            scoreX = (self.particlecloud.poses[i].position.x - mPosiX)/sPosiX
            scoreY = (self.particlecloud.poses[i].position.y - mPosiY)/sPosiY
            sStdScoreX = sStdScoreX + scoreX
            sStdScoreY = sStdScoreY + scoreY
        sStdScoreX = sStdScoreX / particlesNum
        sStdScoreY = sStdScoreY / particlesNum
        avgStdScore = (sStdScoreX + sStdScoreY)/2

        

        # Valid particles filter

        filterArray = PoseArray()
        for i in range(0, len(self.particlecloud.poses)):
            scoreX = (self.particlecloud.poses[i].position.x - mPosiX)/sPosiX
            scoreY = (self.particlecloud.poses[i].position.y - mPosiY)/sPosiY
            sAvgStdScore = (scoreX + scoreY)/2
            # rospy.loginfo("sAvg")
            # rospy.loginfo(sAvgStdScore)
            # rospy.loginfo("avg")
            # rospy.loginfo(avgStdScore)
            if sAvgStdScore <= avgStdScore:
                filterArray.poses.append(self.particlecloud.poses[i])
        
        # rospy.loginfo(filterArray)
        newParticlesNum = len(filterArray.poses)
        
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

        for i in range(0, newParticlesNum):
            pose.position.x += filterArray.poses[i].position.x
            pose.position.y += filterArray.poses[i].position.y
            pose.orientation.z += filterArray.poses[i].orientation.z
            pose.orientation.w += filterArray.poses[i].orientation.w

        # Average
        pose.position.x /= newParticlesNum
        pose.position.y /= newParticlesNum
        pose.orientation.z /= newParticlesNum
        pose.orientation.w /= newParticlesNum

        # Return
        return pose

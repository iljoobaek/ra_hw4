#!/usr/bin/env python

PACKAGE_NAME = 'hw4'

# Standard Python Imports
import os
import copy
import time
import math
import numpy as np
np.random.seed(0)
import scipy

import collections
import Queue

import signal

# OpenRAVE
import openravepy
#openravepy.RaveInitialize(True, openravepy.DebugLevel.Debug)


curr_path = os.getcwd()
relative_ordata = '/models'
ordata_path_thispack = curr_path + relative_ordata


#this sets up the OPENRAVE_DATA environment variable to include the files we're using
openrave_data_path = os.getenv('OPENRAVE_DATA', '')
openrave_data_paths = openrave_data_path.split(':')
if ordata_path_thispack not in openrave_data_paths:
  if openrave_data_path == '':
      os.environ['OPENRAVE_DATA'] = ordata_path_thispack
  else:
      datastr = str('%s:%s'%(ordata_path_thispack, openrave_data_path))
      os.environ['OPENRAVE_DATA'] = datastr

#set database file to be in this folder only
relative_ordatabase = '/database'
ordatabase_path_thispack = curr_path + relative_ordatabase
os.environ['OPENRAVE_DATABASE'] = ordatabase_path_thispack

#get rid of warnings
openravepy.RaveInitialize(True, openravepy.DebugLevel.Fatal)
openravepy.misc.InitOpenRAVELogging()


#constant for max distance to move any joint in a discrete step
#this constant is for arm movement!
MAX_MOVE_AMOUNT = 0.1

WHEEL_RADIUS = 0.20
ROBOT_LENGTH = 0.25
TIMESTEP_AMOUNT = 1 #0.02

#constant for max distance to move any joint in a discrete step
TRANS_PER_DIR = 0.1


class RoboHandler:
  def __init__(self):
    self.E=1
    self.openrave_init()
    self.problem_init()

    self.run_problem_navsearch()
    #self.run_problem_nav_and_grasp()


  #######################################################
  # the usual initialization for openrave
  #######################################################
  def openrave_init(self):
    self.env = openravepy.Environment()
    self.env.SetViewer('qtcoin')
    self.env.GetViewer().SetName('HW4 Viewer')
    self.env.Load('models/%s_navplan.env.xml' %PACKAGE_NAME)
    # time.sleep(3) # wait for viewer to initialize. May be helpful to uncomment
    self.robot = self.env.GetRobots()[0]

    #set right wam as active manipulator
    with self.env:
      self.robot.SetActiveManipulator('right_wam');
      self.manip = self.robot.GetActiveManipulator()

      #set active indices to be right arm only
      self.robot.SetActiveDOFs(self.manip.GetArmIndices() )
      self.end_effector = self.manip.GetEndEffector()

  #######################################################
  # problem specific initialization
  #######################################################
  def problem_init(self):
    self.target_kinbody = self.env.GetKinBody("target")

    # create a grasping module
    self.gmodel = openravepy.databases.grasping.GraspingModel(self.robot, self.target_kinbody)
    
    # load grasps
    if not self.gmodel.load():
      self.gmodel.autogenerate()

    self.grasps = self.gmodel.grasps
    self.graspindices = self.gmodel.graspindices

    # load ikmodel
    self.ikmodel = openravepy.databases.inversekinematics.InverseKinematicsModel(self.robot,iktype=openravepy.IkParameterization.Type.Transform6D)
    if not self.ikmodel.load():
      self.ikmodel.autogenerate()

    # create taskmanip
    self.taskmanip = openravepy.interfaces.TaskManipulation(self.robot)
  
    # move arms out of way
    self.robot.SetDOFValues(np.array([4,2,0,-1,0,0,0]),self.robot.GetManipulator('left_wam').GetArmIndices() )
    self.robot.SetDOFValues(np.array([4.0,-1.5,0,1.5,0,0,0]),self.robot.GetManipulator('right_wam').GetArmIndices() )

    #save the current robot transform
    self.start_trans = self.robot.GetTransform()
    self.start_DOFS = self.robot.GetActiveDOFValues()


    #initialize the transition transformations for base movmement
    self.init_transition_transforms()


  #######################################################
  # navsearch to transform
  #######################################################
  def run_problem_navsearch(self):
    th = -np.pi/2
    x_trans = -0.5
    y_trans = 0.5
    goal_trans = [np.copy(self.start_trans)]
    rot_to_goal = np.array([[np.cos(th), -np.sin(th), 0],
                              [np.sin(th), np.cos(th), 0],
                              [0, 0, 1]])
    goal_trans[0][0:3,0:3] = np.dot(rot_to_goal, self.start_trans[0:3,0:3])
    goal_trans[0][0,3] += x_trans
    goal_trans[0][1,3] += y_trans


    th = -np.pi/2
    x_trans = 0.5
    y_trans = 3.0
    goal_trans.append(np.copy(self.start_trans))
    rot_to_goal = np.array([[np.cos(th), -np.sin(th), 0],
                              [np.sin(th), np.cos(th), 0],
                              [0, 0, 1]])
    goal_trans[1][0:3,0:3] = np.dot(rot_to_goal, self.start_trans[0:3,0:3])
    goal_trans[1][0,3] += x_trans
    goal_trans[1][1,3] += y_trans
    
    with self.env:
      self.robot.SetTransform(self.start_trans)
      
    # test to see where the goals are
    print self.start_trans
    #self.init_transition_transforms()
    #self.run_basetranforms(goal_trans)

    # get the trajectory!
    base_transforms = self.astar_to_transform(goal_trans)

    with self.env:
      self.robot.SetTransform(self.start_trans)

    self.run_basetranforms(base_transforms)


  #######################################################
  # grasp an object by first driving to a location
  # then performing grasp
  #######################################################
  def run_problem_nav_and_grasp(self):
    self.robot.GetController().Reset()
    # move hand to preshape of grasp
    # --- important --
    # I noted they were all the same, otherwise you would need to do this separately for each grasp!
    with self.env:
      self.robot.SetDOFValues(self.grasps[0][self.graspindices['igrasppreshape']], self.manip.GetGripperIndices()) # move to preshape

    with self.env:
      self.robot.SetTransform(self.start_trans)
      self.robot.SetActiveDOFValues(self.start_DOFS)
    
    base_transforms,arm_traj = self.nav_and_grasp()

    with self.env:
      self.robot.SetTransform(self.start_trans)
      self.robot.SetActiveDOFValues(self.start_DOFS)

    self.run_basetranforms(base_transforms)
    self.robot.GetController().SetPath(arm_traj)
    self.robot.WaitForController(0)
    self.taskmanip.CloseFingers()

  #######################################################
  # finds the arm configurations (in cspace) that correspond
  # to valid grasps
  # num_goal: number of grasps to consider
  # num_dofs_per_goal: number of IK solutions per grasp
  #######################################################
  def get_goal_dofs(self, num_goals=1, num_dofs_per_goal=1):
    validgrasps,validindices = self.gmodel.computeValidGrasps(returnnum=num_goals) 

    curr_IK = self.robot.GetActiveDOFValues()

    goal_dofs = np.array([])
    for grasp, graspindices in zip(validgrasps, validindices):
      Tgoal = self.gmodel.getGlobalGraspTransform(grasp, collisionfree=True)
      sols = self.manip.FindIKSolutions(Tgoal, openravepy.IkFilterOptions.CheckEnvCollisions)

      # magic that makes sols only the unique elements - sometimes there are multiple IKs
      sols = np.unique(sols.view([('',sols.dtype)]*sols.shape[1])).view(sols.dtype).reshape(-1,sols.shape[1]) 
      sols_scores = []
      for sol in sols:
        sols_scores.append( (sol, np.linalg.norm(sol-curr_IK)) )

      # sort by closest to current IK
      sols_scores.sort(key=lambda tup:tup[1])
      sols = np.array([x[0] for x in sols_scores])
      
      # sort randomly
      #sols = np.random.permutation(sols)

      #take up to num_dofs_per_goal
      last_ind = min(num_dofs_per_goal, sols.shape[0])
      goal_dofs = np.append(goal_dofs,sols[0:last_ind])

    goal_dofs = goal_dofs.reshape(goal_dofs.size/7, 7)

    return goal_dofs


  #TODO
  #######################################################
  # Returns a base trajectory and subsequent arm trajectory
  # which will grasp the target object from the current configuration
  #######################################################
  def nav_and_grasp(self):
    #example of calling a function with timeout:
    #base_transforms = run_func_with_timeout(self.astar_to_transform, args=[base_transform_goals], timeout=40)
    return None

  #TODO
  #######################################################
  # Samples a configuration suitable for grasp
  #######################################################
  def sample_for_grasp(self):
    return None


  #TODO
  #Or just copy your old one here
  #######################################################
  # Bi-Directional RRT
  # find a path from the current configuration to ANY goal in goals
  # goals: list of possible goal configurations
  # RETURN: a trajectory to the goal
  #######################################################
  def birrt_to_goal(self, goals):
    return None

  #TODO
  #######################################################
  # BASE MOVEMENT with A* SEARCH
  # find a path from the current configuration to transform
  # RETURN: an array of ALL intermediate transforms.
  # Thus, you should use self.full_transforms when returning!
  #######################################################
  def astar_to_transform(self, goal_transforms):
    # test for a star using HW2
    self.run_difficult_problem()
    return None
  
  def run_difficult_problem(self):
    self.robot.GetController().Reset()

    with self.env:
      self.robot.SetDOFValues(self.grasps[0][self.graspindices['igrasppreshape']], self.manip.GetGripperIndices()) # move to preshape
    with self.env:
      self.robot.SetActiveDOFValues([5.459, -0.981,  -1.113,  1.473 , -1.124, -1.332,  1.856])    

    self.init_transition_arrays()
    #goals = self.get_goal_dofs(7,1)
    goals = np.array([[ 0.93422058, -1.10221021, -0.2       ,  2.27275587, -0.22977831, -1.09393251, -2.23921746],
       [ 1.38238176, -1.05017481,  0.        ,  1.26568204,  0.15001448,  1.32813949, -0.06022621],
       [ 1.16466262, -1.02175153, -0.3       ,  1.26568204, -2.62343746, -1.43813577, -0.37988181],
       [ 3.45957137, -0.48619817,  0.        ,  2.0702298 , -1.12033301, -1.33241556,  1.85646563],
       [ 1.65311863, -1.17157253,  0.4       ,  2.18692683, -2.38248898,  0.73272595, -0.23680544],
       [ 1.59512823, -1.07309638,  0.5       ,  2.26315055,  0.57257592, -1.15576369, -0.30723627],
       [ 1.67038884, -1.16082512,  0.4       ,  2.05339849, -2.0205527 ,  0.54970211, -0.4386743 ]])

    traj = self.search_to_goal_astar(goals)

    with self.env:
      self.robot.SetActiveDOFValues([5.459, -0.981,  -1.113,  1.473 , -1.124, -1.332,  1.856])
    
    self.robot.GetController().SetPath(traj)
    self.robot.WaitForController(0)
    self.taskmanip.CloseFingers()
  
  #######################################################
  # A* SEARCH
  # find a path from the current configuration to ANY goal in goals
  # goals: list of possible goal configurations
  # RETURN: a trajectory to the goal
  #######################################################
  def search_to_goal_astar(self,goals): 
    print('Running Astar')
    initial = self.robot.GetActiveDOFValues()
    goals=np.ndarray.round(goals,1)
    #goals = goals.round(1)
    initial=np.ndarray.round(initial,1)
    #initial = initial.round(1)

    goal = set()
    for element in goals:
        goal.add(self.convert_for_dict_withround(element))
    vco = set()


    f_score=dict()
    #g_score=dict()
 
 
    Q = Queue.PriorityQueue() #openset
    path = dict()
    ini_c=self.config_to_priorityqueue_tuple(0,initial,goals)
    print ini_c
    Q.put(ini_c)
    f_score[ini_c[1]]=ini_c[0]


    while not Q.empty():
      parent=Q.get()
      current=self.convert_from_dictkey_withround(parent[1])

      g_score=parent[0]-self.E*self.min_manhattan_dist_to_goals(current,goals)
      if parent[1] in goal:
        print('success')
        point=self.creat_traj_points(tuple(current),path)
        return self.points_to_traj(point)
      
      if parent[1] not in vco:
        vco.add(parent[1])
        action=self.transition_config(current)
        for neighbor in action:
          with self.env:
            self.robot.SetActiveDOFValues(neighbor)
            if (self.env.CheckCollision(self.robot) or self.robot.CheckSelfCollision() or self.check_dof_limits(neighbor)):
              continue
   
            if self.convert_for_dict_withround(neighbor) not in vco:
              temp=self.config_to_priorityqueue_tuple(g_score+TRANS_PER_DIR,neighbor,goals)
              Q.put(temp)
              if self.convert_for_dict_withround(neighbor) not in f_score.keys() or f_score[self.convert_for_dict_withround(neighbor)] > temp[0]:
                f_score[self.convert_for_dict_withround(neighbor)] = temp[0]
                path[tuple(neighbor)] = tuple(current)      
    print('failure')      
    return
  
  
  #check limits
  def check_dof_limits(self, config):
    upper,lower = self.robot.GetDOFLimits(self.manip.GetArmIndices())
    flag=True
    for i in range(len(upper)):
      if config[i]>upper[i]:
        flag=False
      if config[i]<lower[i]:
        flag=False
    return flag
  
  
  def creat_traj_points(self, child,dictin):
    point=[]
    i=0
    point.append(child)
    while dictin.get(point[i],'none')  != 'none':
      point.append(dictin[point[i]])
      i=i+1
    point=point[::-1]
    return point

  #######################################################
  # Check if the config is close enough to goal
  # Returns true if any goal in goals is within
  # BOTH distance_thresh and theta_thresh
  #######################################################
  def is_at_goal_basesearch(self, config, goals, dist_thresh = 0.02, theta_thresh = np.pi/12):
    for goal in goals:
      if (np.linalg.norm(config[0:2]-goal[0:2]) <= dist_thresh and np.abs(config[2] - goal[2]) <= theta_thresh):
        return True
    return False



  #TODO
  #######################################################
  # Initialize the movement transforms
  # These are equivalent to applying a fixed control for some
  # amount of time
  #######################################################
  def init_transition_transforms(self):
    self.transition_transforms = []
    self.full_transforms = []
    
    # get initial x, y, thera
    # x: param[0], y: param[1], thera: param[2]
    cur_param = self.transform_to_params(self.start_trans)

    # expand tree using w1 and w2 combination
    # combination 1 : w1 = 1, w2 = 1
    w_1 = 1
    w_2 = 1    
    new_param = self.get_xythera_from_angularvel(w_1, w_2, TIMESTEP_AMOUNT, cur_param)
    self.transition_transforms.append(self.params_to_transform(new_param)) # convert to tranform and add it to list
      
    # combination 2 : w1 = -1, w2 = 1
    w_1 = -1
    w_2 = 1    
    new_param = self.get_xythera_from_angularvel(w_1, w_2, TIMESTEP_AMOUNT, cur_param)
    self.transition_transforms.append(self.params_to_transform(new_param)) # convert to transform and add it to list    

    # combination 3 : w1 = 1, w2 = -1
    w_1 = 1
    w_2 = -1    
    new_param = self.get_xythera_from_angularvel(w_1, w_2, TIMESTEP_AMOUNT, cur_param)
    self.transition_transforms.append(self.params_to_transform(new_param)) # convert to transform and add it to list
    
    for i in range(3):
      print self.transition_transforms[i]
      #time.sleep(5)
      #self.robot.SetTransform(np.dot(self.start_trans, self.transition_transforms[i]))

  
  # get x, y, thera from differential angular velocities w1 and w2
  # input)
  #    w1, w2: angular velocities
  #    x,y,thera in previous time step: numpy array, x: param[0], y: param[1], thera: param[2]
  # output)
  #    x,y,thera in current time step: numpy array, x: param[0], y: param[1], thera: param[2]
  def get_xythera_from_angularvel(self,w_1,w_2,time_step, prev_param):
    thera_t = (WHEEL_RADIUS/(2*ROBOT_LENGTH))*(w_1-w_2)*time_step + prev_param[2]
    thera_t = prev_param[2]-thera_t
    x_t = (WHEEL_RADIUS/2)*(-w_1*np.sin(thera_t)-w_2*np.sin(thera_t))*time_step + prev_param[0]
    y_t = (WHEEL_RADIUS/2)*(w_1*np.cos(thera_t)+w_2*np.cos(thera_t))*time_step + prev_param[1]
    return np.array([prev_param[0]-x_t, prev_param[1]-y_t, thera_t])
    
  
  #TODO
  #######################################################
  # Applies the specified controls to the initial transform
  # returns a list of all intermediate transforms
  #######################################################
  def controls_to_transforms(self,trans,controls,timestep_amount):
    return None
  
  #######################################################
  # Initialize the movements you can apply in any direction
  # Don't forget to use TRANS_PER_DIR - the max distance you
  # can move any joint in a step (defined above)
  #######################################################
  def init_transition_arrays(self):
    self.transition_arrays = []
    self.transition_arrays=np.eye(7)*TRANS_PER_DIR
    self.transition_arrays=np.append(self.transition_arrays,np.eye(7)*(-TRANS_PER_DIR),0)
    return

  #TODO
  #######################################################
  # Take the current configuration and apply each of your
  # transition arrays to it
  #######################################################
  def transition_config(self, config):

    new_configs = np.zeros((14,7))
    for i in range (14):
      new_configs[i,:]=config+ self.transition_arrays[i]
    return new_configs

  #TODO
  #######################################################
  # Implement a heuristic for base navigation
  #######################################################
  #def config_to_priorityqueue_tuple(self, dist, config, goals):
  #  # make sure to replace the 0 with your priority queue value!
  #  return (0.0, config.tolist())
  
  def config_to_priorityqueue_tuple(self, dist, config, goals):
    priority= dist+(self.E*self.min_manhattan_dist_to_goals(config,goals))
    return (priority,self.convert_for_dict_withround(config))


  #######################################################
  # ASSUMES TRANSFORM ONLY ROTATED ABOUT Z
  # Takes rotation or transform, and returns the angle of rotation
  #######################################################
  def rot_matrix_to_angle(self,transform):
    return np.arctan2(transform[1,0], transform[0,0])

  #######################################################
  # ASSUMES TRANSFORM ONLY ROTATED ABOUT Z
  # Takes in an x,y,theta, and returns a transform
  #######################################################
  def xyt_to_transform(self,x,y,theta):
    t = np.array([[np.cos(theta), -np.sin(theta), 0, x],
                 [np.sin(theta), np.cos(theta), 0, y],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]] )
    return t

  #######################################################
  # Convert between our params (array with x,y,theta) and 4x4 transform
  #######################################################
  def params_to_transform(self,params):
    return self.xyt_to_transform(params[0], params[1], params[2])

  def transform_to_params(self,transform):
    return np.array([transform[0,3], transform[1,3], self.rot_matrix_to_angle(transform)])
  #######################################################
  # minimum distance from config to any goal in goals
  # distance metric: euclidean
  # returns the distance AND closest goal
  #######################################################
  def min_euclid_dist_to_goals(self, config, goals):
    dists = np.sum((config-goals)**2,axis=1)**(1./2)
    min_ind = np.argmin(dists)
    return dists[min_ind], goals[min_ind]


  #######################################################
  # Convert to and from numpy array to a hashable function
  #######################################################
  def convert_for_dict(self, item):
    return tuple(item)

  def convert_from_dictkey(self, item):
    return np.array(item)


  #######################################################
  # Convert to and from numpy array to a hashable function
  # includes rounding
  #######################################################
  def convert_for_dict_withround(self, item):
    #return tuple(np.int_(item*100))
    return tuple(np.round((item*100),1))

  def convert_from_dictkey_withround(self, item):
    #return np.array(item)/100.
    return np.array(item)/100


  def points_to_traj(self, points):
    traj = openravepy.RaveCreateTrajectory(self.env,'')
    traj.Init(self.robot.GetActiveConfigurationSpecification())
    for idx,point in enumerate(points):
      traj.Insert(idx,point)
    openravepy.planningutils.RetimeActiveDOFTrajectory(traj,self.robot,hastimestamps=False,maxvelmult=1,maxaccelmult=1,plannername='ParabolicTrajectoryRetimer')
    return traj


  def run_basetranforms(self, transforms):
    for trans in transforms:
      with self.env:
        self.robot.SetTransform(trans)
      time.sleep(0.01)


  #######################################################
  # minimum distance from config (singular) to any other config in o_configs
  # distance metric: euclidean
  # returns the distance AND index
  #######################################################
  def min_euclid_dist_one_to_many(self, config, o_configs):
    dists = np.sum((config-o_configs)**2,axis=1)**(1./2)
    min_ind = np.argmin(dists)
    return dists[min_ind], min_ind


  #######################################################
  # minimum distance from configs (plural) to any other config in o_configs
  # distance metric: euclidean
  # returns the distance AND indices into config and o_configs
  #######################################################
  def min_euclid_dist_many_to_many(self, configs, o_configs):
    dists = []
    inds = []
    for o_config in o_configs:
      [dist, ind] = self.min_euclid_dist_one_to_many(o_config, configs)
      dists.append(dist)
      inds.append(ind)
    min_ind_in_inds = np.argmin(dists)
    return dists[min_ind_in_inds], inds[min_ind_in_inds], min_ind_in_inds


  #######################################################
  # minimum distance from config to any goal in goals
  # distance metric: manhattan
  # returns the distance AND closest goal
  #######################################################
  def min_manhattan_dist_to_goals(self, config, goals):
    dist_g = []
    for g in goals:
      dist_g .append(sum(tuple([abs(round(item1 - item2,1)) for item1, item2 in zip(config,tuple(g))])))
    return min(dist_g)
    
    
  def dist_to_goals(self, config, goals):
    dist = math.sqrt(sum(tuple([pow(abs(item1 - item2),2) for item1, item2 in zip(config,goals)]))) 
    return dist
 

  #######################################################
  # close the fingers when you get to the grasp position
  #######################################################
  def close_fingers(self):
    self.taskmanip.CloseFingers()
    self.robot.WaitForController(0) #ensures the robot isn't moving anymore
    #self.robot.Grab(target) #attaches object to robot, so moving the robot will move the object now



def handler(signum, frame):
  raise Exception("end of time")

def run_func_with_timeout(func, args = (), timeout=1000000000):
  signal.signal(signal.SIGALRM, handler)
  signal.alarm(timeout)
  result = None
  try:
    result = func(args)
  except Exception, exc: 
    print exc
    pass
  finally:
    signal.alarm(0)

  return result

if __name__ == '__main__':
  robo = RoboHandler()
  #time.sleep(10000) #to keep the openrave window open
  

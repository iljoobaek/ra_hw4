#!/usr/bin/env python

PACKAGE_NAME = 'hw2'

# Standard Python Imports
import os
import copy
import time
import math
import numpy as np
np.random.seed(0)
import scipy

import collections
from collections import deque
import Queue
import operator



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
TRANS_PER_DIR = 0.1


class RoboHandler:
  def __init__(self):
    self.E=1
    self.openrave_init()
    self.problem_init()
    #self.run_simple_problem()
    #self.close_fingers()
    self.run_difficult_problem()
    self.close_fingers()
    print('weighted astar')
    self.E=1.5
    self.run_difficult_problem()
    self.close_fingers()
    self.run_simple_problem_second()
    self.close_fingers()


  #######################################################
  # the usual initialization for openrave
  #######################################################
  def openrave_init(self):
    self.env = openravepy.Environment()
    self.env.SetViewer('qtcoin')
    self.env.GetViewer().SetName('HW2 Viewer')
    self.env.Load('models/%s.env.xml' %PACKAGE_NAME)
    # time.sleep(3) # wait for viewer to initialize. May be helpful to uncomment
    self.robot = self.env.GetRobots()[0]

    #set right wam as active manipulator
    with self.env:
      self.robot.SetActiveManipulator('right_wam');
      self.manip = self.robot.GetActiveManipulator()

      #set active indices to be right arm only
      self.robot.SetActiveDOFs(self.manip.GetArmIndices() )
      self.end_effector = self.manip.GetEndEffector()
      
      #recorder = RaveCreateModule(self.env,'viewerrecorder')
      #self.env.AddModule(recorder,'')
      #codecs = recorder.SendCommand('GetCodecs') # linux only
      #filename = 'openrave.mpg'
      #codec = 13 # mpeg4
      #recorder.SendCommand('Start 640 480 30 codec %d timing realtime filename %s\nviewer %s'%(codec,filename,env.GetViewer().GetName()))


  #######################################################
  # problem specific initialization
  #######################################################
  def problem_init(self):
    self.target_kinbody = self.env.GetKinBody("target")
    self.table_kinbody = self.env.GetKinBody("table")

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
  
    # move left arm out of way
    self.robot.SetDOFValues(np.array([4,2,0,-1,0,0,0]),self.robot.GetManipulator('left_wam').GetArmIndices() )


  #######################################################
  # Simpler search problem - uses breadth first search algorithm
  #######################################################
  def run_simple_problem(self):
    self.robot.GetController().Reset()

    # move hand to preshape of grasp
    # --- important --
    # I noted they were all the same, otherwise you would need to do this separately for each grasp!
    with self.env:
      self.robot.SetDOFValues(self.grasps[0][self.graspindices['igrasppreshape']], self.manip.GetGripperIndices()) # move to preshape
    
    
    self.init_transition_arrays()
    goal = np.array([ 0.9, -1.1, -0.2,  2.3, -0.2, -1.1, -2.2])
    
    with self.env:
      self.robot.SetActiveDOFValues([ 1.23, -1.10, -0.3,  2.37, -0.23, -1.29, -2.23])

    # get the trajectory!
    traj = self.search_to_goal_breadthfirst(goal)

    with self.env:
      self.robot.SetActiveDOFValues([ 1.23, -1.10, -0.3,  2.37, -0.23, -1.29, -2.23])

    
    self.robot.GetController().SetPath(traj)
    self.robot.WaitForController(0)
    self.taskmanip.CloseFingers()


#######################

  def run_simple_problem_second(self):
    self.robot.GetController().Reset()
    
   
    with self.env:
      self.robot.SetDOFValues(self.grasps[0][self.graspindices['igrasppreshape']], self.manip.GetGripperIndices()) # move to preshape
    
    
    self.init_transition_arrays()
    goal =np.array( [ 0.9, -1.1, -0.2,  2.3, -0.2, -1.1, -2.2])
    with self.env:
      self.robot.SetActiveDOFValues([ 1.23, -1.10, -0.3,  2.37, -0.23, -1.29, -2.23])

    # get the trajectory!
    traj = self.search_to_goal_depthfirst(goal)

    with self.env:
      self.robot.SetActiveDOFValues([ 1.23, -1.10, -0.3,  2.37, -0.23, -1.29, -2.23])

    
    self.robot.GetController().SetPath(traj)
    self.robot.WaitForController(0)
    self.taskmanip.CloseFingers()



  #######################################################
  # Harder search problem - uses A* algorithm
  #######################################################
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

#################################


  def run_difficult_problem_second(self):
    self.robot.GetController().Reset()

  
    with self.env:
      self.robot.SetDOFValues(self.grasps[0][self.graspindices['igrasppreshape']], self.manip.GetGripperIndices()) # move to preshape
    

    self.init_transition_arrays()
    goals = np.array([[ 0.93422058, -1.10221021, -0.2       ,  2.27275587, -0.22977831, -1.09393251, -2.23921746],
       [ 1.38238176, -1.05017481,  0.        ,  1.26568204,  0.15001448,  1.32813949, -0.06022621],
       [ 1.16466262, -1.02175153, -0.3       ,  1.26568204, -2.62343746, -1.43813577, -0.37988181],
       [ 3.45957137, -0.48619817,  0.        ,  2.0702298 , -1.12033301, -1.33241556,  1.85646563],
       [ 1.65311863, -1.17157253,  0.4       ,  2.18692683, -2.38248898,  0.73272595, -0.23680544],
       [ 1.59512823, -1.07309638,  0.5       ,  2.26315055,  0.57257592, -1.15576369, -0.30723627],
       [ 1.67038884, -1.16082512,  0.4       ,  2.05339849, -2.0205527 ,  0.54970211, -0.4386743 ]])
 
    with self.env:
      self.robot.SetActiveDOFValues([5.459, -0.981,  -1.113,  1.473 , -1.124, -1.332,  1.856])

    # get the trajectory!
    traj = self.search_to_goal_astar_weighted(goals)

    with self.env:
      self.robot.SetActiveDOFValues([5.459, -0.981,  -1.113,  1.473 , -1.124, -1.332,  1.856])
    
    self.robot.GetController().SetPath(traj)
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


  ### TODO ###  
  #######################################################
  # DEPTH FIRST SEARCH
  # find a path from the current configuration to ANY goal in goals
  # goals: list of possible goal configurations
  # RETURN: a trajectory to the goal
  #######################################################
  def search_to_goal_depthfirst(self, goals):
	  print ('Running depth First Search')
	  goal=np.ndarray.round(goals,1)
	  initial = self.robot.GetActiveDOFValues()
	  initial=np.ndarray.round(initial,1)

	  Q = deque()
	  Q.append(self.convert_for_dict(initial))
	  mydict = dict()

	  lst = set()
	  lst.add(self.convert_for_dict(initial))
	  
	  # While the Q isn't empty
	  while(len(Q)>0): 
	      # pop the top item off of the Q and call it Parent 
	      parent = Q.pop()
              man_dist = self.dist_to_goals(self.convert_from_dictkey(parent), goal)
	      
	      if parent == self.convert_for_dict(goal) or man_dist<TRANS_PER_DIR:
		  print('sucess')
		  point =self.creat_traj_points(tuple(self.convert_from_dictkey(parent)),mydict)
		  traj=self.points_to_traj(point)
		  return traj

	      if (self.env.CheckCollision(self.robot) or self.robot.CheckSelfCollision() or self.check_dof_limits(self.convert_from_dictkey(parent))):
	        continue
#######################################
	      U=self.transition_config(self.convert_from_dictkey(parent))
	      for u in U: 
		      # Check if the child has already been added to the Q
		      if  self.convert_for_dict(u) not in lst:
			     
			      # Add the child to the dictionary
			      mydict[tuple(u)] = tuple(self.convert_from_dictkey(parent))
			      Q.append(self.convert_for_dict(u))
			      lst.add(self.convert_for_dict(u))
			      
			      # Check if the child is the goal      
			    
	  print('failure')		      
	  return 




  ### TODO ###  
  #######################################################
  # BREADTH FIRST SEARCH
  # find a path from the current configuration to ANY goal in goals
  # goals: list of possible goal configurations
  # RETURN: a trajectory to the goal
  #######################################################
  def search_to_goal_breadthfirst(self, goals):
	  print ('Running Breadth First Search')
	  goal=np.ndarray.round(goals,1)
	  initial = self.robot.GetActiveDOFValues()
	  initial=np.ndarray.round(initial,1)

 
	  Q = deque()
	  Q.append(self.convert_for_dict(initial))
	  mydict = dict()

	  lst = set()
	  lst.add(self.convert_for_dict(initial))
	  
	  # While the Q isn't empty
	  while(len(Q)>0): 
	      # pop the top item off of the Q and call it Parent 
	      parent = Q.popleft()
              man_dist = self.dist_to_goals(self.convert_from_dictkey(parent), goal)
	      
	      if parent == self.convert_for_dict(goal) or man_dist<TRANS_PER_DIR:
		  print('sucess')
		  point =self.creat_traj_points(tuple(self.convert_from_dictkey(parent)),mydict)
		  traj=self.points_to_traj(point)
		  return traj

	      if (self.env.CheckCollision(self.robot) or self.robot.CheckSelfCollision() or self.check_dof_limits(self.convert_from_dictkey(parent))):
	        continue
#######################################
	      U=self.transition_config(self.convert_from_dictkey(parent))

	      for u in U: 
		      # Check if the child has already been added to the Q
		      if  self.convert_for_dict(u) not in lst:
			     
			      # Add the child to the dictionary
			      mydict[tuple(u)] = tuple(self.convert_from_dictkey(parent))
			      Q.append(self.convert_for_dict(u))
			      lst.add(self.convert_for_dict(u))
			      
			      # Check if the child is the goal      
			    
	  print('failure')		      
	  return 



  ### TODO ###  
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
        goal.add(self.convert_for_dict(element))
    vco = set()


    f_score=dict()
    #g_score=dict()
 
 
    Q = Queue.PriorityQueue() #openset
    path = dict()
    ini_c=self.config_to_priorityqueue_tuple(0,initial,goals)
    Q.put(ini_c)
    f_score[ini_c[1]]=ini_c[0]


    while not Q.empty():
      parent=Q.get()
      current=self.convert_from_dictkey(parent[1])

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
   
	    if self.convert_for_dict(neighbor) not in vco:
	      temp=self.config_to_priorityqueue_tuple(g_score+TRANS_PER_DIR,neighbor,goals)
	      Q.put(temp)
	      if self.convert_for_dict(neighbor) not in f_score.keys() or f_score[self.convert_for_dict(neighbor)] > temp[0]:
		  f_score[self.convert_for_dict(neighbor)] = temp[0]
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


  
  def config_to_priorityqueue_tuple(self, dist, config, goals):

    priority= dist+(self.E*self.min_manhattan_dist_to_goals(config,goals))

    return (priority,self.convert_for_dict(config))

  


####################################################################
  def creat_traj_points(self, child,dictin):
    point=[]
    i=0
    point.append(child)

    while dictin.get(point[i],'none')  != 'none':

   
      point.append(dictin[point[i]])
      i=i+1
      
    point=point[::-1]
    	
    return point

  ### TODO ###  (not required but I found it useful)
  #######################################################
  # Pick a heuristic for 
  #######################################################


  #######################################################
  # Convert to and from numpy array to a hashable function
  #######################################################
  def convert_for_dict(self, item):
    return tuple(np.round((item*100),1))
    #return tuple(item)

  def convert_from_dictkey(self, item):
    return np.array(item)/100
    #return np.array(item)



  ### TODO ###  (not required but I found it useful)
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


  ### TODO ###  (not required but I found it useful)
  #######################################################
  # Take the current configuration and apply each of your
  # transition arrays to it
  #######################################################
  def transition_config(self, config):

    new_configs = np.zeros((14,7))
    for i in range (14):
      new_configs[i,:]=config+ self.transition_arrays[i]
    return new_configs

  #######################################################
  # Takes in a list of points, and creates a trajectory
  # that goes between them
  #######################################################
  def points_to_traj(self, points):
    traj = openravepy.RaveCreateTrajectory(self.env,'')
    traj.Init(self.robot.GetActiveConfigurationSpecification())
    for idx,point in enumerate(points):
      traj.Insert(idx,point)
    openravepy.planningutils.RetimeActiveDOFTrajectory(traj,self.robot,hastimestamps=False,maxvelmult=1,maxaccelmult=1,plannername='ParabolicTrajectoryRetimer')
    return traj



  ### TODO ###  (not required but I found it useful)
  #######################################################
  # minimum distance from config to any goal in goals
  # distance metric: euclidean
  # returns the distance AND closest goal
  #######################################################
  def min_euclid_dist_to_goals(self, config, goals):
    # replace the 0 and goal with the distance and closest goal
    return 0, goals[0]


  ### TODO ###  (not required but I found it useful)
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
    #self.robot.Grab(self.target_kinbody) #attaches object to robot, so moving the robot will move the object now
    
    

    time.sleep(20)




if __name__ == '__main__':
  robo = RoboHandler()
  #time.sleep(10000) #to keep the openrave window open
  

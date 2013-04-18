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
TIMESTEP_AMOUNT = 0.02

#constant for max distance to move any joint in a discrete step
TRANS_PER_DIR = 0.1


class RoboHandler:
  def __init__(self):
    self.E=0.3
    self.count=0
    self.openrave_init()
    self.problem_init()
    print('start')
    #self.run_problem_navsearch()
    self.run_problem_nav_and_grasp()


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
    
    # get all bodies for collision checking
    self.bodies = self.env.GetBodies()
    # store the active dof limits for dof limit checking
    self.ActiveDOF_Limits = self.robot.GetDOFLimits(self.manip.GetArmIndices())

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
    #print(self.robot.GetTransform())
    
    #save the current robot transform
    self.start_trans = self.robot.GetTransform()
    #print(tuple(self.start_trans))
    self.start_DOFS = self.robot.GetActiveDOFValues()


    #initialize the transition transformations for base movmement
    self.init_transition_transforms()


  #######################################################
  # navsearch to transform
  #######################################################
  def run_problem_navsearch(self):
    print('running A-star')
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
    
    #print('goal param',x_trans,y_trans,th)
    #print('param to trans',self.params_to_transform(goal_trans[0]))
    #print('param to trans',self.params_to_transform(goal_trans[0]))
    #

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
    print('goal param',x_trans,y_trans,th)
    #print('param to trans',self.params_to_transform(goal_trans[1]))
    #time.sleep(3)

      
    # test to see where the goals are
    #print 'self.start_trans xythera', self.transform_to_params(self.start_trans)
    #time.sleep(10)
    #
    #with self.env:
    #  self.robot.SetTransform(goal_trans[0])
    #print 'goal[0] xythera', self.transform_to_params(goal_trans[0])
    #time.sleep(10)
    #
    #with self.env:
    #  self.robot.SetTransform(goal_trans[1])
    #print 'goal[1] xythera', self.transform_to_params(goal_trans[1])
    #time.sleep(10)
    #return

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
      
    #time.sleep(10)
    
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
    
    # set initial location 
    objTransForm = self.env.GetKinBody("target").GetTransform()
    self.sample_pose = np.copy(self.transform_to_params(self.env.GetKinBody("target").GetTransform()))
    robo_pose = np.array([0,0,0])
    
    while 1 :
      # select a location along a line perpendicular to the close edge
      self.sample_pose = self.sample_for_grasp()
      # return sample pose - robot's start pose
      start_pose = self.transform_to_params(self.start_trans)
      #robo_pose[0] = self.sample_pose[0] - start_pose[0]
      #robo_pose[1] = self.sample_pose[1] - start_pose[1]
      #robo_pose = self.sample_pose - start_pose
      #print 'sample_pose, start_pose', self.sample_pose, start_pose
      #print 'robo_pose', robo_pose
      
      # find relative transform between robot and target location
      #print 'start_trans', self.start_trans
      inv_start_trans = np.linalg.inv(self.start_trans)
      #print 'inv start_trans', inv_start_trans
      new_trans = np.dot(inv_start_trans, self.params_to_transform(self.sample_pose))
      base_transform_goals = [np.copy(self.params_to_transform(self.sample_pose))]
      
      with self.env:
        self.robot.SetTransform(np.dot(self.start_trans, new_trans))
        self.robot.SetActiveDOFValues(self.start_DOFS)
        #self.robot.SetTransform(self.params_to_transform(self.sample_pose))
        #self.robot.SetTransform(np.dot(self.params_to_transform(self.sample_pose),self.params_to_transform(robo_pose)))
      time.sleep(1)
            
      # check environment collision
      if self.env.CheckCollision(self.robot) == False:
        print 'No collision'
        
        # check A* path exist
        with self.env:
          self.robot.SetTransform(self.start_trans)
          self.robot.SetActiveDOFValues(self.start_DOFS)
        time.sleep(1)
        base_transforms = run_func_with_timeout(self.astar_to_transform, args=base_transform_goals, timeout=10)
        
        if base_transforms != None :
          print 'Found nav path'
          
          '''
          # to see if the robot really moves the the position
          with self.env:
            self.robot.SetTransform(np.dot(self.start_trans, new_trans))
            self.robot.SetActiveDOFValues(self.start_DOFS)
          self.run_basetranforms(base_transforms)
          time.sleep(50)
          '''
          
          # check Birrt solution exist
          # set robot to the last position of the found path before searching
          with self.env:
            self.robot.SetTransform(base_transforms[self.cnt-1])
            self.robot.SetActiveDOFValues(self.start_DOFS)
          # set target goal
          target_goals = self.get_goal_dofs()
          grasp_traj = run_func_with_timeout(self.birrt_to_goal, args=target_goals, timeout=10)
          if grasp_traj != None :
            print 'Found birrt solution'
            grasp_traj = self.points_to_traj(grasp_traj)
            '''
            # test to see if robot can really grasp the target
            with self.env:
              self.robot.SetActiveDOFValues(self.start_DOFS)
            self.robot.GetController().SetPath(grasp_traj)
            self.robot.WaitForController(0)
            self.taskmanip.CloseFingers()
            time.sleep(50)
            '''
            
            return base_transforms, grasp_traj
          else :
            print 'No birrt solution'
        else :
          print 'No nav path'
      else :
        'Collision'
            
    # if fails, increase distance away from the close edge  
    
    time.sleep(50)
    return None

  #TODO
  #######################################################
  # Samples a configuration suitable for grasp
  #######################################################
  def sample_for_grasp(self):
    POSE_STEP = 0.1
    
    # decide which edge of the table is close to the object
    # get table and cup position
    self.object_pos = self.transform_to_params(self.env.GetKinBody("target").GetTransform())
    self.table_pos = self.transform_to_params(self.env.GetKinBody("table").GetTransform())
    #print 'self.object_pos', self.object_pos
    #print 'self.table_pos', self.table_pos
    
    # difference between table and object
    self.x_diff_table_object = self.table_pos[0]- self.object_pos[0]
    self.y_diff_table_object = self.table_pos[1]- self.object_pos[1]
    # distance between table and object
    self.x_dist_table_object = np.abs(self.x_diff_table_object)
    self.y_dist_table_object = np.abs(self.y_diff_table_object)
    
    
    if self.object_pos[0] <= self.table_pos[0] :
      print 'object is on the left side of the table'
      if self.object_pos[1] >= self.table_pos[1] :
        print 'object is on the top side of the table'
        if self.x_dist_table_object >= self.y_dist_table_object :
          print 'object is close to left edge of the table'
          # set orientation of robot
          self.sample_pose[2] = 0 #-np.pi/2
          # select position of robot
          self.sample_pose[0] -= POSE_STEP
        else :
          print 'object is close to top edge of the table'
          # set orientation of robot
          self.sample_pose[2] = -np.pi/2
          # select position of robot
          self.sample_pose[1] += POSE_STEP      
      else :
        print 'object is on the down side of the table' 
        if self.x_dist_table_object >= self.y_dist_table_object :
          print 'object is close to left edge of the table'
          # set orientation of robot
          self.sample_pose[2] = 0
          # select position of robot
          self.sample_pose[0] -= POSE_STEP
        else :
          print 'object is close to down edge of the table'
          # set orientation of robot
          self.sample_pose[2] = np.pi/2
          # select position of robot
          self.sample_pose[1] -= POSE_STEP     
    else :
      print 'object is on the right side of the table'
      if self.object_pos[1] >= self.table_pos[1] :
        print 'object is on the top side of the table'
        if self.x_dist_table_object >= self.y_dist_table_object :
          print 'object is close to right edge of the table'
          # set orientation of robot
          self.sample_pose[2] = -np.pi
          # select position of robot
          self.sample_pose[0] += POSE_STEP
        else :
          print 'object is close to top edge of the table'     
          # set orientation of robot
          self.sample_pose[2] = -np.pi/2
          # select position of robot
          self.sample_pose[0] += POSE_STEP
      else :
        print 'object is on the down side of the table'  
        if self.x_dist_table_object >= self.y_dist_table_object :
          print 'object is close to right edge of the table'
          # set orientation of robot
          self.sample_pose[2] = -np.pi
          # select position of robot
          self.sample_pose[0] += POSE_STEP
        else :
          print 'object is close to down edge of the table'
          # set orientation of robot
          self.sample_pose[2] = np.pi/2
          # select position of robot
          self.sample_pose[1] -= POSE_STEP
    
    return self.sample_pose


  #TODO
  #Or just copy your old one here
  #######################################################
  # Bi-Directional RRT
  # find a path from the current configuration to ANY goal in goals
  # goals: list of possible goal configurations
  # RETURN: a trajectory to the goal
  #######################################################
  def birrt_to_goal(self, goals):

    # start the timer
    start = time.time()

    # get initial state before any collision checking
    ini_state = self.robot.GetActiveDOFValues().tolist() # convert ini_state from numpy.ndarray to list

    # verify goals
    goals_verified = self.verify_goals(goals)  # also convert goals(num  py.ndarray) to goals_verified(list)
    if not goals_verified:
      print "RRT: None of the goal is valid"
      return None

    # check whether initial state already meets any goal
    cmpresult = self.compare_against_goals(ini_state, goals_verified)[0]
    if cmpresult == 1:
      print "RRT: The initial state already meets the goal"
      return [ini_state.tolist()]
    elif cmpresult == -1:
      print "RRT: Error in comparison between current state and goal"
      return None

    # initialization
    parent_dict = dict()  # dictionary (representing the tree)
    goal_dict = dict()           
    nearest_state = ini_state
    parent_dict[tuple(ini_state)] = 'root' # add initial state into the dictionary
    
    for g in range (len(goals_verified)):     # create virtual node for all goals
      goal_dict[tuple(goals_verified[g])] = 'root'
      
      Ta=parent_dict
      Tb=goal_dict
      
    i=1 # counter to keep track of swap btw goal tree and ini_strt tree, when i is odd it is ini_strt tree else goal tree
    while (1):
       # choose a target state
      target = self.choose_target(goals)
      #extend Tree to target
      # choose the nearest state to the target state
      nearest = self.find_nearest(Ta.keys(), target)
      

      # extend one unit from the nearest state towards the target state
      [qnew, finish, collision] = self.unit_extend(nearest, target)

      if (finish == -1): continue # redundant arget
      if (collision == True): continue # extension causes collision
      Ta[tuple(qnew)] = nearest
      target = qnew


      # check whether any goal reached after extension
      [cmpresult, goal_index] = self.compare_against_goals(qnew, Tb.keys())

      if cmpresult == 1: # a goal reached
        print "RRT: Found solution"
        # return the trajectory # the chk is kept to reverse only the contact point to init_state and not contact_point to goal
        if i%2 ==0:
            traj = self.return_trajectory(Tb, goal_index, Ta, qnew)
        else:
            traj = self.return_trajectory(Ta, qnew,Tb, goal_index)
            

        #print traj
        now = time.time()
        print 'The time for the search to complete:', now - start
        #print "RRT: Shortening the path"
        #traj = self.shorten_path(traj)
        #now2 = time.time()
        #print 'The time for the path shortening to complete:', now2 - now
        return traj
      elif cmpresult == -1: # error
        print "RRT: Error in comparison between expanded state and goal"
        return None

      # continue extenstion until target state reached
      while (finish != 1):
        # extend one unit towards target state
        nearest = self.find_nearest(Tb.keys(), target)
        [qnew, finish, collision] = self.unit_extend(nearest, target)
        if (collision == True): break
        Tb[tuple(qnew)] = nearest
        #nearest = list(qnew)

        # check whether any goal reached after extension
        [cmpresult, goal] = self.compare_against_goals(qnew, Ta.keys())
        if cmpresult == 1:
          print "RRT: Found solution"
          # return the trajectory # the chk is kept to reverse only the contact point to init_state and not contact_point to goal
          if i%2 ==0:
            traj = self.return_trajectory(Ta, goal, Tb, qnew)
          else:
            traj = self.return_trajectory(Tb, qnew,Ta, goal)
          #print traj
          now = time.time()
          print 'The time for the search to complete:', now - start
          #print "RRT: Shortening the path"     
          #traj = self.shorten_path(traj)
          #now2 = time.time()
          #print 'The time for the path shortening to complete:', now2 - now
          return traj
        elif cmpresult == -1: # error
          print "RRT: Error in comparison between expanded state and goal"
          return None
        
      #swap dictionary  
      tempdict=Ta
      Ta=Tb
      Tb=tempdict

      i=i+1
      #if i%2 ==0:
      #  print('ta is goal')      
      #time.sleep(2)
      

  # choose a target to expand
  # do not handle redundant targets
  def choose_target(self,goals):
    candidate_target = []
    for i in range(len(goals[0])):
      tmp = np.random.uniform(self.ActiveDOF_Limits[0][i], self.ActiveDOF_Limits[1][i])
      candidate_target.append(tmp)
    return candidate_target
  '''
  def choose_target(self,goals):
    prob = np.random.uniform()
    # with probability 1-PROB_GOALS, choose a random target within dof limits

    if (prob > PROB_GOALS):
      
      candidate_target = []
      for i in range(len(goals[0])):
        tmp = np.random.uniform(self.ActiveDOF_Limits[0][i], self.ActiveDOF_Limits[1][i])
        candidate_target.append(tmp)
      return candidate_target
    # with probability PROB_GOALS, randomly choose a goal
    else:
      a=np.random.randint(0, len(goals), 1)
      print('index of goal',a) 
      return goals.tolist()[a] # using min-distance may end up stuck in local minima
  '''
      
  # find the tree's nearest state to the target
  def find_nearest(self, parent_dict_keys, target):
    [dist, index] = self.min_euclid_dist_one_to_many(target, parent_dict_keys)
    return list(parent_dict_keys[index])

  # extend the tree from the state that is the nearest to the target toward the target
  # return:
  #   milestone: the extended state
  #   finish: whether milestone meets the target, 1 -> reached target  0 -> still in extension process  -1 -> nearest and meets target or error (redundant target)
  #   collision: whether milestone causes collision
  def unit_extend(self, nearest, target):

    # initialization
    milestone = list(nearest)
    finish = 1
    collision = False

    # check whether the nearest already meets target,
    if self.compare_state(nearest, target) != 0:
      milestone = None
      finish = -1

      return [milestone, finish, collision]
    # extend from nearest to target, each dimension by at most MAX_MOVE_AMOUNT
    len_target = len(target)

    for i in range(len_target):
      cmpresult = self.compare_value(nearest[i], target[i])
      if (cmpresult > 0): milestone[i] += MAX_MOVE_AMOUNT
      elif (cmpresult < 0): milestone[i] -= MAX_MOVE_AMOUNT
      if finish == 1:
        if self.compare_value(milestone[i], target[i]) != 0: finish = 0 # set finish to 0 if at least one dimenstion of target not met

    # check collision
    [objcollision, selfcollision, not_used] = self.state_validation(milestone, True, False)
    collision = objcollision or selfcollision
    return [milestone, finish, collision]

  # shorten the path and return the shortened trajectory
  def shorten_path(self, traj):
    start = time.time() # start the timer
    while(1):
      # exit when maximum time reached or trajectory short enough
      now = time.time()
      len_traj = len(traj)
      if now-start >= MAX_TIME_PATH_SHORTEN or len_traj <= MIN_THRESH_TRAJ_LEN: return traj
      # random select two states to shorten the path in between, make sure index1 < index2
      [index1, index2] = np.random.randint(0, len_traj, 2)
      if index1 == index2: continue
      if index1 > index2:
        tmp = index1
        index1 = index2
        index2 = tmp
      # call shorten_path_recur to shorten the path
      [collision, new_states] = self.shorten_path_recur(traj[index1], traj[index2])
      if collision == True: continue # path shortening failed due to collision
      else: # path shortening successful, modify the trajectory
        for i in reversed(range(index1+1, index2)): traj.pop(i) # remove redundant path between the two selected states
        for i in range(len(new_states)): new_states.insert(index1+i, new_states[i]) # add new path
 
  # recursive function to shorten the path between state1 and state2 (state1 always closer to initial state than state2)
  # return:
  #   collision: whether the shortened path cause collision
  #   new_states: the new path (a list of states)
  def shorten_path_recur(self, state1, state2):
    # initialization
    adjacent = True # whether the two states are adjacent, used to end the recursive loop
    collision = False
    len_state = len(state1)
    new_state = list(state1) # the new state between state1 and state2, bisection method used for collision
    new_states = [] # the new path (a list of states)
    for i in range(len_state): # for each dimension of the state
      diff = round((state2[i] - state1[i]), 1) # unless the goal state chosen, diff should be approximately a multiple of MAX_MOVE_AMOUNT, rounding takes care of accuracy problem due to floating point
      if diff > MAX_MOVE_AMOUNT or diff < -MAX_MOVE_AMOUNT: # the two states are not adjacent
        adjacent = False
        new_state[i] = state1[i] + round(diff/2, 1) # bisection method used
    if adjacent == True: return [collision, new_states] # adjacent states, return collsion = False AND new_states = empty list
    else: # not adjacent states, continue recursive loop
      [objcollision, selfcollision, not_used] = self.state_validation(new_state, True, False) # collision detection
      if objcollision or selfcollision == True: # ends recursion due to collision, return collsion = True and the whole path shortening fails
        collision = True
        return [collision, new_states]
      else: # no collision, continue recursive loop
        [collision_lo, new_states_lo] = self.shorten_path_recur(state1, new_state)
        [collision_hi, new_states_hi] = self.shorten_path_recur(new_state, state2)
        collision = collision_lo or collision_hi
    if collision == True: return [collision, new_states] # ends recursion due to collision in children loop, return collsion = True and the whole path shortening fails
    else: # join the path returned by children loop with new_state in between (starting states: state1 and state2 not included)
          new_states = new_states_lo + [new_state] + new_states_hi
          return [collision, new_states]
    
  # check whether goals reachable, remove goals unreachable
  # return the verified goals, may be empty
  def verify_goals(self, goals):
    goals_verified = []
    for i in range(len(goals)):  # for each goal
      if len(goals[i]) == 0: continue  # if goal is somehow empty
      # check collision and check against dof limits
      [objcollision, selfcollision, goaldoffail] = self.state_validation(goals[i], True, True)
      # attach verified goal to goals_verified
      if objcollision == False and selfcollision == False and goaldoffail == False:
        goals_verified.append(goals[i])
    return goals_verified

  # check whether the state causes collision or exceeds dof limit
  # return:
  #   objcollision: True if collide with object
  #   selfcollision: True if cllide with the robot itself
  #   goaldoffail: True if exceed dof limit
  def state_validation(self, state, collisionchecking, dofchecking):
    with self.env:
      self.robot.SetActiveDOFValues(state)
    objcollision = False
    selfcollision = False
    goaldoffail = False
    if collisionchecking == True: # if collision checking requested
      # check against each body
      for i in range(len(self.bodies)):      
        # check object collision
        if self.env.CheckCollision(self.robot, self.bodies[i]): 
          objcollision = True
          break
        # check self collision
        if self.robot.CheckSelfCollision():
          selfcollision = True
          break
    if dofchecking == True and objcollision == False and collisionchecking == False: # if dof checking requested
      # check against dof limits
      for i in range(len(state)):
        if (state[i] < self.ActiveDOF_Limits[0][self.manip.GetArmIndices()[i]] or state[i] > self.ActiveDOF_Limits[1][self.manip.GetArmIndices()[i]]): # if state is out of DOF limits
          goaldoffail = True
          break
    return [objcollision, selfcollision, goaldoffail]
    
  # return the trajectory
  def return_trajectory(self, goals, goal_index, parent_dict, milestone):
    traj=[]
    traj_goal = []
    traj_strt = []
    traj_goal.append(goal_index) # append the goal
    tmpgoal=goal_index
    # expand goaltree 1
    while (1):
      tmpgoal2 = goals[tuple(tmpgoal)]
      if (goals.has_key(tuple(tmpgoal2))): # check whether tmpstate2 is "root"
        traj_goal.append(tmpgoal2)
        tmpgoal = tmpgoal2
      else: break
    #expand parent_tree    
    tmpstate = milestone
    traj_strt.append(milestone) # append the milestone (which meets the goal) to avoid failure in dictionary key checking
    # iterate until finding the root
    while (1):
      tmpstate2 = parent_dict[tuple(tmpstate)]
      if (parent_dict.has_key(tuple(tmpstate2))): # check whether tmpstate2 is "root"
        traj_strt.append(tmpstate2)
        tmpstate = tmpstate2
      else: break
    traj_strt.reverse() # before revers(), the trajectory is from goal to initial state
    
    # move from intital state to point of contact and from point of contact to goal
    traj=traj_strt+traj_goal
    return traj

  # compare a state against a number of goals
  # return
  #   result: 0 -> NOT equal to any goal; 1 -> equal to a goal; -1 -> error
  #   goal_index: the index of the goal that is met
  def compare_against_goals(self, curr_state, goals):
    result = 0
    goal_index = -1
    for i in range(len(goals)):
      cmpresult = self.compare_state(curr_state, goals[i])
      if cmpresult == 1:
        result = 1
        goal_index = i
        return [result, goals[goal_index]]
      elif cmpresult == -1:
        result = -1
        return [result, goals[goal_index]]
    return [result, goals[goal_index]]

  # compare whether two states are equal with a MAX_MOVE_AMOUNT/2 error tolerance
  # return 0 -> NOT equal; 1 -> equal; -1 -> error
  def compare_state(self, curr_state, goal):

    len_curr_state = len(curr_state)
    len_goal = len(goal)
    if len_curr_state != len_goal: return -1
    for i in range(len_curr_state):
      if curr_state[i] >= goal[i]+MAX_MOVE_AMOUNT/2 or curr_state[i] < goal[i]-MAX_MOVE_AMOUNT/2:
        return 0
    return 1

  # compare whether two values are equal with a MAX_MOVE_AMOUNT/2 error tolerance
  # return direction: 0 -> equal; 1 -> value1 should increase ; -1 -> value1 should decrease
  def compare_value(self, value1, value2):
    direction = 0
    if value1 >= value2 + MAX_MOVE_AMOUNT/2:
      direction = -1
      return direction
    if value1 < value2 - MAX_MOVE_AMOUNT/2:
      direction = 1
      return direction
    return direction  

  #TODO
  #######################################################
  # BASE MOVEMENT with A* SEARCH
  # find a path from the current configuration to transform
  # RETURN: an array of ALL intermediate transforms.
  # Thus, you should use self.full_transforms when returning!
  #######################################################
  def astar_to_transform(self, goal_transforms):
    # test for a star using HW2
    base_trans=self.search_to_goal_astar(goal_transforms)
    return base_trans
  
  
  #######################################################
  # A* SEARCH
  # find a path from the current configuration to ANY goal in goals
  # goals: list of possible goal configurations
  # RETURN: a trajectory to the goal
  #######################################################
  def search_to_goal_astar(self,goals): 
    print('Running Astar')
    
    # start the timer
    start = time.time()
    
    initial =self.transform_to_params(self.start_trans)
    #print('initial',initial)
    goal_size=np.shape(goals)
    #print('goal_size',goal_size,len(goals))
    goal = set()
    for element in range(goal_size[0]):
        #print('goal_pra',self.transform_to_params(goals[element]))
        #print('goal_pra af',self.convert_for_dict_withround(self.transform_to_params(goals[element])))
        goal.add(self.convert_for_dict_withround(self.transform_to_params(goals[element])))
        #time.sleep(1)
    #config=self.convert_for_dict_withround(config)
    vco = set()
    f_score=dict()
    #g_score=dict()
    Q = Queue.PriorityQueue() #openset
    path = dict()
    path_ind = dict()
    ini_c=self.config_to_priorityqueue_tuple(0,self.convert_for_dict_withround(initial),goal)
    #print('ini_c', ini_c)
    Q.put(ini_c)
    f_score[ini_c[1]]=ini_c[0]
    
    #print('initial crap')
    #print('Que',Q)
    #print('f_score',f_score)
    while not Q.empty():
      parent=Q.get() # pass x,y,theta
      #print('goal',goal)

      #print('parent',parent[1])
      #time.sleep(2)
      #print('priority',parent[0])
      
      #time.sleep(1)      
      current=(self.convert_from_dictkey_withround(parent[1]))
      #print('current',current)
      g_score=parent[0]-self.E*self.dist_to_goals((parent[1]),goal)
      
      #print('g_score',g_score,self.E*self.dist_to_goals(current,goal))
      #time.sleep(1)
      
      if parent[1] in goal or self.is_at_goal_basesearch(self.convert_from_dictkey_withround(parent[1]), goal):
        print('success')

        base=self.creat_traj_base(parent[1],path)
        base_trans=self.creat_transform(base,path_ind)

        return base_trans
      
      if parent[1] not in vco:
        #print('being added to vco',parent[1])
        vco.add(parent[1])
        #print('visited after add',vco)
        action=self.transition_config(self.params_to_transform(current))
        #print('set of action',action)
        action_size=np.shape(action)
        #print('action size',action_size)
        for neig_trans in range(action_size[0]):
          #print('in neighbour trans',action[neig_trans])
            
          neighbor=self.transform_to_params(action[neig_trans])
          #print('neighbor',neighbor)
          #time.sleep(5)
          #print('in neighbour config',neighbor)
          
          with self.env:
            self.robot.SetTransform(action[neig_trans])
            if (self.env.CheckCollision(self.robot) or self.robot.CheckSelfCollision()):
              #print 'astar collision'
              continue
            
          if self.convert_for_dict_withround(neighbor) not in vco:
            #print('self.convert_for_dict_withround(neighbor) not in vco')
            #temp=self.config_to_priorityqueue_tuple(g_score+self.dist_to_goal(parent[1],self.convert_for_dict_withround(neighbor)),self.convert_for_dict_withround(neighbor),goal)
            temp=self.config_to_priorityqueue_tuple(g_score+1,self.convert_for_dict_withround(neighbor),goal)
            #print('temp',temp)
            Q.put(temp)
            #time.sleep(5)
            if self.convert_for_dict_withround(neighbor) not in f_score.keys() or f_score[self.convert_for_dict_withround(neighbor)] > temp[0]:
              f_score[self.convert_for_dict_withround(neighbor)] = temp[0]
              #print 'nei',tuple(neighbor),'curr',tuple(current)
              path[temp[1]] = self.convert_for_dict_withround(current)

              path_ind[temp[1]] = neig_trans
    
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
  
  
  def creat_traj_base(self, child,dictin):
    #print 'child', child
    base=[]
    i=0
    base.append(child)
    #print 'base',base
    #print 'dict', dictin[base[i]]

    while dictin.get(base[i],'none') !='none':
      #print('base',base[i])
      base.append(dictin[base[i]])
      i=i+1
      
    base=base[::-1]
    return base
  
  def creat_transform(self, base,path_ind):
    base_transform=[np.copy(self.start_trans)]
    base_transform[0][0:4,0:4]=self.start_trans
    cnt=1
    #print 'base',base[i]
    for i in range (1,len(base)):
      
      index=path_ind[base[i]]

      for i in range (0,5):
        base_transform.append(np.copy(self.start_trans))
        base_transform[cnt][0:4,0:4]=np.dot(base_transform[cnt-1],self.full_transforms[index])

        cnt=cnt+1
    self.cnt=cnt-1
    return base_transform
   #######################################################
  # Check if the config is close enough to goal
  # Returns true if any goal in goals is within
  # BOTH distance_thresh and theta_thresh
  #######################################################
  def is_at_goal_basesearch(self, config, goals, dist_thresh = 0.02, theta_thresh = np.pi/12):
    for goal in goals:
      # iljoo : config needs to be converted to for_dickey_withround?
      goal=self.convert_from_dictkey_withround(goal)
      #print ('config[0:2]',config[0:2])
      #print ('goal[0:2]', goal[0:2])
      #print ('config[2]', config[2])
      #print ('goal[2]', goal[2])
      #print ('check done', np.linalg.norm(config[0:2]-goal[0:2]), np.abs(config[2] - goal[2]))
      if (np.linalg.norm(config[0:2]-goal[0:2]) <= dist_thresh and np.abs(config[2] - goal[2]) <= theta_thresh):
        return True
    return False



  #TODO
  #######################################################
  # Initialize the movement transforms
  # These are equivalent to applying a fixed control for some
  # amount of time
  # for navigation of the robot base
  #######################################################
  def init_transition_transforms(self):
    self.transition_transforms = [np.copy(self.start_trans)]
    self.full_transforms = [np.copy(self.start_trans)]
    
    with self.env:
      self.robot.SetTransform(self.start_trans)

    TIMESTEP=1
    # get initial x, y, thera
    # x: param[0], y: param[1], thera: param[2]
    cur_param = self.transform_to_params(self.start_trans)

    self.count=0

    w_1 = 1
    w_2 = 1    
    new_param = self.get_xythera_from_angularvel(w_1, w_2, TIMESTEP, cur_param)
    #print 'new param' ,new_param
    self.transition_transforms [self.count][0:4,0:4]=self.params_to_transform(new_param)
    #print(self.transition_transforms [self.count])
   

    #
    self.count=self.count+1
    w_1 = 0
    w_2 = 1  
    new_param = self.get_xythera_from_angularvel(w_1, w_2, TIMESTEP, cur_param)
    #print 'new param' ,new_param
    self.transition_transforms.append(np.copy(self.start_trans))
    self.transition_transforms [self.count][0:4,0:4]=self.params_to_transform(new_param)
    #print(self.transition_transforms [self.count])
    
    self.count=self.count+1
    w_1 = 1
    w_2 = 0  
    new_param = self.get_xythera_from_angularvel(w_1, w_2, TIMESTEP, cur_param)
    #print 'new param' ,new_param
    self.transition_transforms.append(np.copy(self.start_trans))
    self.transition_transforms [self.count][0:4,0:4]=self.params_to_transform(new_param)
    #print(self.transition_transforms [self.count])
    
    self.count=self.count+1       
    w_1 = 1
    w_2 = -1
    new_param = self.get_xythera_from_angularvel(w_1, w_2, TIMESTEP, cur_param)
    #print 'new param' ,new_param
    self.transition_transforms.append(np.copy(self.start_trans))
    self.transition_transforms [self.count][0:4,0:4]=self.params_to_transform(new_param)
    #print(self.transition_transforms [self.count])
    
    self.count=self.count+1
    w_1 = -1
    w_2 = 1    
    new_param = self.get_xythera_from_angularvel(w_1, w_2, TIMESTEP, cur_param)
    #print 'new param' ,new_param
    self.transition_transforms.append(np.copy(self.start_trans))
    self.transition_transforms [self.count][0:4,0:4]=self.params_to_transform(new_param)
    #print(self.transition_transforms [self.count])
    
    self.count=0
    TIMESTEP=0.2

    w_1 = 1
    w_2 = 1    
    new_param = self.get_xythera_from_angularvel(w_1, w_2, TIMESTEP, cur_param)
    #print 'new param' ,new_param
    self.full_transforms [self.count][0:4,0:4]=self.params_to_transform(new_param)
    #print(self.full_transforms [self.count])
   

    #
    self.count=self.count+1
    w_1 = 0
    w_2 = 1  
    new_param = self.get_xythera_from_angularvel(w_1, w_2, TIMESTEP, cur_param)
    #print 'new param' ,new_param
    self.full_transforms.append(np.copy(self.start_trans))
    self.full_transforms [self.count][0:4,0:4]=self.params_to_transform(new_param)
    #print(self.full_transforms [self.count])
    
    self.count=self.count+1
    w_1 = 1
    w_2 = 0  
    new_param = self.get_xythera_from_angularvel(w_1, w_2, TIMESTEP, cur_param)
    #print 'new param' ,new_param
    self.full_transforms.append(np.copy(self.start_trans))
    self.full_transforms [self.count][0:4,0:4]=self.params_to_transform(new_param)
    #print(self.full_transforms [self.count])
    
    self.count=self.count+1       
    w_1 = 1
    w_2 = -1
    new_param = self.get_xythera_from_angularvel(w_1, w_2, TIMESTEP, cur_param)
    #print 'new param' ,new_param
    self.full_transforms.append(np.copy(self.start_trans))
    self.full_transforms [self.count][0:4,0:4]=self.params_to_transform(new_param)

    
    self.count=self.count+1
    w_1 = -1
    w_2 = 1    
    new_param = self.get_xythera_from_angularvel(w_1, w_2, TIMESTEP, cur_param)
    self.full_transforms.append(np.copy(self.start_trans))
    self.full_transforms [self.count][0:4,0:4]=self.params_to_transform(new_param)

    
    # test to see how the action set move  
    #time.sleep(5)
    #for i in range(0, self.count+1):    TIMESTEP
    #  with self.env:
    #    self.robot.SetTransform(np.dot(self.start_trans, self.transition_transforms[i]))
    #  print 'transition_transforms[i]', i, self.transform_to_params(self.transition_transforms[i])
    #  time.sleep(5)
    #time.sleep(50)
    
  
  # get x, y, thera from differential angular velocities w1 and w2
  # input)
  #    w1, w2: angular velocities
  #    x,y,thera in previous time step: numpy array, x: param[0], y: param[1], thera: param[2]
  # output)
  #    x,y,thera in current time step: numpy array, x: param[0], y: param[1], thera: param[2]
  def get_xythera_from_angularvel(self,w_1,w_2,time_step, prev_param):
    thera_t = (WHEEL_RADIUS/(2*ROBOT_LENGTH))*(w_1-w_2)*time_step
    #+ prev_param[2]
    #thera_t = prev_param[2]-thera_t
    y_t = (WHEEL_RADIUS/2)*(w_1*np.sin(thera_t)+w_2*np.sin(thera_t))*time_step
    #+ prev_param[0]
    x_t = (WHEEL_RADIUS/2)*(w_1*np.cos(thera_t)+w_2*np.cos(thera_t))*time_step
    #+ prev_param[1]
    return np.array([x_t, y_t, thera_t])
    #return np.array([prev_param[0]-x_t, prev_param[1]-y_t, thera_t])
    
  
  #TODO
  #######################################################
  # Applies the specified controls to the initial transform
  # returns a list of all intermediate transforms
  #######################################################
  def controls_to_transforms(self,w_1,w_2,time_step):
    thera_t = (WHEEL_RADIUS/(2*ROBOT_LENGTH))*(w_1-w_2)*time_step
    #+ prev_param[2]
    #thera_t = prev_param[2]-thera_t
    y_t = (WHEEL_RADIUS/2)*(w_1*np.sin(thera_t)+w_2*np.sin(thera_t))*time_step
    #+ prev_param[0]
    x_t = (WHEEL_RADIUS/2)*(w_1*np.cos(thera_t)+w_2*np.cos(thera_t))*time_step
    #+ prev_param[1]
    return np.array([x_t, y_t, thera_t])
    return None
  
  

  #TODO
  #######################################################
  # Take the current configuration and apply each of your
  # transition arrays to it
  #######################################################
  def transition_config(self, config):
    new_configs=[np.copy(self.start_trans)]
    for i in range(0,self.count+1):
      new_configs[i][0:4,0:4] =np.dot(config,self.transition_transforms[i])

      
      if i <self.count:
        new_configs.append(np.copy(self.start_trans))

    #print 'new_configs', new_configs

    return new_configs

  #TODO
  #######################################################
  # Implement a heuristic for base navigation
  #######################################################
  #def config_to_priorityqueue_tuple(self, dist, config, goals):
  #  # make sure to replace the 0 with your priority queue value!
  #  return (0.0, config.tolist())
  
  def config_to_priorityqueue_tuple(self, dist, config, goals):
    #print('dist',dist)

    #config=self.convert_for_dict_withround(config)
    priority= dist+(self.E*self.dist_to_goals(config,goals))
    #print(self.dist_to_goals(config,goals),self.E*self.dist_to_goals(config,goals)) 
    #print('priority',priority)
    return (priority,config)


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
    #print 'trans', np.array([transform[0,3], transform[1,3], self.rot_matrix_to_angle(transform)])
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
    for trans in range(0,self.cnt):
      #print trans,self.cnt
      #print transforms[trans]
      with self.env:
        self.robot.SetTransform(transforms[trans])
      time.sleep(0.1)
      #print 'basetransform', trans


  #######################################################
  # minimum distance from config (singular) to any other config in o_configs
  # distance metric: euclidean
  # returns the distance AND index
  #######################################################
  def min_euclid_dist_one_to_many(self, config, o_configs):
    #dists = np.sum((config-o_configs)**2,axis=1)**(1./2)
    #min_ind = np.argmin(dists)
    #return dists[min_ind], min_ind
    dists = np.sum((np.array(config)-np.array(o_configs))**2,axis=1)**(1./2)
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
  #def min_manhattan_dist_to_goals(self, config, goals):
  #  dist_g = []
  #  for g in goals:
  #    dist_g .append(sum(tuple([abs(round(item1 - item2,1)) for item1, item2 in zip(config,tuple(g))])))
  #  return min(dist_g)
    
    
  def dist_to_goals(self, config, goals):
    dists =[]
    inds = []

    for goal in goals:
      #dist = np.abs(config[0]-goal[0])+np.abs(config[1] - goal[1]) #+np.abs(config[2] - goal[2])
      
      dist =np.linalg.norm(np.asarray(config[0:2])-np.asarray(goal[0:2]))#+ np.abs(config[2] - goal[2])
      dists.append(dist)
      inds.append(goal)
    min_ind_in_inds = np.argmin(dists)
    #print('dist',dists)
    #print('mim',dists[min_ind_in_inds])
    #time.sleep(3)
    return dists[min_ind_in_inds]
  
  def dist_to_goal(self, config, goal):
    #print('dist config',config)
    #print('dist goals',goal)
    #dist = np.abs(config[0]-goal[0])+np.abs(config[1] - goal[1]) #+np.abs(config[2] - goal[2])
    dist =np.linalg.norm(np.asarray(config[0:2])-np.asarray(goal[0:2]))#+ np.abs(config[2] - goal[2])
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
  time.sleep(10000) #to keep the openrave window open
  


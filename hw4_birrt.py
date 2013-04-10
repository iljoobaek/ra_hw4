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
MAX_MOVE_AMOUNT = 0.1

# global variables for parameter tuning
PROB_GOALS = 0.2 # probability for choosing a goal as the expansion target
MIN_THRESH_TRAJ_LEN = 5 # minimum threshold of satisfactory trajectory length
MAX_TIME_PATH_SHORTEN = 5 # maximum time allowed for path shortening

class RoboHandler:
  def __init__(self):
    self.openrave_init()
    self.problem_init()

    self.run_problem_birrt()
    time.sleep(100)




  #######################################################
  # the usual initialization for openrave
  #######################################################
  def openrave_init(self):
    self.env = openravepy.Environment()
    self.env.SetViewer('qtcoin')
    self.env.GetViewer().SetName('HW4 Viewer')
    self.env.Load('models/%s_birrt.env.xml' %PACKAGE_NAME)
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
    #self.ikmodel = openravepy.databases.inversekinematics.InverseKinematicsModel(self.robot,iktype=openravepy.IkParameterization.Type.Transform6D)
    #if not self.ikmodel.load():
    #  self.ikmodel.autogenerate()

    # create taskmanip
    self.taskmanip = openravepy.interfaces.TaskManipulation(self.robot)
  
    # move left arm out of way
    self.robot.SetDOFValues(np.array([4,2,0,-1,0,0,0]),self.robot.GetManipulator('left_wam').GetArmIndices() )


  #######################################################
  # Harder search problem from last time - use an RRT to solve
  #######################################################
  def run_problem_birrt(self):
    self.robot.GetController().Reset()

    # move hand to preshape of grasp
    # --- important --
    # I noted they were all the same, otherwise you would need to do this separately for each grasp!
    with self.env:
      self.robot.SetDOFValues(self.grasps[0][self.graspindices['igrasppreshape']], self.manip.GetGripperIndices()) # move to preshape
    

    #goals = self.get_goal_dofs(10,3)
    goals = np.array([[ 1.53442279, -1.11094749,  0.2       ,  1.89507469,  0.9253871 ,
        -0.27590187, -0.93353661],
       [ 1.08088326, -1.11094749, -0.2       ,  1.89507469, -1.15533182,
        -0.47627667,  1.40590175],
       [ 1.64865961, -1.08494965,  0.3       ,  1.89507469,  1.12567395,
        -0.42894989, -1.20064072],
       [ 1.58020381, -1.09009898,  0.3       ,  1.88331188,  1.12057975,
        -0.38546846, -1.14447409],
       [ 1.69349022, -1.05374533,  0.4       ,  1.88331188,  1.2076898 ,
        -0.55054165, -1.30156536],
       [ 1.80822781, -1.00617436,  0.5       ,  1.88331188,  1.23775906,
        -0.72454447, -1.40740396],
       [ 0.99085319, -1.15391791, -0.2       ,  2.02311018, -0.73232284,
        -0.60044153,  0.9098408 ],
       [ 1.56004258, -1.12730671,  0.3       ,  2.02311018,  0.68660509,
        -0.56962218, -0.85889052],
       [ 1.67574177, -1.08946411,  0.4       ,  2.02311018,  0.83605503,
        -0.69762048, -1.08462636],
       [ 0.98566097, -1.15236693, -0.2       ,  2.03233934, -0.72377213,
        -0.61047535,  0.90372445],
       [ 1.55901234, -1.12557036,  0.3       ,  2.03233934,  0.67519725,
        -0.57794147, -0.84513898],
       [ 1.67568121, -1.08744563,  0.4       ,  2.03233934,  0.82590826,
        -0.7053313 , -1.07222512],
       [ 3.62542331, -0.50373029, -0.1       ,  2.15372919, -0.90608947,
        -1.35422117,  1.22439759],
       [ 4.1163159 , -0.54152784, -0.2       ,  2.15372919, -0.82842861,
        -1.04081465,  0.94191546],
       [ 3.62542331, -0.50373029, -0.1       ,  2.15372919, -4.04768212,
         1.35422117, -1.91719506],
       [ 1.08601757, -1.12399348, -0.1       ,  1.98216027, -0.53511583,
        -0.50586635,  0.66089972],
       [ 1.44668278, -1.10760314,  0.2       ,  1.98216027,  0.44896204,
        -0.47742308, -0.55906299],
       [ 1.5684208 , -1.07995335,  0.3       ,  1.98216027,  0.68165593,
        -0.5789909 , -0.87398179],
       [ 1.69349022, -1.05374533,  0.4       ,  1.88331188,  1.2076898 ,
        -0.55054165,  1.8400273 ],
       [ 1.58020381, -1.09009898,  0.3       ,  1.88331188,  1.12057975,
        -0.38546846,  1.99711856],
       [ 1.58020381, -1.09009898,  0.3       ,  1.88331188, -2.0210129 ,
         0.38546846, -1.14447409],
       [ 3.49661161, -0.34059995, -0.1       ,  1.38477553,  1.20833943,
         1.53448864, -0.39066223],
       [ 3.88076306, -0.36079555, -0.2       ,  1.38477553,  1.01389006,
         1.32684258, -0.28712797],
       [ 4.55120287, -0.42927425, -0.3       ,  1.38477553,  0.50597369,
         1.0068676 ,  0.07352285],
       [ 1.71823564, -1.04694097,  0.5       ,  2.01730926,  0.91767346,
        -0.80895727,  1.95274455],
       [ 1.60263915, -1.09602265,  0.4       ,  2.01730926,  0.81743246,
        -0.66449298,  2.13438883],
       [ 1.83615837, -0.98539873,  0.6       ,  2.01730926,  0.97511267,
        -0.96908448,  1.8045713 ],
       [ 1.60313817, -1.09414142,  0.4       ,  2.01536424,  0.81746904,
        -0.66473871, -1.0084334 ],
       [ 1.71902033, -1.04498968,  0.5       ,  2.01536424,  0.91747166,
        -0.8094239 , -1.19031272],
       [ 1.83728186, -0.98334683,  0.6       ,  2.01536424,  0.97461756,
        -0.96979975, -1.33875245]]) 
 
    with self.env:
      self.robot.SetActiveDOFValues([5.459, -0.981,  -1.113,  1.473 , -1.124, -1.332,  1.856])

    # get the trajectory!
    #traj = self.birrt_to_goal(goals)
    traj = self.rrt_to_goal(goals)

    with self.env:
      self.robot.SetActiveDOFValues([5.459, -0.981,  -1.113,  1.473 , -1.124, -1.332,  1.856])

    traj = self.points_to_traj(traj)
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


  #TODO
  #######################################################
  # Bi-Directional RRT
  # find a path from the current configuration to ANY goal in goals
  # goals: list of possible goal configurations
  # RETURN: a trajectory to the goal
  #######################################################
  def birrt_to_goal(self, goals):
    return None


  #######################################################
  # RRT
  # find a path from the current configuration to ANY goal in goals
  # goals: list of possible goal configurations
  # RETURN: a trajectory to the goal
  #######################################################
  def rrt_to_goal(self, goals):

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
        print "RRT: Shortening the path"
        traj = self.shorten_path(traj)
        now2 = time.time()
        print 'The time for the path shortening to complete:', now2 - now
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
          print "RRT: Shortening the path"     
          traj = self.shorten_path(traj)
          now2 = time.time()
          print 'The time for the path shortening to complete:', now2 - now
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
  
  
  #######################################################
  # Convert to and from numpy array to a hashable function
  #######################################################
  def convert_for_dict(self, item):
    #return tuple(np.int_(item*100))
    return tuple(item)

  def convert_from_dictkey(self, item):
    #return np.array(item)/100.
    return np.array(item)



  def points_to_traj(self, points):
    traj = openravepy.RaveCreateTrajectory(self.env,'')
    traj.Init(self.robot.GetActiveConfigurationSpecification())
    for idx,point in enumerate(points):
      traj.Insert(idx,point)
    openravepy.planningutils.RetimeActiveDOFTrajectory(traj,self.robot,hastimestamps=False,maxvelmult=1,maxaccelmult=1,plannername='ParabolicTrajectoryRetimer')
    return traj




  #######################################################
  # minimum distance from config (singular) to any other config in o_configs
  # distance metric: euclidean
  # returns the distance AND index
  #######################################################
  #def min_euclid_dist_one_to_many(self, config, o_configs):
  #  dists = np.sum((config-o_configs)**2,axis=1)**(1./2)
  #  min_ind = np.argmin(dists)
  #  return dists[min_ind], min_ind

  def min_euclid_dist_one_to_many(self, config, o_configs):
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
  # close the fingers when you get to the grasp position
  #######################################################
  def close_fingers(self):
    self.taskmanip.CloseFingers()
    self.robot.WaitForController(0) #ensures the robot isn't moving anymore
    #self.robot.Grab(target) #attaches object to robot, so moving the robot will move the object now




if __name__ == '__main__':
  robo = RoboHandler()
  #time.sleep(10000) #to keep the openrave window open
  

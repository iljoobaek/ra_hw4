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

    self.run_problem_constrain_birrt()




  #######################################################
  # the usual initialization for openrave
  #######################################################
  def openrave_init(self):
    self.env = openravepy.Environment()
    self.env.SetViewer('qtcoin')
    self.env.GetViewer().SetName('HW4 Viewer')
    self.env.Load('models/%s_constrain_rrt.env.xml' %PACKAGE_NAME)
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
  
    # move left arm out of way
    self.robot.SetDOFValues(np.array([4,2,0,-1,0,0,0]),self.robot.GetManipulator('left_wam').GetArmIndices() )
    
    # initialize values for jacobian
    self.jacobian_init()


  #######################################################
  # use a Constrained Bi-Directional RRT to solve
  #######################################################
  def run_problem_constrain_birrt(self):
    self.robot.GetController().Reset()
    #startconfig = np.array([ 4.54538305,  1.05544618,  0., -0.50389025, -3.14159265,  0.55155592, -2.97458672])
    startconfig = np.array([ 2.37599388,-0.32562851, 0.,         1.61876989,-3.14159265, 1.29314139, -0.80519756])

    # move hand to preshape of grasp
    # --- important --
    # I noted they were all the same, otherwise you would need to do this separately for each grasp!
    with self.env:
      self.robot.SetDOFValues(self.grasps[0][self.graspindices['igrasppreshape']], self.manip.GetGripperIndices()) # move to preshape
    

    #goals = self.get_goal_dofs(10,3)
    goals = np.array([[ 2.3056527 , -0.6846652 ,  0.9       ,  1.88331188, -2.00747441,
         1.51061724,  1.39244389],
       [ 2.17083819, -0.78792566,  0.8       ,  1.88331188, -1.95347928,
         1.29737952,  1.48609958],
       [ 2.04534288, -0.87443137,  0.7       ,  1.88331188, -1.91940061,
         1.0969241 ,  1.56834924],
       [ 1.92513441, -0.94671347,  0.6       ,  1.88331188, -1.90216145,
         0.9064694 ,  1.64815832],
       [ 1.80822781, -1.00617436,  0.5       ,  1.88331188, -1.90383359,
         0.72454447,  1.7341887 ],
       [ 1.69349022, -1.05374533,  0.4       ,  1.88331188, -1.93390285,
         0.55054165,  1.8400273 ],
       [ 1.58020381, -1.09009898,  0.3       ,  1.88331188, -2.0210129 ,
         0.38546846,  1.99711856],
       [ 5.36700926,  0.74723102, -2.3       ,  1.88331188, -1.9731337 ,
         1.38427661,  1.44916799],
       [ 5.23832771,  0.84028589, -2.4       ,  1.88331188, -1.93147499,
         1.17896707,  1.53489493],
       [ 1.58020381, -1.09009898,  0.3       ,  1.88331188,  1.12057975,
        -0.38546846, -1.14447409],
       [ 0.99085319, -1.15391791, -0.2       ,  2.02311018, -3.8739155 ,
         0.60044153, -2.23175185],
       [ 2.16880663, -0.80864954,  0.8       ,  2.02311018, -2.14530038,
         1.37567029,  1.48036157],
       [ 2.03768854, -0.90096949,  0.7       ,  2.02311018, -2.14142654,
         1.18510744,  1.60819768],
       [ 1.91338002, -0.97735382,  0.6       ,  2.02311018, -2.16321053,
         1.0085189 ,  1.73830671],
       [ 1.7932191 , -1.03976983,  0.5       ,  2.02311018, -2.21449075,
         0.84548711,  1.88281559],
       [ 1.67574177, -1.08946411,  0.4       ,  2.02311018, -2.30553762,
         0.69762048,  2.0569663 ],
       [ 1.56004258, -1.12730671,  0.3       ,  2.02311018, -2.45498756,
         0.56962218,  2.28270213],
       [ 0.99085319, -1.15391791, -0.2       ,  2.02311018, -0.73232284,
        -0.60044153,  0.9098408 ],
       [ 1.56004258, -1.12730671,  0.3       ,  2.02311018,  0.68660509,
        -0.56962218, -0.85889052],
       [ 1.67574177, -1.08946411,  0.4       ,  2.02311018,  0.83605503,
        -0.69762048, -1.08462636],
       [ 0.98566097, -1.15236693, -0.2       ,  2.03233934, -3.86536478,
         0.61047535, -2.2378682 ],
       [ 2.17524582, -0.8034851 ,  0.8       ,  2.03233934, -2.15035065,
         1.38676951,  1.4845399 ],
       [ 2.04175346, -0.89711285,  0.7       ,  2.03233934, -2.14716777,
         1.19433304,  1.61476842],
       [ 1.91575237, -0.97434547,  0.6       ,  2.03233934, -2.17015174,
         1.01666957,  1.74689361],
       [ 1.79426556, -1.03734381,  0.5       ,  2.03233934, -2.22296312,
         0.85314889,  1.89334198],
       [ 1.67568121, -1.08744563,  0.4       ,  2.03233934, -2.31568439,
         0.7053313 ,  2.06936754],
       [ 1.55901234, -1.12557036,  0.3       ,  2.03233934, -2.46639541,
         0.57794147,  2.29645368],
       [ 0.98566097, -1.15236693, -0.2       ,  2.03233934, -0.72377213,
        -0.61047535,  0.90372445],
       [ 1.55901234, -1.12557036,  0.3       ,  2.03233934,  0.67519725,
        -0.57794147, -0.84513898],
       [ 1.67568121, -1.08744563,  0.4       ,  2.03233934,  0.82590826,
        -0.7053313 , -1.07222512],
       [ 1.08601757, -1.12399348, -0.1       ,  1.98216027, -3.67670848,
         0.50586635, -2.48069293],
       [ 2.2422838 , -0.73929453,  0.8       ,  1.98216027, -2.18885146,
         1.4282525 ,  1.40806164],
       [ 2.08923017, -0.84082763,  0.7       ,  1.98216027, -2.17107639,
         1.22303934,  1.55512312],
       [ 1.95014943, -0.92258716,  0.6       ,  1.98216027, -2.18362634,
         1.03654076,  1.69775671],
       [ 1.81876453, -0.9884749 ,  0.5       ,  1.98216027, -2.22804033,
         0.86594147,  1.85167077],
       [ 1.69208629, -1.04051597,  0.4       ,  1.98216027, -2.31387296,
         0.71194491,  2.03383432],
       [ 1.5684208 , -1.07995335,  0.3       ,  1.98216027, -2.45993672,
         0.5789909 ,  2.26761086],
       [ 1.44668278, -1.10760314,  0.2       ,  1.98216027, -2.69263061,
         0.47742308,  2.58252966],
       [ 1.08601757, -1.12399348, -0.1       ,  1.98216027, -0.53511583,
        -0.50586635,  0.66089972],
       [ 1.44668278, -1.10760314,  0.2       ,  1.98216027,  0.44896204,
        -0.47742308, -0.55906299],
       [ 2.0687755 ,  0.2760535 , -0.5       , -0.7156698 , -3.16570512,
         1.53420622,  0.32545303],
       [ 5.18898933, -0.44935039, -0.3       ,  1.17672238, -3.34753525,
        -1.25173407, -2.67065332],
       [ 5.18898933, -0.44935039, -0.3       ,  1.17672238, -0.20594259,
         1.25173407,  0.47093934],
       [ 4.26617914, -0.33212585, -0.3       ,  1.17672238,  0.64725314,
         1.39568075, -0.16769878],
       [ 5.03751866, -0.25019943,  2.6       , -0.7156698 , -2.97097236,
         1.54134808,  0.22815683],
       [ 5.38829682, -0.30193089,  2.7       , -0.7156698 , -3.37919587,
         1.54818631,  0.43162252],
       [ 4.26617914, -0.33212585, -0.3       ,  1.17672238, -2.49433951,
        -1.39568075,  2.97389387],
       [ 2.0687755 ,  0.2760535 , -0.5       , -0.7156698 , -0.02411247,
        -1.53420622, -2.81613962],
       [ 5.38829682, -0.30193089,  2.7       , -0.7156698 , -0.23760322,
        -1.54818631, -2.70997013],
       [ 5.03751866, -0.25019943,  2.6       , -0.7156698 ,  0.17062029,
        -1.54134808, -2.91343582]])


    with self.env:
      self.robot.SetActiveDOFValues(startconfig)
    
    # get the trajectory!
    #traj = self.constrain_birrt_to_goal(goals)
    #return;
    time.sleep(15)
    traj = self.rrt_to_goal(goals)

    with self.env:
      self.robot.SetActiveDOFValues(startconfig)

    traj = self.points_to_traj(traj)
    self.robot.GetController().SetPath(traj)
    self.robot.WaitForController(0)
    self.taskmanip.CloseFingers()


  #TODO
  #######################################################
  # Constrained Bi-Directional RRT
  # Keep the z value of the end effector where it is!
  # find a path from the current configuration to ANY goal in goals
  # goals: list of possible goal configurations
  # RETURN: a trajectory to the goal
  #######################################################
  def constrain_birrt_to_goal(self, goals):
    
    # get target height z value
    self.z_val_orig = self.manip.GetTransform()[2,3] # take z value from transform
    print "target z_val_orig = %s" %self.z_val_orig
    
    # get the current arm configuration
    cur_config = self.robot.GetActiveDOFValues()
    print cur_config
    
    new_config = self.project_z_val_manip(cur_config, self.z_val_orig)
    
    return None

  #######################################################
  # check random target is align with the constraint of that z value
  #######################################################
  def compare_target_z_val(self, target, z_val):
    
    # get the current end-effector's transformation
    with self.env:
      self.robot.SetActiveDOFValues(target)
      cur_transform = self.manip.GetTransform()
    
    # get current configuration's z height
    cur_z_val = self.manip.GetTransform()[2,3] # take z value from transform
    
    # get the new end-effector's transformation to be in the target z_val
    # find difference between current and target height
    diff_z_val = z_val - cur_z_val
    if self.prev_target_diff_z_val == diff_z_val:
      self.count_target_diff_z_val += 1
      if self.count_target_diff_z_val > 5:
        print "Stuck at the same target"
        self.count_target_diff_z_val = 0
        self.prev_target_diff_z_val = 0
        return True
    self.prev_target_diff_z_val = diff_z_val
    #print "target diff z_val %s" %diff_z_val
    
    return False

  #TODO
  #######################################################
  # projects onto the constraint of that z value
  #######################################################
  def project_z_val_manip(self, c, z_val):
    # test
    #z_val = 0.6
    #time.sleep(15)
    
    # set the arm to input configuration
    with self.env:
      self.robot.SetActiveDOFValues(c)
    
    # update current joint information
    q = self.robot.GetActiveDOFValues()
    
    # update current height
    Z_des = z_val
    
    # set small time step
    delta_t = 0.02
    
    # set iteration counter 0
    itr_cnt = 0 
    while 1 :
      # count iteration
      itr_cnt += 1
      
      # calculate Z_cur
      Z_cur = self.manip.GetTransform()[2,3] # take z value from transform
      #print "Z_des %s Z_cur %s, Z_cur-Z_des %s" %(Z_des, Z_cur, np.abs(Z_cur-Z_des))
            
      # calculate dZ_cur/dq = Jxq
      #################################################################
      # calculate Jacobian
      # Compute Jacobian
      j_spatial = self.manip.CalculateJacobian()
      
      # consider only dZ_cur/dq
      dZ_cur = j_spatial[2]
      #################################################################
       
      # calculate gradient
      # df/dq = 2x(Z_cur - Z_des)xdZ_cur/dq
      df = 2*(Z_cur-Z_des)*dZ_cur
      #print "df %s" %df
      
      # calculate next q
      # q = q - df/dq x delta_t
      q = q - df*delta_t
      
      # check collision
      if self.checkForCollisions(q) :
        return None
      
      # set new configuration
      #with self.env:
      #  self.robot.SetActiveDOFValues(q)
      
      # check if gradient is zero
      # if q is zero is done
      if np.linalg.norm(df) < 0.02 :
        #print "projection is done %s" %itr_cnt
        #print "Z_des %s Z_cur %s, Z_cur-Z_des %s" %(Z_des, Z_cur, np.abs(Z_cur-Z_des))
        return q
  
  # Initialze 
  def jacobian_init(self):
    # Store the joint limits for collision checking
    self.limits = self.robot.GetActiveDOFLimits()
    self.lowLimit = self.limits[0]
    self.upLimit = self.limits[1]    
    # Find DOFs for computing collision
    self.activeDOFs = self.robot.GetActiveDOFIndices()
  
  # Check collision
  def checkForCollisions(self, config):
    with self.env:
      self.robot.SetActiveDOFValues(config)
    if min(self.lowLimit[x] <= config[x] for x in range(len(config))): # Check if joint within lower limits
      if min(self.upLimit[x] >= config[x] for x in range(len(config))): # Check if joint within upper limits
        if not self.robot.CheckSelfCollision():  # Check for self collision
          if not self.env.CheckCollision(self.robot):  # Check for collision with environment
            return False
    return True

  #######################################################
  # RRT
  # find a path from the current configuration to ANY goal in goals
  # goals: list of possible goal configurations
  # RETURN: a trajectory to the goal
  #######################################################
  def rrt_to_goal(self, goals):

    # start the timer
    start = time.time()
    
    # get target height z value
    self.z_val_orig = self.manip.GetTransform()[2,3] # take z value from transform
    print "target z_val_orig = %s" %self.z_val_orig
    self.count_target_diff_z_val = 0
    self.prev_target_diff_z_val = 0

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
    nearest_state = ini_state
    parent_dict[tuple(ini_state)] = 'root' # add initial state into the dictionary

    while (1):
      # choose a target state
      target = self.choose_target(parent_dict.keys(), goals_verified)

      # choose the nearest state to the target state
      nearest = self.find_nearest(parent_dict.keys(), target)

      # extend one unit from the nearest state towards the target state
      [milestone, finish, collision] = self.unit_extend(nearest, target)
      #print "1 collision %s" %collision
      if (finish == -1): continue # redundant arget
      if (collision == True): continue # extension causes collision
      parent_dict[tuple(milestone)] = nearest
      nearest = milestone

      # check whether any goal reached after extension
      [cmpresult, goal_index] = self.compare_against_goals(milestone, goals_verified)
      if cmpresult == 1: # a goal reached
        print "RRT: Found solution"
        # return the trajectory
        traj = self.return_trajectory(goals_verified, goal_index, parent_dict, milestone)
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
        [milestone, finish, collision] = self.unit_extend(nearest, target)
        #print "2 collision %s" %collision
        if (collision == True): break
        parent_dict[tuple(milestone)] = nearest
        nearest = list(milestone)

        # check whether any goal reached after extension
        [cmpresult, goal_index] = self.compare_against_goals(milestone, goals_verified)
        if cmpresult == 1:
          print "RRT: Found solution"
          # return the trajectory
          traj = self.return_trajectory(goals_verified, goal_index, parent_dict, milestone)
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

  # choose a target to expand
  # do not handle redundant targets
  def choose_target(self, parent_dict_keys, goals):
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
      return goals[np.random.randint(0, len(goals), 1)] # using min-distance may end up stuck in local minima
      
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
      return [milestone, finish]
    # extend from nearest to target, each dimension by at most MAX_MOVE_AMOUNT
    len_target = len(target)
    for i in range(len_target):
      cmpresult = self.compare_value(nearest[i], target[i])
      if (cmpresult > 0): milestone[i] += MAX_MOVE_AMOUNT
      elif (cmpresult < 0): milestone[i] -= MAX_MOVE_AMOUNT
      if finish == 1:
        if self.compare_value(milestone[i], target[i]) != 0: finish = 0 # set finish to 0 if at least one dimenstion of target not met
    
    # project extended configuration to constraint axis
    # if the random target align with target constraint, ignore this target
    
    local_minima = self.compare_target_z_val(np.asarray(target), self.z_val_orig)
    if local_minima == True:
      finish = 1
      collision = True
      return [milestone, finish, collision]
    # if the projected configuration is in collision, then return collision
    new_milestone = self.project_z_val_manip(milestone, self.z_val_orig)
    if new_milestone == None:
      finish = 1
      collision = True
      return [milestone, finish, collision]
    milestone = new_milestone.tolist()
    
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
    traj = []
    traj.append(goals[goal_index]) # append the goal
    tmpstate = milestone
    traj.append(milestone) # append the milestone (which meets the goal) to avoid failure in dictionary key checking
    # iterate until finding the root
    while (1):
      tmpstate2 = parent_dict[tuple(tmpstate)]
      if (parent_dict.has_key(tuple(tmpstate2))): # check whether tmpstate2 is "root"
        traj.append(tmpstate2)
        tmpstate = tmpstate2
      else: break
    traj.reverse() # before revers(), the trajectory is from goal to initial state
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
        return [result, goal_index]
      elif cmpresult == -1:
        result = -1
        return [result, goal_index]
    return [result, goal_index]

  # compare whether two states are equal with a MAX_MOVE_AMOUNT/2 error tolerance
  # return 0 -> NOT equal; 1 -> equal; -1 -> error
  def compare_state(self, curr_state, goal):
    len_curr_state = len(curr_state)
    len_goal = len(goal)
    if len_curr_state != len_goal: return -1
    for i in range(len_curr_state):
      if curr_state[i] >= goal[i]+MAX_MOVE_AMOUNT/2 or curr_state[i] < goal[i]-MAX_MOVE_AMOUNT/2: return 0
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
  time.sleep(10000) #to keep the openrave window open
  

# Guide for installation and running the simulation

1.  Clone the repo. Name the folder as "catkin_ws" and put it under the home directory.
2.  cd ~/catkin_ws && catkin_make_isolated
3.  . devel_isolated/setup.bash
4.  roslaunch ~/catkin_ws/src/traceback_bringup/launch/gazebo_and_rviz.launch
5.  Open the second terminal.
6.  roslaunch ~/catkin_ws/src/traceback_bringup/launch/test.launch

## Change gazebo world
Under ~/catkin_ws/src/traceback_bringup/launch/gazebo_and_rviz.launch, change the gazebo world as instructed

## Change the variables for experiments
Under ~/catkin_ws/src/traceback_bringup/launch/test.launch, change "test_mode" and "loop_closure_confidence_threshold".

# Prerequisite and installation
A linux environment is needed. It can be a linux machine or Windows WSL2.

## Install gazebo classic
https://classic.gazebosim.org/tutorials?tut=install_ubuntu

1. 
```
curl -sSL http://get.gazebosim.org | sh
```

2. gazebo

## Install Noetic (Ubuntu)
- http://wiki.ros.org/noetic/Installation/Ubuntu

## Installing gazebo_ros_pkgs (ROS 1)
- https://classic.gazebosim.org/tutorials?tut=ros_installing&cat=connect_ros

1. 
```
sudo apt-get install ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control
```

2. Test
```
roscore & rosrun gazebo_ros gazebo

rostopic list
rosservice list
```

3. roscore may not be shutdown, use:
```
killall roscore
```

### Create a ROS Workspace
- http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment

- After creating the workspace, ROS VSCode extension can be used.

1. 
```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin_make_isolated
```

2. To use the packages that I built,
```
source devel/setup.bash
```

3. Check
```
echo $ROS_PACKAGE_PATH
```

### TurtleBot3 Simulation (using Gazebo)
1. Install ROS on Remote PC
```
sudo apt-get install ros-noetic-joy ros-noetic-teleop-twist-joy \
  ros-noetic-teleop-twist-keyboard ros-noetic-laser-proc \
  ros-noetic-rgbd-launch ros-noetic-rosserial-arduino \
  ros-noetic-rosserial-python ros-noetic-rosserial-client \
  ros-noetic-rosserial-msgs ros-noetic-amcl ros-noetic-map-server \
  ros-noetic-move-base ros-noetic-urdf ros-noetic-xacro \
  ros-noetic-compressed-image-transport ros-noetic-rqt* ros-noetic-rviz \
  ros-noetic-gmapping ros-noetic-navigation ros-noetic-interactive-markers
```

2. Install Dependent ROS Packages
```
sudo apt install ros-noetic-dynamixel-sdk
sudo apt install ros-noetic-turtlebot3-msgs
sudo apt install ros-noetic-turtlebot3
```

3. 
```
cd ~/catkin_ws/src/
git clone -b noetic-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
cd ~/catkin_ws && catkin_make_isolated
```

4. Since I will be using waffle_pi all the time, open `~/.bashrc`
```
export TURTLEBOT3_MODEL=waffle_pi
```

5. Test
```
roslaunch turtlebot3_gazebo turtlebot3_house.launch
```

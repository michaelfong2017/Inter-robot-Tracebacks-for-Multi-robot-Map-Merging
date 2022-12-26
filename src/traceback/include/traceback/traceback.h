#ifndef TRACEBACK_H_
#define TRACEBACK_H_

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Transform.h>
#include <map_msgs/OccupancyGridUpdate.h>
#include <nav_msgs/OccupancyGrid.h>
#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>

#include <forward_list>
#include <unordered_map>

namespace traceback
{
  struct MapSubscription
  {
    nav_msgs::OccupancyGrid::ConstPtr readonly_map;
    ros::Subscriber map_sub;
    ros::Subscriber map_updates_sub;
    bool is_self;
  };

  class Traceback
  {
  public:
    Traceback();
    
    void spin();

  private:
    ros::NodeHandle node_;

    /* parameters */
    double discovery_rate_;
    double estimation_rate_;
    std::string robot_map_topic_;
    std::string robot_map_updates_topic_;
    std::string robot_namespace_;

    // maps robots namespaces to maps. does not own
    std::unordered_map<std::string, MapSubscription *> robots_;
    // owns maps -- iterator safe
    std::forward_list<MapSubscription> subscriptions_;
    size_t subscriptions_size_;
    boost::shared_mutex subscriptions_mutex_;

    void poseEstimation();

    void fullMapUpdate(const nav_msgs::OccupancyGrid::ConstPtr &msg,
                       MapSubscription &map);

    void topicSubscribing();

    std::string robotNameFromTopic(const std::string &topic);
    bool isRobotMapTopic(const ros::master::TopicInfo &topic);

    void executetopicSubscribing();
    void executeposeEstimation();
  };
} // namespace traceback
#endif
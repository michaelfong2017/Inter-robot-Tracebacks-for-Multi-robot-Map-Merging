#ifndef TRACEBACK_H_
#define TRACEBACK_H_

#include <traceback/transform_estimator.h>
#include <traceback/camera_image_processor.h>

#include <move_base_msgs/MoveBaseAction.h>

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Transform.h>
#include <sensor_msgs/Image.h>
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
    std::string robot_namespace; // e.g /tb3_0
  };

  struct CameraSubscription
  {
    sensor_msgs::ImageConstPtr readonly_camera_image;
    ros::Subscriber camera_sub;
    std::string robot_namespace; // e.g /tb3_0
  };

  class Traceback
  {
  public:
    Traceback();

    void spin();

  private:
    ros::NodeHandle node_;

    /* parameters */
    double update_target_rate_;
    double discovery_rate_;
    double estimation_rate_;
    double confidence_threshold_;
    std::string robot_map_topic_;
    std::string robot_map_updates_topic_;
    std::string robot_namespace_;

    std::string robot_camera_image_topic_;
    double camera_image_update_rate_;

    const tf::TransformListener tf_listener_; ///< @brief Used for transforming
    double transform_tolerance_;              ///< timeout before transform errors

    // maps robots namespaces to maps. does not own
    std::unordered_map<std::string, MapSubscription *> robots_to_map_subscriptions_;
    // owns maps -- iterator safe
    std::forward_list<MapSubscription> map_subscriptions_;
    size_t map_subscriptions_size_;
    boost::shared_mutex map_subscriptions_mutex_;

    std::unordered_map<std::string, CameraSubscription *> robots_to_camera_subscriptions_;
    std::forward_list<CameraSubscription> camera_subscriptions_;
    size_t camera_subscriptions_size_;
    boost::shared_mutex camera_subscriptions_mutex_;

    TransformEstimator transform_estimator_;
    // maps transform indexes to robots namespaces
    std::unordered_map<size_t, std::string> transforms_indexes_;
    std::vector<float> resolutions_; // e.g. ~0.05

    CameraImageProcessor camera_image_processor_;

    std::unordered_map<std::string, ros::Publisher> robots_to_goal_publisher_;
    std::string traceback_goal_topic_ = "traceback/goal";

    void updateTargetPoses();

    geometry_msgs::Pose getRobotPose(const std::string robot_name);
    geometry_msgs::Pose getRobotPose(const std::string &global_frame, const std::string &robot_base_frame, const tf::TransformListener &tf_listener, const double &transform_tolerance);

    void receiveUpdatedCameraImage();

    void poseEstimation();

    void CameraImageUpdate(const sensor_msgs::ImageConstPtr &msg,
                           CameraSubscription &subscription);
    void fullMapUpdate(const nav_msgs::OccupancyGrid::ConstPtr &msg,
                       MapSubscription &subscription);

    void topicSubscribing();

    std::string robotNameFromTopic(const std::string &topic);
    bool isRobotMapTopic(const ros::master::TopicInfo &topic);
    bool isRobotCameraTopic(const ros::master::TopicInfo &topic);

    void executeUpdateTargetPoses();
    void executeTopicSubscribing();
    void executeReceiveUpdatedCameraImage();
    void executePoseEstimation();
  };
} // namespace traceback
#endif
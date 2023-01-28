#ifndef TRACEBACK_H_
#define TRACEBACK_H_

#include <traceback_msgs/GoalAndImage.h>
#include <traceback_msgs/ImageAndImage.h>
#include <traceback_msgs/TracebackTransforms.h>

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

  struct AcceptRejectStatus
  {
    int accept_count;
    int reject_count;
    bool accepted;
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
    double essential_mat_confidence_threshold_;
    int accept_count_needed_;
    int reject_count_needed_;
    int consecutive_abort_count_needed_;
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

    std::unordered_map<std::string, ros::Publisher> robots_to_goal_and_image_publisher_;
    std::string traceback_goal_and_image_topic_ = "traceback/goal_and_image";
    std::unordered_map<std::string, ros::Subscriber> robots_to_image_and_image_subscriber_;
    std::string traceback_image_and_image_topic_ = "traceback/image_and_image";

    std::unordered_map<std::string, bool> robots_to_in_traceback;
    std::unordered_map<std::string, std::list<PoseImagePair>::iterator> robots_to_current_it;

    std::unordered_map<std::string, std::vector<std::vector<cv::Mat>>> robots_src_to_current_transforms_vectors_;

    ros::Publisher traceback_transforms_publisher_;
    std::string traceback_transforms_topic_ = "/traceback/traceback_transforms";

    // map[tracer_robot][traced_robot]
    // Ordered, meaning that map["/tb3_0"]["/tb3_1"] and map["/tb3_1"]["/tb3_0"] are recorded differently.
    std::unordered_map<std::string, std::unordered_map<std::string, AcceptRejectStatus>> pairwise_accept_reject_status_;
    // Pause aborted ordered pair of traceback to prevent being stuck in local optimums
    std::unordered_map<std::string, std::unordered_map<std::string, int>> pairwise_abort_;
    std::unordered_map<std::string, std::unordered_map<std::string, bool>> pairwise_paused_;
    std::unordered_map<std::string, std::unordered_map<std::string, ros::Timer>> pairwise_resume_timer_;

    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<TransformNeeded>>> pairwise_transform_needed_history_;

    std::unordered_map<std::string, std::unordered_map<std::string, cv::Mat>> best_transforms_;
    std::unordered_set<std::string> has_best_transforms_;

    void tracebackImageAndImageUpdate(const traceback_msgs::ImageAndImage::ConstPtr &msg);

    void updateTargetPoses();

    void startOrContinueTraceback(std::string robot_name_src, std::string robot_name_dst);

    std::unordered_map<std::string, ros::Publisher> robots_to_visualize_marker_publisher_;
    std::string visualize_goal_topic_ = "traceback/visualize/goal";
    void visualizeGoal(geometry_msgs::PoseStamped pose_stamped, std::string robot_name); // robot_name is e.g. /tb3_0

    geometry_msgs::Pose getRobotPose(std::string robot_name);
    geometry_msgs::Pose getRobotPose(const std::string &global_frame, const std::string &robot_base_frame, const tf::TransformListener &tf_listener, const double &transform_tolerance);

    void receiveUpdatedCameraImage();

    void poseEstimation();

    void CameraImageUpdate(const sensor_msgs::ImageConstPtr &msg,
                           CameraSubscription &subscription);
    void fullMapUpdate(const nav_msgs::OccupancyGrid::ConstPtr &msg,
                       MapSubscription &subscription);

    void topicSubscribing();

    void matToQuaternion(cv::Mat &mat, geometry_msgs::Quaternion &q);

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
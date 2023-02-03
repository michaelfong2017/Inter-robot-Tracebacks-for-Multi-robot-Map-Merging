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
#include <sensor_msgs/PointCloud2.h>
#include <map_msgs/OccupancyGridUpdate.h>
#include <nav_msgs/OccupancyGrid.h>
#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

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
    // TODO synchronize image and pcl
    sensor_msgs::PointCloud2ConstPtr point_cloud;
    ros::Subscriber camera_rgb_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> camera_point_cloud_sub;
    std::string robot_namespace; // e.g /tb3_0
  };

  struct AcceptRejectStatus
  {
    int accept_count;
    int reject_count;
    bool accepted;
  };

  struct FirstTracebackResult
  {
    double first_x;
    double first_y;
    double first_tracer_to_traced_tx;
    double first_tracer_to_traced_ty;
    double first_tracer_to_traced_r;
  };

  struct TriangulationResult
  {
    // Original, just for debug
    cv::Mat world_transform;
    // Original transform adjusted by triangulation result
    cv::Mat adjusted_transform;
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
    std::string robot_camera_point_cloud_topic_;
    double camera_image_update_rate_;

    const tf::TransformListener tf_listener_; ///< @brief Used for transforming
    double transform_tolerance_;              ///< timeout before transform errors

    // maps robots namespaces to maps. does not own
    std::unordered_map<std::string, MapSubscription *> robots_to_map_subscriptions_;
    // owns maps -- iterator safe
    std::forward_list<MapSubscription> map_subscriptions_;
    size_t map_subscriptions_size_;
    boost::shared_mutex map_subscriptions_mutex_;
    std::vector<cv::Point2d> map_origins_;

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

    std::unordered_map<std::string, bool> robots_to_in_traceback_;
    std::unordered_map<std::string, std::list<PoseImagePair>::iterator> robots_to_current_it_;

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

    std::unordered_map<std::string, std::unordered_map<std::string, FirstTracebackResult>> pairwise_first_traceback_result_;
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<TriangulationResult>>> pairwise_triangulation_result_history_;

    std::unordered_map<std::string, std::unordered_map<std::string, cv::Mat>> best_transforms_;
    std::unordered_set<std::string> has_best_transforms_;

    void tracebackImageAndImageUpdate(const traceback_msgs::ImageAndImage::ConstPtr &msg);

    void continueTraceback(std::string tracer_robot, std::string traced_robot, double src_map_origin_x, double src_map_origin_y, double dst_map_origin_x, double dst_map_origin_y);

    void updateTargetPoses();

    void startOrContinueTraceback(std::string robot_name_src, std::string robot_name_dst, double src_map_origin_x, double src_map_origin_y, double dst_map_origin_x, double dst_map_origin_y);

    std::unordered_map<std::string, ros::Publisher> robots_to_visualize_marker_publisher_;
    std::string visualize_goal_topic_ = "traceback/visualize/goal";
    void visualizeGoal(geometry_msgs::PoseStamped pose_stamped, std::string robot_name); // robot_name is e.g. /tb3_0

    geometry_msgs::Pose getRobotPose(std::string robot_name);
    geometry_msgs::Pose getRobotPose(const std::string &global_frame, const std::string &robot_base_frame, const tf::TransformListener &tf_listener, const double &transform_tolerance);

    void receiveUpdatedCameraImage();

    void poseEstimation();

    void CameraPointCloudUpdate(const sensor_msgs::PointCloud2ConstPtr &msg);

    void CameraImageUpdate(const sensor_msgs::ImageConstPtr &msg,
                           CameraSubscription &subscription);
    void fullMapUpdate(const nav_msgs::OccupancyGrid::ConstPtr &msg,
                       MapSubscription &subscription);

    void topicSubscribing();

    void matToQuaternion(cv::Mat &mat, geometry_msgs::Quaternion &q);

    void imageTransformToMapTransform(cv::Mat &image, cv::Mat &map, float src_resolution, float dst_resolution, double src_map_origin_x, double src_map_origin_y, double dst_map_origin_x, double dst_map_origin_y);

    double findLengthOfTranslationByTriangulation(double first_x, double first_y, double first_tracer_to_traced_tx, double first_tracer_to_traced_ty, double second_x, double second_y, double second_tracer_to_traced_tx, double second_tracer_to_traced_ty);

    void findAdjustedTransformation(cv::Mat &original, cv::Mat &adjusted, double scale, double first_tracer_to_traced_tx, double first_tracer_to_traced_ty, double transform_needed_r, double first_x, double first_y, float src_resolution);

    // Traceback feedbacks can be
    // 1. first traceback, abort with enough consecutive count      -> Exit traceback process, cooldown
    // 2. first traceback, abort without enough consecutive count   -> next goal first traceback
    // 3. first traceback, match                                    -> same goal second traceback, first half triangulation
    // 4. first traceback, reject                                   -> Exit traceback process
    // 5. first traceback, does not match but not yet reject        -> next goal first traceback
    // 6. second traceback, abort                                   -> next goal first traceback
    // 7. second traceback, accept                                  -> Exit traceback process, combine all triangulation results
    // 8. second traceback, match but not yet aceept                -> next goal first traceback, push this triangulation result
    // 9. second traceback, does not match                          -> next goal first traceback
    // 10. first traceback, match but unwanted translation angle    -> next goal first traceback, do not increment reject count
    void writeTracebackFeedbackHistory(std::string tracer, std::string traced, std::string feedback);

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
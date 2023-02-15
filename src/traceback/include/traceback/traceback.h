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
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

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
    sensor_msgs::PointCloud2ConstPtr point_cloud;
    message_filters::Subscriber<sensor_msgs::Image> camera_rgb_sub;
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

  struct TransformAdjustmentResult
  {
    // For debug
    std::string current_time;
    // For debug
    TransformNeeded transform_needed;
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
    std::string test_mode_;
    std::string estimation_mode_;
    std::string adjustment_mode_;
    double update_target_rate_;
    double discovery_rate_;
    double estimation_rate_;
    double confidence_threshold_;
    double unreasonable_goal_distance_;
    double essential_mat_confidence_threshold_;
    double point_cloud_match_score_;
    double point_cloud_close_score_;
    double point_cloud_close_translation_;
    double point_cloud_close_rotation_;
    int accept_count_needed_;
    int reject_count_needed_;
    int abort_count_needed_;
    std::string robot_map_topic_;
    std::string robot_map_updates_topic_;
    std::string robot_namespace_;

    std::string robot_camera_image_topic_;
    std::string robot_camera_point_cloud_topic_;
    int check_obstacle_nearby_pixel_distance_;
    double abort_threshold_distance_;
    double camera_image_update_rate_;
    double data_push_rate_;
    int camera_pose_image_queue_skip_count_;
    int camera_pose_image_max_queue_size_;

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

    std::unordered_map<std::string, std::vector<cv::detail::ImageFeatures>> robots_to_image_features_;
    std::unordered_map<std::string, std::vector<geometry_msgs::Pose>> robots_to_poses_;
    void modifyTransformsBasedOnOrigins(std::vector<cv::Mat> &transforms,
                                        std::vector<cv::Mat> &out,
                                        std::vector<cv::Point2d> &map_origins,
                                        std::vector<float> &resolutions);

    std::unordered_map<std::string, ros::Publisher> robots_to_goal_and_image_publisher_;
    std::string traceback_goal_and_image_topic_ = "traceback/goal_and_image";
    std::unordered_map<std::string, ros::Subscriber> robots_to_image_and_image_subscriber_;
    std::string traceback_image_and_image_topic_ = "traceback/image_and_image";

    std::unordered_map<std::string, bool> robots_to_in_traceback_;
    std::unordered_map<std::string, size_t> robots_to_current_it_;
    std::unordered_map<std::string, boost::shared_mutex> robots_to_current_it_mutex_;

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
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<TransformAdjustmentResult>>> pairwise_triangulation_result_history_;

    // For point cloud mode to store the arrived pose of the first traceback
    // so that transform needed is later computed from the last close enough arrived pose
    // and the first arrived pose.
    std::unordered_map<std::string, std::unordered_map<std::string, geometry_msgs::Pose>> pairwise_first_traceback_arrived_pose_;
    // 1 means first traceback, 2 means first sub-traceback, etc.
    std::unordered_map<std::string, std::unordered_map<std::string, size_t>> pairwise_sub_traceback_number_;

    std::unordered_map<std::string, std::unordered_map<std::string, cv::Mat>> best_transforms_;
    std::unordered_set<std::string> has_best_transforms_;

    void tracebackImageAndImageUpdate(const traceback_msgs::ImageAndImage::ConstPtr &msg);

    void continueTraceback(std::string tracer_robot, std::string traced_robot, double src_map_origin_x, double src_map_origin_y, double dst_map_origin_x, double dst_map_origin_y, bool is_middle_abort = false);

    void updateTargetPoses();

    size_t findMinIndex(std::vector<PoseImagePair> &pose_image_pairs, double threshold_distance, std::string robot_name_dst, cv::Mat pose_dst);

    bool hasObstacleNearby(MapSubscription &subscription, int distance);

    void startOrContinueTraceback(std::string robot_name_src, std::string robot_name_dst, double src_map_origin_x, double src_map_origin_y, double dst_map_origin_x, double dst_map_origin_y);

    std::unordered_map<std::string, ros::Publisher> robots_to_visualize_marker_publisher_;
    std::string visualize_goal_topic_ = "traceback/visualize/goal";
    void visualizeGoal(geometry_msgs::PoseStamped pose_stamped, std::string robot_name, bool is_src = true); // robot_name is e.g. /tb3_0

    geometry_msgs::Pose getRobotPose(std::string robot_name);
    geometry_msgs::Pose getRobotPose(const std::string &global_frame, const std::string &robot_base_frame, const tf::TransformListener &tf_listener, const double &transform_tolerance);

    void receiveUpdatedCameraImage();
    void pushData();

    void poseEstimation();

    // Do not really synchronize anyway
    void CameraImageUpdate(const sensor_msgs::ImageConstPtr &msg);
    void CameraPointCloudUpdate(const sensor_msgs::PointCloud2ConstPtr &msg);

    void fullMapUpdate(const nav_msgs::OccupancyGrid::ConstPtr &msg,
                       MapSubscription &subscription);

    void topicSubscribing();

    void matToQuaternion(cv::Mat &mat, geometry_msgs::Quaternion &q);

    double quaternionToYaw(geometry_msgs::Quaternion &q);

    void imageTransformToMapTransform(cv::Mat &image, cv::Mat &map, float src_resolution, float dst_resolution, double src_map_origin_x, double src_map_origin_y, double dst_map_origin_x, double dst_map_origin_y);

    double findLengthOfTranslationByTriangulation(double first_x, double first_y, double first_tracer_to_traced_tx, double first_tracer_to_traced_ty, double second_x, double second_y, double second_tracer_to_traced_tx, double second_tracer_to_traced_ty);

    void findAdjustedTransformation(cv::Mat &original, cv::Mat &adjusted, double scale, double first_tracer_to_traced_tx, double first_tracer_to_traced_ty, double transform_needed_r, double first_x, double first_y, float src_resolution);

    // Function without parameters.
    // Currently only test with tb3_0, tb3_1 and tb3_2.
    // Their initial poses (x, y, r) are        (-7, 1, 0), (7, 1, 0) and (0.5, 3, 0.785) in global coordinates.
    // So, the global origin object (0, 0, 0rad) is in (7, -1, 0), (-7, -1, 0) and (-0.5, -3, -0.785) in their respective map.
    // Hence, the wanted transformation matrix T to transform any point from tb3_0 frame to tb3_1 frame
    // aims to fulfill T * (7, -1, 0) = (-7, -1, 0)
    // In pixel coordinates where the resolution is defined to be 0.05.
    // T * (140, -20, 0) = (-140, -20, 0)
    // (140, -20, 0) for tb3_0, (-140, -20, 0) for tb3_1 and (-10, -60, -0.785) for tb3_2
    // are hardcoded ground truth for evaluation.
    void evaluateWithGroundTruth(cv::Mat &original, cv::Mat &adjusted, std::string tracer_robot, std::string traced_robot, std::string current_time = "");

    std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_map<std::string, size_t>>> pairwise_proposed_count_;
    void collectProposingData(double score, std::string threshold, std::string tracer_robot, std::string traced_robot);

    // The below cases are for pointcloud mode.
    // For matching, need to match both images and point clouds.
    // 1. abort with enough consecutive count      -> Exit traceback process, cooldown
    // 2. abort without enough consecutive count   -> next goal
    // 3. accept (match and close)                 -> Exit traceback process, combine all point cloud matching results
    // 4. match and close but not yet accept       -> next goal, compute and push point cloud matching result
    // 5. match and not close                      -> same goal repeat, always keep the first arrived pose
    // 6. reject                                   -> Exit traceback process
    // 7. does not match but not yet reject        -> next goal
    //
    // The below 10 cases are for triangulation mode.
    // Traceback feedbacks can be
    // 1. first traceback, abort with enough consecutive count      -> Exit traceback process, cooldown
    // 2. first traceback, abort without enough consecutive count   -> next goal first traceback
    // 3. first traceback, match                                    -> same goal second traceback, first half triangulation
    // 4. first traceback, reject                                   -> Exit traceback process
    // 5. first traceback, does not match but not yet reject        -> next goal first traceback
    // 6. second traceback, abort                                   -> next goal first traceback
    // 7. second traceback, accept                                  -> Exit traceback process, combine all triangulation results
    // 8. second traceback, match but not yet accept                -> next goal first traceback, push this triangulation result
    // 9. second traceback, does not match                          -> next goal first traceback
    // 10. first traceback, match but unwanted translation angle    -> next goal first traceback, do not increment reject count
    void writeTracebackFeedbackHistory(std::string tracer, std::string traced, std::string feedback);

    std::string robotNameFromTopic(const std::string &topic);
    bool isRobotMapTopic(const ros::master::TopicInfo &topic);
    bool isRobotCameraTopic(const ros::master::TopicInfo &topic);

    void executeUpdateTargetPoses();
    void executeTopicSubscribing();
    void executeReceiveUpdatedCameraImage();
    void executePushData();
    void executePoseEstimation();
  };
} // namespace traceback
#endif
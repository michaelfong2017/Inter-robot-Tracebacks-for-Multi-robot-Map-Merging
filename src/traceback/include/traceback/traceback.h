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

#include <cv_bridge/cv_bridge.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
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
    sensor_msgs::ImageConstPtr readonly_depth_image;
    message_filters::Subscriber<sensor_msgs::Image> camera_rgb_sub;
    message_filters::Subscriber<sensor_msgs::Image> camera_depth_sub;
    std::string robot_namespace; // e.g /tb3_0
  };

  struct LoopClosureConstraint
  {
    // x-position of "from" robot where the loop closure is found
    double x;
    // y-position of "from" robot where the loop closure is found
    double y;
    // x-translation from "from" robot to "to" robot, in pixels
    double tx;
    // y-translation from "from" robot to "to" robot, in pixels
    double ty;
    // rotation from "from" robot to "to" robot, in radians
    double r;
  };

  struct AcceptRejectStatus
  {
    int accept_count;
    int reject_count;
    bool accepted;
  };

  struct TransformAdjustmentResult
  {
    // For debug
    std::string current_time;
    TransformNeeded transform_needed;
    // Original, just for debug
    cv::Mat world_transform;
    // Original transform adjusted by solvepnp result
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
    double update_target_rate_;
    double discovery_rate_;
    double estimation_rate_;

    double confidence_threshold_;
    double unreasonable_goal_distance_;
    double essential_mat_confidence_threshold_;
    int accept_count_needed_;
    int reject_count_needed_;
    int abort_count_needed_;
    std::string robot_map_topic_;
    std::string robot_map_updates_topic_;
    std::string robot_namespace_;

    std::string robot_camera_image_topic_;
    std::string robot_camera_depth_image_topic_;
    int check_obstacle_nearby_pixel_distance_;
    double abort_threshold_distance_;
    double camera_image_update_rate_;
    double data_push_rate_;
    int camera_pose_image_queue_skip_count_;
    int camera_pose_image_max_queue_size_;

    // For every pair of robots, do LM optimization
    // read loop closure constraints, but not necessary to lock it
    double transform_optimization_rate_;
    int last_total_loop_constraint_count_;

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

    std::unordered_map<std::string, std::vector<FeaturesDepthsPose>> robots_to_image_features_depths_pose_;
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

    // "from" must be alphabetically smaller than "to", e.g. from "/tb3_0" to "/tb3_1"
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<LoopClosureConstraint>>> robot_to_robot_traceback_loop_closure_constraints_;
    // "from" must be alphabetically smaller than "to", e.g. from "/tb3_0" to "/tb3_1"
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<LoopClosureConstraint>>> robot_to_robot_candidate_loop_closure_constraints_;
    // "from" must be alphabetically smaller than "to", e.g. from "/tb3_0" to "/tb3_1"
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<LoopClosureConstraint>>> robot_to_robot_loop_closure_constraints_;
    boost::shared_mutex loop_constraints_mutex_;
    // "from" must be alphabetically smaller than "to", e.g. from "/tb3_0" to "/tb3_1"
    std::unordered_map<std::string, std::unordered_map<std::string, cv::Mat>> robot_to_robot_optimized_transform_;

    bool readOptimizedTransform(cv::Mat &transform, cv::Mat &inv_transform, std::string from, std::string to);

    // "from" can be alphabetically smaller or greater than "to"
    std::unordered_map<std::string, std::unordered_map<std::string, cv::Mat>> robot_to_robot_traceback_in_progress_transform_;
    // "from" must be alphabetically smaller than "to", e.g. from "/tb3_0" to "/tb3_1"
    std::unordered_map<std::string, std::unordered_map<std::string, size_t>> robot_to_robot_traceback_accept_count_;

    ros::Publisher traceback_transforms_publisher_;
    std::string traceback_transforms_topic_ = "/traceback/traceback_transforms";

    // map[tracer_robot][traced_robot]
    // Ordered, meaning that map["/tb3_0"]["/tb3_1"] and map["/tb3_1"]["/tb3_0"] are recorded differently.
    std::unordered_map<std::string, std::unordered_map<std::string, AcceptRejectStatus>> pairwise_accept_reject_status_;
    // Pause aborted ordered pair of traceback to prevent being stuck in local optimums
    std::unordered_map<std::string, std::unordered_map<std::string, int>> pairwise_abort_;
    std::unordered_map<std::string, std::unordered_map<std::string, bool>> pairwise_paused_;
    std::unordered_map<std::string, std::unordered_map<std::string, ros::Timer>> pairwise_resume_timer_;

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

    void addTracebackLoopClosureConstraint(cv::Mat &adjusted_transform, double src_x, double src_y, std::string src_robot, std::string dst_robot);
    void addCandidateLoopClosureConstraint(cv::Mat &adjusted_transform, double src_x, double src_y, std::string src_robot, std::string dst_robot);
    void addLoopClosureConstraint(cv::Mat &adjusted_transform, double src_x, double src_y, std::string src_robot, std::string dst_robot);

    void transformOptimization();

    void poseEstimation();

    // Do not really synchronize anyway
    void CameraImageUpdate(const sensor_msgs::ImageConstPtr &msg);
    void CameraDepthImageUpdate(const sensor_msgs::ImageConstPtr &msg);

    void fullMapUpdate(const nav_msgs::OccupancyGrid::ConstPtr &msg,
                       MapSubscription &subscription);

    void topicSubscribing();

    cv_bridge::CvImageConstPtr sensorImageToCvImagePtr(const sensor_msgs::Image &image);

    void matToQuaternion(cv::Mat &mat, geometry_msgs::Quaternion &q);

    double quaternionToYaw(geometry_msgs::Quaternion &q);

    void imageTransformToMapTransform(cv::Mat &image, cv::Mat &map, float src_resolution, float dst_resolution, double src_map_origin_x, double src_map_origin_y, double dst_map_origin_x, double dst_map_origin_y);

    double findLengthOfTranslationByTriangulation(double first_x, double first_y, double first_tracer_to_traced_tx, double first_tracer_to_traced_ty, double second_x, double second_y, double second_tracer_to_traced_tx, double second_tracer_to_traced_ty);

    void findAdjustedTransformation(cv::Mat &original, cv::Mat &adjusted, double transform_needed_tx, double transform_needed_ty, double transform_needed_r, double arrived_x, double arrived_y, float src_resolution);

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

    cv::Mat evaluateMatch(cv::Mat &proposed, double pose_x, double pose_y, std::string tracer_robot, std::string traced_robot, std::string current_time);

    bool is_first_match_and_collect_ = true;
    std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_map<std::string, size_t>>> pairwise_proposed_count_;
    void collectProposingData(double pose_x, double pose_y, double predicted_pose_x, double predicted_pose_y, double score, std::string threshold, std::string tracer_robot, std::string traced_robot, std::string current_time, bool same_interval);

    // Traceback feedback can be one of the following 6 cases:
    // 1. abort with enough count       -> exit traceback process, cooldown
    // 2. abort without enough count    -> next goal
    // 3. match and solved and accept               -> exit traceback process, optimize transform
    // 4. match and solved but not yet accept       -> next goal, increment accept count
    // 5. match but cannot solved                   -> next goal
    // 6. does not match and reject                 -> exit traceback process
    // 7. does not match but not yet reject         -> next goal, increment reject count
    void writeTracebackFeedbackHistory(std::string tracer, std::string traced, std::string feedback);

    std::string robotNameFromTopic(const std::string &topic);
    bool isRobotMapTopic(const ros::master::TopicInfo &topic);
    bool isRobotCameraTopic(const ros::master::TopicInfo &topic);

    void executeUpdateTargetPoses();
    void executeTopicSubscribing();
    void executeReceiveUpdatedCameraImage();
    void executePushData();
    void executePoseEstimation();
    void executeTransformOptimization();
  };
} // namespace traceback
#endif
#include <traceback/traceback.h>

#include <thread>
#include <algorithm>

namespace traceback
{
  Traceback::Traceback() : subscriptions_size_(0), tf_listener_(ros::Duration(10.0))
  {
    ros::NodeHandle private_nh("~");

    private_nh.param("update_target_rate", update_target_rate_, 0.2);
    private_nh.param("discovery_rate", discovery_rate_, 0.05);
    private_nh.param("estimation_rate", estimation_rate_, 0.5);
    private_nh.param("estimation_confidence", confidence_threshold_, 1.0);
    private_nh.param<std::string>("robot_map_topic", robot_map_topic_, "map");
    private_nh.param<std::string>("robot_map_updates_topic",
                                  robot_map_updates_topic_, "map_updates");
    private_nh.param<std::string>("robot_namespace", robot_namespace_, "");
    // transform tolerance is used for all tf transforms here
    private_nh.param("transform_tolerance", transform_tolerance_, 0.3);
  }

  void Traceback::updateTargetPoses()
  {
    ROS_DEBUG("Update target poses started.");

    std::vector<std::vector<cv::Mat>> transforms_vectors;
    std::vector<cv::Point2f> centers;
    std::vector<std::vector<double>> confidences;
    // Ensure consistency of transforms_vectors_, centers_ and confidences_
    {
      boost::shared_lock<boost::shared_mutex> lock(transform_estimator_.updates_mutex_);
      transforms_vectors = transform_estimator_.getTransformsVectors();
      // transform_estimator_.printTransformsVectors(transforms_vectors);

      centers = transform_estimator_.getCenters();
      for (auto &p : centers)
      {
        ROS_INFO("center = (%f, %f)", p.x, p.y);
      }

      confidences = transform_estimator_.getConfidences();
      transform_estimator_.printConfidences(confidences);
    }

    for (size_t i = 0; i < confidences.size(); ++i)
    {
      auto it = max_element(confidences[i].begin(), confidences[i].end());
      size_t max_position = it - confidences[i].begin();
      double max_confidence = *it;

      ROS_INFO("confidences[%zu] (max_position, max_confidence) = (%zu, %f)", i, max_position, max_confidence);

      // No transform that passes confidence_threshold_ exists
      if (abs(max_confidence - 0.0) < transform_estimator_.ZERO_ERROR)
      {
        continue;
      }

      // Get current pose
      std::string global_frame = ros::names::append(ros::names::append(transforms_indexes_[i], transforms_indexes_[i]), "map");
      std::string robot_base_frame = ros::names::append(transforms_indexes_[i], "base_link");

      std::string tf_error;
      tf_listener_.waitForTransform(global_frame, robot_base_frame, ros::Time(),
                                    ros::Duration(0.1), ros::Duration(0.01),
                                    &tf_error);

      geometry_msgs::Pose pose = getRobotPose(global_frame, robot_base_frame, tf_listener_, transform_tolerance_);

      ROS_INFO("{%s} pose %zu (x, y) = (%f, %f)", transforms_indexes_[i].c_str(), i, pose.position.x, pose.position.y);

      // Transform current pose from src frame to dst frame
      cv::Mat pose_src(3, 1, CV_64F);
      pose_src.at<double>(0, 0) = pose.position.x / resolutions_[i];
      pose_src.at<double>(1, 0) = pose.position.y / resolutions_[i];
      pose_src.at<double>(2, 0) = 1.0;

      cv::Mat pose_dst = transforms_vectors[max_position][i] * pose_src;
      pose_dst.at<double>(0, 0) *= resolutions_[max_position];
      pose_dst.at<double>(1, 0) *= resolutions_[max_position];

      ROS_INFO("transformed pose (x, y) = (%f, %f)", pose_dst.at<double>(0, 0), pose_dst.at<double>(1, 0));
    }
  }

  geometry_msgs::Pose Traceback::getRobotPose(const std::string &global_frame, const std::string &robot_base_frame, const tf::TransformListener &tf_listener, const double &transform_tolerance)
  {
    tf::Stamped<tf::Pose> global_pose;
    global_pose.setIdentity();
    tf::Stamped<tf::Pose> robot_pose;
    robot_pose.setIdentity();
    // /tb3_0/base_link
    robot_pose.frame_id_ = robot_base_frame;
    robot_pose.stamp_ = ros::Time();
    ros::Time current_time =
        ros::Time::now(); // save time for checking tf delay later

    // get the global pose of the robot
    try
    {
      tf_listener.transformPose(global_frame, robot_pose, global_pose);
    }
    catch (tf::LookupException &ex)
    {
      ROS_ERROR_THROTTLE(1.0, "No Transform available Error looking up robot "
                              "pose: %s\n",
                         ex.what());
      return {};
    }
    catch (tf::ConnectivityException &ex)
    {
      ROS_ERROR_THROTTLE(1.0, "Connectivity Error looking up robot pose: %s\n",
                         ex.what());
      return {};
    }
    catch (tf::ExtrapolationException &ex)
    {
      ROS_ERROR_THROTTLE(1.0, "Extrapolation Error looking up robot pose: %s\n",
                         ex.what());
      return {};
    }
    // check global_pose timeout
    if (current_time.toSec() - global_pose.stamp_.toSec() >
        transform_tolerance)
    {
      ROS_WARN_THROTTLE(1.0, "Costmap2DClient transform timeout. Current time: "
                             "%.4f, global_pose stamp: %.4f, tolerance: %.4f",
                        current_time.toSec(), global_pose.stamp_.toSec(),
                        transform_tolerance);
      return {};
    }

    geometry_msgs::PoseStamped msg;
    tf::poseStampedTFToMsg(global_pose, msg);
    return msg.pose;
  }

  void Traceback::poseEstimation()
  {
    ROS_DEBUG("Grid pose estimation started.");
    std::vector<nav_msgs::OccupancyGridConstPtr> grids;
    grids.reserve(subscriptions_size_);
    transforms_indexes_.clear();
    resolutions_.clear();
    {
      boost::shared_lock<boost::shared_mutex> lock(subscriptions_mutex_);
      size_t i = 0;
      for (auto &subscription : subscriptions_)
      {
        // In case the map topic is just subscribed and pose estimation is started before
        // receiving the first map update from that topic, this subscription should
        // be skipped to prevent error.
        if (!subscription.readonly_map) {
          ++i;
          continue;
        }
        grids.push_back(subscription.readonly_map);
        transforms_indexes_.insert({i, subscription.robot_namespace});
        resolutions_.emplace_back(subscription.readonly_map->info.resolution);
        ++i;
      }
    }

    transform_estimator_.feed(grids.begin(), grids.end());
    transform_estimator_.estimateTransforms(FeatureType::ORB,
                                            confidence_threshold_);
  }

  void Traceback::fullMapUpdate(const nav_msgs::OccupancyGrid::ConstPtr &msg,
                                MapSubscription &subscription)
  {
    ROS_DEBUG("received full map update");
    if (subscription.readonly_map &&
        subscription.readonly_map->header.stamp > msg->header.stamp)
    {
      // we have been overrunned by faster update. our work was useless.
      return;
    }

    subscription.readonly_map = msg;
  }
  /*
   * Subcribe to pose and map topics
   */
  void Traceback::topicSubscribing()
  {
    ros::master::V_TopicInfo topic_infos;
    std::string robot_name;
    std::string map_topic;
    std::string map_updates_topic;

    ros::master::getTopics(topic_infos);
    // default msg constructor does no properly initialize quaternion

    for (const auto &topic : topic_infos)
    {
      // we check only map topic
      if (!isRobotMapTopic(topic))
      {
        continue;
      }

      robot_name = robotNameFromTopic(topic.name);
      if (robots_.count(robot_name))
      {
        // we already know this robot
        continue;
      }

      ROS_INFO("adding robot [%s] to system", robot_name.c_str());
      {
        std::lock_guard<boost::shared_mutex> lock(subscriptions_mutex_);
        subscriptions_.emplace_front();
        ++subscriptions_size_;
      }

      // no locking here. robots_ are used only in this procedure
      MapSubscription &subscription = subscriptions_.front();
      robots_.insert({robot_name, &subscription});

      /* subscribe callbacks */
      map_topic = ros::names::append(robot_name, robot_map_topic_);
      // map_updates_topic =
      //     ros::names::append(robot_name, robot_map_updates_topic_);

      subscription.robot_namespace = robot_name;

      ROS_INFO("Subscribing to MAP topic: %s.", map_topic.c_str());
      subscription.map_sub = node_.subscribe<nav_msgs::OccupancyGrid>(
          map_topic, 50,
          [this, &subscription](const nav_msgs::OccupancyGrid::ConstPtr &msg)
          {
            fullMapUpdate(msg, subscription);
          });
      // subscription.is_self = robot_name == ros::this_node::getNamespace();
      // ROS_INFO("subscription.is_self: %s", subscription.is_self ? "true" : "false");

      // ROS_INFO("Subscribing to MAP updates topic: %s.",
      //          map_updates_topic.c_str());
      // subscription.map_updates_sub =
      //     subscription_nh.subscribe<map_msgs::OccupancyGridUpdate>(
      //         map_updates_topic, 50,
      //         [this, &subscription](
      //             const map_msgs::OccupancyGridUpdate::ConstPtr& msg) {
      //           partialMapUpdate(msg, subscription);
      //         });
    }
  }

  std::string Traceback::robotNameFromTopic(const std::string &topic)
  {
    return ros::names::parentNamespace(topic);
  }

  bool Traceback::isRobotMapTopic(const ros::master::TopicInfo &topic)
  {
    /* test whether topic is robot_map_topic_ */
    std::string topic_namespace = ros::names::parentNamespace(topic.name);
    bool is_map_topic =
        ros::names::append(topic_namespace, robot_map_topic_) == topic.name;

    /* test whether topic contains *anywhere* robot namespace */
    auto pos = topic.name.find(robot_namespace_);
    bool contains_robot_namespace = pos != std::string::npos;

    /* we support only occupancy grids as maps */
    bool is_occupancy_grid = topic.datatype == "nav_msgs/OccupancyGrid";

    return is_occupancy_grid && contains_robot_namespace && is_map_topic;
  }

  void Traceback::executeUpdateTargetPoses()
  {
    ros::Rate r(update_target_rate_);
    while (node_.ok())
    {
      updateTargetPoses();
      r.sleep();
    }
  }

  void Traceback::executeTopicSubscribing()
  {
    ros::Rate r(discovery_rate_);
    while (node_.ok())
    {
      topicSubscribing();
      r.sleep();
    }
  }

  void Traceback::executePoseEstimation()
  {
    ros::Rate r(estimation_rate_);
    while (node_.ok())
    {
      poseEstimation();
      r.sleep();
    }
  }

  /*
   * spin()
   */
  void Traceback::spin()
  {
    ros::spinOnce();
    std::thread subscribing_thr([this]()
                                { executeTopicSubscribing(); });
    std::thread estimation_thr([this]()
                               { executePoseEstimation(); });
    std::thread update_target_thr([this]()
                                  { executeUpdateTargetPoses(); });
    ros::spin();
    update_target_thr.join();
    estimation_thr.join();
    subscribing_thr.join();
  }

} // namespace traceback

int main(int argc, char **argv)
{
  ros::init(argc, argv, "traceback");
  if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                     ros::console::levels::Debug))
  {
    ros::console::notifyLoggerLevelsChanged();
  }
  traceback::Traceback traceback;
  traceback.spin();

  return 0;
}
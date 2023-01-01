#include <traceback/traceback.h>

#include <thread>

namespace traceback
{
  Traceback::Traceback() : subscriptions_size_(0)
  {
    ros::NodeHandle private_nh("~");
    std::string frame_id;

    private_nh.param("update_target_rate", update_target_rate_, 0.2);
    private_nh.param("discovery_rate", discovery_rate_, 0.05);
    private_nh.param("estimation_rate", estimation_rate_, 0.5);
    private_nh.param("estimation_confidence", confidence_threshold_, 1.0);
    private_nh.param<std::string>("robot_map_topic", robot_map_topic_, "map");
    private_nh.param<std::string>("robot_map_updates_topic",
                                  robot_map_updates_topic_, "map_updates");
    private_nh.param<std::string>("robot_namespace", robot_namespace_, "");
  }

  void Traceback::updateTargetPoses()
  {
    ROS_DEBUG("Update target poses started.");

    // Ensure consistency of transforms_vectors_, centers_ and confidences_
    {
      boost::shared_lock<boost::shared_mutex> lock(transform_estimator_.updates_mutex_);
      std::vector<std::vector<cv::Mat>> transforms_vectors = transform_estimator_.getTransformsVectors();
      transform_estimator_.printTransformsVectors(transforms_vectors);

      std::vector<cv::Point2f> centers = transform_estimator_.getCenters();
      for (auto &p : centers)
      {
        ROS_INFO("center = (%f, %f)", p.x, p.y);
      }

      std::vector<std::vector<double>> confidences = transform_estimator_.getConfidences();
      transform_estimator_.printConfidences(confidences);
    }
  }

  void Traceback::poseEstimation()
  {
    ROS_DEBUG("Grid pose estimation started.");
    std::vector<nav_msgs::OccupancyGridConstPtr> grids;
    grids.reserve(subscriptions_size_);
    {
      boost::shared_lock<boost::shared_mutex> lock(subscriptions_mutex_);
      size_t i = 0;
      for (auto &subscription : subscriptions_)
      {
        grids.push_back(subscription.readonly_map);
        transforms_indexes_.insert({i, subscription.robot_namespace});
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
      ROS_INFO("Subscribing to MAP topic: %s.", map_topic.c_str());
      subscription.map_sub = node_.subscribe<nav_msgs::OccupancyGrid>(
          map_topic, 50,
          [this, &subscription](const nav_msgs::OccupancyGrid::ConstPtr &msg)
          {
            fullMapUpdate(msg, subscription);
          });
      // subscription.is_self = robot_name == ros::this_node::getNamespace();
      // ROS_INFO("subscription.is_self: %s", subscription.is_self ? "true" : "false");
      subscription.robot_namespace = robot_name;

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
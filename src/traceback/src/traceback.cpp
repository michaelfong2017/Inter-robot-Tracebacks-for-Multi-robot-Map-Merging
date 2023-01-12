#include <traceback/traceback.h>

#include <thread>
#include <algorithm>
#include <regex>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <sensor_msgs/image_encodings.h>

namespace traceback
{
  Traceback::Traceback() : map_subscriptions_size_(0), camera_subscriptions_size_(0), tf_listener_(ros::Duration(10.0))
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

    private_nh.param<std::string>("camera_image_topic", robot_camera_image_topic_, "camera/rgb/image_raw"); // Don't use image_raw
    private_nh.param("camera_image_update_rate", camera_image_update_rate_, 0.1);
  }

  void Traceback::tracebackImageAndImageUpdate(const traceback_msgs::ImageAndImage::ConstPtr &msg)
  {
    ROS_INFO("tracebackImageAndImageUpdate");

    // Mark this min_index as visited so that it will not be repeatedly visited again and again.
    auto all = camera_image_processor_.robots_to_all_visited_pose_image_pair_indexes_.find(msg->robot_name);
    if (all != camera_image_processor_.robots_to_all_visited_pose_image_pair_indexes_.end())
    {
      all->second.emplace(msg->stamp);
    }
    else
    {
      camera_image_processor_.robots_to_all_visited_pose_image_pair_indexes_.insert({msg->robot_name, {}});
    }

    std::string current_time = std::to_string(round(ros::Time::now().toSec() * 100.0) / 100.0);

    cv_bridge::CvImageConstPtr cv_ptr_tracer;
    try
    {
      if (sensor_msgs::image_encodings::isColor(msg->tracer_image.encoding))
        cv_ptr_tracer = cv_bridge::toCvCopy(msg->tracer_image, sensor_msgs::image_encodings::BGR8);
      else
        cv_ptr_tracer = cv_bridge::toCvCopy(msg->tracer_image, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception &e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // Process cv_ptr->image using OpenCV
    ROS_INFO("Process cv_ptr->image using OpenCV");
    cv::imwrite(current_time + "_tracer.png",
                cv_ptr_tracer->image);

    //
    //
    cv_bridge::CvImageConstPtr cv_ptr_traced;
    try
    {
      if (sensor_msgs::image_encodings::isColor(msg->traced_image.encoding))
        cv_ptr_traced = cv_bridge::toCvCopy(msg->traced_image, sensor_msgs::image_encodings::BGR8);
      else
        cv_ptr_traced = cv_bridge::toCvCopy(msg->traced_image, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception &e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // Process cv_ptr->image using OpenCV
    ROS_INFO("Process cv_ptr->image using OpenCV");
    cv::imwrite(current_time + "_traced.png",
                cv_ptr_traced->image);
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
      std::string robot_name_src = transforms_indexes_[i];
      geometry_msgs::Pose pose = getRobotPose(robot_name_src);

      ROS_INFO("{%s} pose %zu (x, y) = (%f, %f)", robot_name_src.c_str(), i, pose.position.x, pose.position.y);

      // Transform current pose from src frame to dst frame
      cv::Mat pose_src(3, 1, CV_64F);
      pose_src.at<double>(0, 0) = pose.position.x / resolutions_[i];
      pose_src.at<double>(1, 0) = pose.position.y / resolutions_[i];
      pose_src.at<double>(2, 0) = 1.0;

      cv::Mat pose_dst = transforms_vectors[max_position][i] * pose_src;
      pose_dst.at<double>(0, 0) *= resolutions_[max_position];
      pose_dst.at<double>(1, 0) *= resolutions_[max_position];

      ROS_INFO("transformed pose (x, y) = (%f, %f)", pose_dst.at<double>(0, 0), pose_dst.at<double>(1, 0));

      std::string robot_name_dst = transforms_indexes_[max_position];
      double min_distance = DBL_MAX;
      size_t min_index = -1;
      size_t index = 0;
      for (auto pair : camera_image_processor_.robots_to_all_pose_image_pairs_[robot_name_dst])
      {
        if (camera_image_processor_.robots_to_all_visited_pose_image_pair_indexes_[robot_name_dst].count(pair.stamp))
        {
          ++index;
          continue;
        }

        double dst_x = pair.pose.position.x;
        double dst_y = pair.pose.position.y;
        double src_x = pose_dst.at<double>(0, 0);
        double src_y = pose_dst.at<double>(1, 0);
        double distance = sqrt(pow(dst_x - src_x, 2) + pow(dst_y - src_y, 2));
        if (distance < min_distance)
        {
          min_distance = distance;
          min_index = index;
        }
        ++index;
      }

      double goal_x = camera_image_processor_.robots_to_all_pose_image_pairs_[robot_name_dst][min_index].pose.position.x;
      double goal_y = camera_image_processor_.robots_to_all_pose_image_pairs_[robot_name_dst][min_index].pose.position.y;

      // Transform goal from dst frame to src (robot i) frame
      cv::Mat goal_dst(3, 1, CV_64F);
      goal_dst.at<double>(0, 0) = goal_x / resolutions_[max_position];
      goal_dst.at<double>(1, 0) = goal_y / resolutions_[max_position];
      goal_dst.at<double>(2, 0) = 1.0;

      cv::Mat goal_src = transforms_vectors[i][max_position] * goal_dst;
      goal_src.at<double>(0, 0) *= resolutions_[i];
      goal_src.at<double>(1, 0) *= resolutions_[i];

      ROS_INFO("transformed goal_src (x, y) = (%f, %f)", goal_src.at<double>(0, 0), goal_src.at<double>(1, 0));

      geometry_msgs::Point target_position;
      target_position.x = goal_src.at<double>(0, 0);
      target_position.y = goal_src.at<double>(1, 0);
      target_position.z = 0.0f;

      // Transform rotation
      // Note that due to scaling, the "rotation matrix" values can exceed 1, and therefore need to normalize it.
      geometry_msgs::Quaternion goal_q = camera_image_processor_.robots_to_all_pose_image_pairs_[robot_name_dst][min_index].pose.orientation;
      geometry_msgs::Quaternion transform_q;
      cv::Mat transform = transforms_vectors[i][max_position];
      double a = transform.at<double>(0, 0);
      double b = transform.at<double>(1, 0);
      double mag = sqrt(a * a + b * b);
      if (mag != 0)
      {
        a /= mag;
        b /= mag;
      }
      if (a > 1)
        a == 0.9999;
      if (a < -0.9999)
        a == -1;
      if (b > 1)
        b == 0.9999;
      if (b < -1)
        b == -0.9999;
      transform_q.w = std::sqrt(2. + 2. * a) * 0.5;
      transform_q.x = 0.;
      transform_q.y = 0.;
      transform_q.z = std::copysign(std::sqrt(2. - 2. * a) * 0.5, b);
      tf2::Quaternion tf2_goal_q;
      tf2_goal_q.setW(goal_q.w);
      tf2_goal_q.setX(goal_q.x);
      tf2_goal_q.setY(goal_q.y);
      tf2_goal_q.setZ(goal_q.z);
      tf2::Quaternion tf2_transform_q;
      tf2_transform_q.setW(transform_q.w);
      tf2_transform_q.setX(transform_q.x);
      tf2_transform_q.setY(transform_q.y);
      tf2_transform_q.setZ(transform_q.z);
      tf2::Quaternion tf2_new_q = tf2_transform_q * tf2_goal_q;

      ROS_INFO("goal_q (x, y, z, w) = (%f, %f, %f, %f)", goal_q.x, goal_q.y, goal_q.z, goal_q.w);
      ROS_INFO("transform_q (x, y, z, w) = (%f, %f, %f, %f)", transform_q.x, transform_q.y, transform_q.z, transform_q.w);
      ROS_INFO("tf2_new_q (x, y, z, w) = (%f, %f, %f, %f)", tf2_new_q.x(), tf2_new_q.y(), tf2_new_q.z(), tf2_new_q.w());

      geometry_msgs::Quaternion new_q;
      new_q.w = tf2_new_q.w();
      new_q.x = tf2_new_q.x();
      new_q.y = tf2_new_q.y();
      new_q.z = tf2_new_q.z();
      // Transform rotation END

      move_base_msgs::MoveBaseGoal goal;
      goal.target_pose.pose.position = target_position;
      goal.target_pose.pose.orientation = new_q;
      goal.target_pose.header.frame_id = robot_name_src + robot_name_src + "/map";
      goal.target_pose.header.stamp = ros::Time::now();

      traceback_msgs::GoalAndImage goal_and_image;
      goal_and_image.goal = goal;
      goal_and_image.image = camera_image_processor_.robots_to_all_pose_image_pairs_[robot_name_dst][min_index].image;
      goal_and_image.stamp = camera_image_processor_.robots_to_all_pose_image_pairs_[robot_name_dst][min_index].stamp;
      ROS_DEBUG("Goal and image to be sent");
      robots_to_goal_and_image_publisher_[robot_name_src].publish(goal_and_image);
    }
  }

  geometry_msgs::Pose Traceback::getRobotPose(const std::string robot_name)
  {
    std::string global_frame = ros::names::append(ros::names::append(robot_name, robot_name), "map");
    std::string robot_base_frame = ros::names::append(robot_name, "base_link");

    std::string tf_error;
    tf_listener_.waitForTransform(global_frame, robot_base_frame, ros::Time(),
                                  ros::Duration(0.1), ros::Duration(0.01),
                                  &tf_error);

    return getRobotPose(global_frame, robot_base_frame, tf_listener_, this->transform_tolerance_);
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

  void Traceback::receiveUpdatedCameraImage()
  {
    ROS_DEBUG("Receive updated camera image started.");
    for (auto current : camera_image_processor_.robots_to_current_image_)
    {
      std::string robot_name = current.first;
      geometry_msgs::Pose pose = camera_image_processor_.robots_to_current_pose_[current.first];
      PoseImagePair pose_image_pair;
      pose_image_pair.pose = pose;
      pose_image_pair.image = current.second;
      pose_image_pair.stamp = ros::Time::now().toNSec();

      auto all = camera_image_processor_.robots_to_all_pose_image_pairs_.find(current.first);
      if (all != camera_image_processor_.robots_to_all_pose_image_pairs_.end())
      {
        all->second.emplace_back(pose_image_pair);
        PoseImagePair latestPair = *std::max_element(camera_image_processor_.robots_to_all_pose_image_pairs_[current.first].begin(), camera_image_processor_.robots_to_all_pose_image_pairs_[current.first].end());
        ROS_INFO("latestPair.stamp: %ld", latestPair.stamp);
      }
      else
      {
        camera_image_processor_.robots_to_all_pose_image_pairs_.insert({current.first, {pose_image_pair}});
      }
    }
  }

  void Traceback::poseEstimation()
  {
    ROS_DEBUG("Grid pose estimation started.");
    std::vector<nav_msgs::OccupancyGridConstPtr> grids;
    grids.reserve(map_subscriptions_size_);
    transforms_indexes_.clear();
    resolutions_.clear();
    {
      boost::shared_lock<boost::shared_mutex> lock(map_subscriptions_mutex_);
      size_t i = 0;
      for (auto &subscription : map_subscriptions_)
      {
        // In case the map topic is just subscribed and pose estimation is started before
        // receiving the first map update from that topic, this subscription should
        // be skipped to prevent error.
        if (!subscription.readonly_map)
        {
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

  void Traceback::CameraImageUpdate(const sensor_msgs::ImageConstPtr &msg, CameraSubscription &subscription)
  {
    // ROS_DEBUG("received camera image update");
    // ROS_DEBUG("from robot %s", subscription.robot_namespace.c_str());
    // cv_bridge::CvImageConstPtr cv_ptr;
    // try
    // {
    //   if (sensor_msgs::image_encodings::isColor(msg->encoding))
    //     cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    //   else
    //     cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
    // }
    // catch (cv_bridge::Exception &e)
    // {
    //   ROS_ERROR("cv_bridge exception: %s", e.what());
    //   return;
    // }

    // Process cv_ptr->image using OpenCV
    // ROS_INFO("Process cv_ptr->image using OpenCV");
    // Insert if not exists, update if exists.
    camera_image_processor_.robots_to_current_image_[subscription.robot_namespace] = *msg;
    camera_image_processor_.robots_to_current_pose_[subscription.robot_namespace] = getRobotPose(subscription.robot_namespace);
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

  // Subcribe to pose and map topics
  void Traceback::topicSubscribing()
  {
    ros::master::V_TopicInfo topic_infos;
    std::string robot_name;

    ros::master::getTopics(topic_infos);
    // default msg constructor does no properly initialize quaternion

    for (const auto &topic : topic_infos)
    {
      if (isRobotMapTopic(topic))
      {
        std::string map_topic;
        std::string map_updates_topic;

        robot_name = robotNameFromTopic(topic.name);
        if (robots_to_map_subscriptions_.count(robot_name))
        {
          // we already know this robot
          continue;
        }

        ROS_INFO("(map topic) adding robot [%s] to system", robot_name.c_str());
        {
          std::lock_guard<boost::shared_mutex> lock(map_subscriptions_mutex_);
          map_subscriptions_.emplace_front();
          ++map_subscriptions_size_;
        }

        // no locking here. robots_to_map_subscriptions_ are used only in this procedure
        MapSubscription &subscription = map_subscriptions_.front();
        robots_to_map_subscriptions_.insert({robot_name, &subscription});

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
      else if (isRobotCameraTopic(topic))
      {
        std::string camera_topic;

        robot_name = robotNameFromTopic(topic.name);
        if (robots_to_camera_subscriptions_.count(robot_name))
        {
          // we already know this robot
          continue;
        }

        ROS_INFO("(camera topic) adding robot [%s] to system", robot_name.c_str());
        {
          std::lock_guard<boost::shared_mutex> lock(camera_subscriptions_mutex_);
          camera_subscriptions_.emplace_front();
          ++camera_subscriptions_size_;
        }

        // no locking here. robots_to_camera_subscriptions_ are used only in this procedure
        CameraSubscription &subscription = camera_subscriptions_.front();
        robots_to_camera_subscriptions_.insert({robot_name, &subscription});

        /* subscribe callbacks */
        camera_topic = ros::names::append(robot_name, robot_camera_image_topic_);

        subscription.robot_namespace = robot_name;

        ROS_INFO("Subscribing to CAMERA topic: %s.", camera_topic.c_str());

        // Insert empty std::vector to the map to prevent future error when accessing the map by robot name.
        auto it = camera_image_processor_.robots_to_all_pose_image_pairs_.find(subscription.robot_namespace);
        if (it == camera_image_processor_.robots_to_all_pose_image_pairs_.end())
        {
          camera_image_processor_.robots_to_all_pose_image_pairs_.insert({subscription.robot_namespace, {}});
        }

        subscription.camera_sub = node_.subscribe<sensor_msgs::Image>(
            camera_topic, 50,
            [this, &subscription](const sensor_msgs::ImageConstPtr &msg)
            {
              CameraImageUpdate(msg, subscription);
            });

        // Create goal publisher for this robot
        robots_to_goal_and_image_publisher_.emplace(robot_name, node_.advertise<traceback_msgs::GoalAndImage>(ros::names::append(robot_name, traceback_goal_and_image_topic_), 10));

        robots_to_image_and_image_subscriber_.emplace(robot_name, node_.subscribe<traceback_msgs::ImageAndImage>(
                                                                      ros::names::append(robot_name, traceback_image_and_image_topic_), 10,
                                                                      [this](const traceback_msgs::ImageAndImage::ConstPtr &msg)
                                                                      {
                                                                        tracebackImageAndImageUpdate(msg);
                                                                      }));
      }
    }
  }

  // Return /tb3_0 for cases such as /tb3_0/map, /tb3_0/camera/rgb/image_raw and /tb3_0/tb3_0/map (this case is not used).
  std::string Traceback::robotNameFromTopic(const std::string &topic)
  {
    std::regex str_expr("(/tb3.*?)/.*");
    std::smatch sm;
    std::regex_match(topic, sm, str_expr);

    if (sm.size() != 2)
    {
      ROS_ERROR("robotNameFromTopic returns unexpected result! Number of regex matches is %zu instead of the expected 2!", sm.size());
      return "";
    }

    return sm[1];
  }

  bool Traceback::isRobotMapTopic(const ros::master::TopicInfo &topic)
  {
    /* test whether topic contains *anywhere* robot namespace */
    auto pos = topic.name.find(robot_namespace_);
    bool contains_robot_namespace = pos != std::string::npos;
    if (!contains_robot_namespace)
    {
      return false;
    }

    /* test whether topic is robot_map_topic_ */
    std::string robot_namespace = robotNameFromTopic(topic.name);
    bool is_map_topic =
        ros::names::append(robot_namespace, robot_map_topic_) == topic.name;
    if (!is_map_topic)
    {
      return false;
    }

    /* we support only occupancy grids as maps */
    bool is_occupancy_grid = topic.datatype == "nav_msgs/OccupancyGrid";

    return is_occupancy_grid;
  }

  bool Traceback::isRobotCameraTopic(const ros::master::TopicInfo &topic)
  {
    /* test whether topic contains *anywhere* robot namespace */
    auto pos = topic.name.find(robot_namespace_);
    bool contains_robot_namespace = pos != std::string::npos;
    if (!contains_robot_namespace)
    {
      return false;
    }

    /* test whether topic is robot_camera_topic_ */
    std::string robot_namespace = robotNameFromTopic(topic.name);
    bool is_camera_topic =
        ros::names::append(robot_namespace, robot_camera_image_topic_) == topic.name;
    if (!is_camera_topic)
    {
      return false;
    }

    /* we support only Image (not CompressedImage) */
    bool is_image = topic.datatype == "sensor_msgs/Image";

    return is_image;
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

  void Traceback::executeReceiveUpdatedCameraImage()
  {
    ros::Rate r(camera_image_update_rate_);
    while (node_.ok())
    {
      receiveUpdatedCameraImage();
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
    std::thread receive_camera_image_thr([this]()
                                         { executeReceiveUpdatedCameraImage(); });
    std::thread estimation_thr([this]()
                               { executePoseEstimation(); });
    std::thread update_target_thr([this]()
                                  { executeUpdateTargetPoses(); });
    ros::spin();
    update_target_thr.join();
    estimation_thr.join();
    receive_camera_image_thr.join();
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
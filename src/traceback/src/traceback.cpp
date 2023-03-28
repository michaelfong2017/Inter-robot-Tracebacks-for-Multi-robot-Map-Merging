#include <traceback/traceback.h>

#include <thread>
#include <algorithm>
#include <regex>
#include <fstream>

#include <image_transport/image_transport.h>
#include <opencv2/imgcodecs.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/Marker.h>
#include <boost/filesystem.hpp>

namespace traceback
{
  Traceback::Traceback() : map_subscriptions_size_(0), camera_subscriptions_size_(0), last_total_loop_constraint_count_(0), result_index_(0), tf_listener_(ros::Duration(10.0))
  {
    ros::NodeHandle private_nh("~");

    private_nh.param<std::string>("test_mode", test_mode_, "normal");
    private_nh.param<std::string>("estimation_mode", estimation_mode_, "image");
    private_nh.param("update_target_rate", update_target_rate_, 0.2);
    private_nh.param("discovery_rate", discovery_rate_, 0.05);
    private_nh.param("estimation_rate", estimation_rate_, 0.5);
    private_nh.param("transform_optimization_rate", transform_optimization_rate_, 0.2);
    private_nh.param("unreasonable_goal_distance", unreasonable_goal_distance_, 5.0);
    private_nh.param("estimation_confidence", confidence_threshold_, 1.0);
    private_nh.param("essential_mat_confidence", essential_mat_confidence_threshold_, 1.0);
    private_nh.param("far_from_accepted_transform_threshold", far_from_accepted_transform_threshold_, 5.0);
    private_nh.param("accept_count_needed", accept_count_needed_, 8);
    private_nh.param("reject_count_needed", reject_count_needed_, 2);
    private_nh.param("abort_count_needed", abort_count_needed_, 3);
    private_nh.param<std::string>("robot_map_topic", robot_map_topic_, "map");
    private_nh.param<std::string>("robot_map_updates_topic",
                                  robot_map_updates_topic_, "map_updates");
    private_nh.param<std::string>("robot_namespace", robot_namespace_, "");
    // transform tolerance is used for all tf transforms here
    private_nh.param("transform_tolerance", transform_tolerance_, 0.3);

    private_nh.param<std::string>("camera_image_topic", robot_camera_image_topic_, "camera/rgb/image_raw"); // Don't use image_raw
    private_nh.param<std::string>("camera_depth_image_topic", robot_camera_depth_image_topic_, "camera/depth/image_raw");
    private_nh.param("check_obstacle_nearby_pixel_distance", check_obstacle_nearby_pixel_distance_, 3);
    private_nh.param("abort_threshold_distance", abort_threshold_distance_, 2.0);
    private_nh.param("camera_image_update_rate", camera_image_update_rate_, 0.2); // Too high update rate can result in "continue traceback looping"
    private_nh.param("data_push_rate", data_push_rate_, 2.0);
    private_nh.param("camera_pose_image_queue_skip_count", camera_pose_image_queue_skip_count_, 20);
    private_nh.param("camera_pose_image_max_queue_size", camera_pose_image_max_queue_size_, 100);
    private_nh.param("features_depths_max_queue_size", features_depths_max_queue_size_, 100);

    // Create directories for debugging
    for (auto &dir_path : {"tb3_0_tb3_1", "tb3_0_tb3_2", "tb3_1_tb3_0", "tb3_1_tb3_2", "tb3_2_tb3_0", "tb3_2_tb3_1"})
    {
      boost::filesystem::path dir(dir_path);

      if (!(boost::filesystem::exists(dir)))
      {
        ROS_DEBUG("Directory %s does not exist", "tb3_0_tb3_1");

        if (boost::filesystem::create_directory(dir))
          ROS_DEBUG("Directory %s is successfully created", "tb3_0_tb3_1");
      }
    }

    {
      std::ofstream fw("_result.csv", std::ofstream::app);
      if (fw.is_open())
      {
        fw << "result_index"
           << ","
           << "timestamp"
           << ","
           << "from_robot"
           << ","
           << "to_robot"
           << ","
           << "x"
           << ","
           << "y"
           << ","
           << "tx"
           << ","
           << "ty"
           << ","
           << "r"
           << ","
           << "match_score"
           << ","
           << "t_error"
           << ","
           << "r_error" << std::endl;
        fw.close();
      }
    }
  }

  void Traceback::tracebackImageAndImageUpdate(const traceback_msgs::ImageAndImage::ConstPtr &msg)
  {
    ROS_INFO("tracebackImageAndImageUpdate");

    std::string tracer_robot = msg->tracer_robot;
    std::string traced_robot = msg->traced_robot;
    double src_map_origin_x = msg->src_map_origin_x;
    double src_map_origin_y = msg->src_map_origin_y;
    double dst_map_origin_x = msg->dst_map_origin_x;
    double dst_map_origin_y = msg->dst_map_origin_y;
    geometry_msgs::Pose arrived_pose = msg->arrived_pose;

    // Mark this min_index as visited so that it will not be repeatedly visited again and again.
    // Also valid for aborted goal
    auto all = camera_image_processor_.robots_to_all_visited_pose_image_pair_indexes_.find(traced_robot);
    if (all != camera_image_processor_.robots_to_all_visited_pose_image_pair_indexes_.end())
    {
      all->second.emplace(msg->stamp);
    }
    else
    {
      camera_image_processor_.robots_to_all_visited_pose_image_pair_indexes_.insert({traced_robot, {}});
      camera_image_processor_.robots_to_all_visited_pose_image_pair_indexes_[traced_robot].emplace(msg->stamp);
    }
    // END

    // Abort is based on the fact that the location cannot be reached
    // 1 or 2
    if (msg->aborted)
    {
      // 1. abort with enough count
      if (++pairwise_abort_[tracer_robot][traced_robot] >= abort_count_needed_ && pairwise_abort_[tracer_robot][traced_robot] >= 2 * pairwise_accept_reject_status_[tracer_robot][traced_robot].accept_count)
      {
        writeTracebackFeedbackHistory(tracer_robot, traced_robot, "1. abort with enough count");

        pairwise_abort_[tracer_robot][traced_robot] = 0;

        // Allow more time for normal exploration to prevent being stuck at local optimums
        pairwise_paused_[tracer_robot][traced_robot] = true;
        pairwise_resume_timer_[tracer_robot][traced_robot] = node_.createTimer(
            ros::Duration(60, 0),
            [this, tracer_robot, traced_robot](const ros::TimerEvent &)
            { pairwise_paused_[tracer_robot][traced_robot] = false; },
            true);

        pairwise_accept_reject_status_[tracer_robot][traced_robot].accept_count = 0;
        pairwise_accept_reject_status_[tracer_robot][traced_robot].reject_count = 0;
        robots_to_in_traceback_[tracer_robot] = false;

        {
          std::string src_robot = tracer_robot < traced_robot ? tracer_robot : traced_robot;
          std::string dst_robot = tracer_robot < traced_robot ? traced_robot : tracer_robot;
          robot_to_robot_traceback_loop_closure_constraints_[src_robot][dst_robot].clear();
        }
        // 1. abort with enough count END
        return;
      }
      // 2. abort without enough count
      else
      {
        writeTracebackFeedbackHistory(tracer_robot, traced_robot, "2. abort without enough count");

        /** just for finding min_it */
        {
          // Get current pose
          geometry_msgs::Pose pose = getRobotPose(tracer_robot);

          size_t tracer_robot_index;
          size_t traced_robot_index;
          for (auto it = transforms_indexes_.begin(); it != transforms_indexes_.end(); ++it)
          {
            if (it->second == tracer_robot)
            {
              tracer_robot_index = it->first;
            }
            else if (it->second == traced_robot)
            {
              traced_robot_index = it->first;
            }
          }

          cv::Mat transform = robot_to_robot_traceback_in_progress_transform_[tracer_robot][traced_robot];

          cv::Mat pose_src(3, 1, CV_64F);
          pose_src.at<double>(0, 0) = pose.position.x / resolutions_[tracer_robot_index];
          pose_src.at<double>(1, 0) = pose.position.y / resolutions_[tracer_robot_index];
          pose_src.at<double>(2, 0) = 1.0;

          cv::Mat pose_dst = transform * pose_src;
          pose_dst.at<double>(0, 0) *= resolutions_[traced_robot_index];
          pose_dst.at<double>(1, 0) *= resolutions_[traced_robot_index];
          // pose_dst.at<double>(0, 0) += src_map_origin_x;
          // pose_dst.at<double>(1, 0) += src_map_origin_y;
          // // Also adjust the difference between the origins
          // pose_dst.at<double>(0, 0) += dst_map_origin_x;
          // pose_dst.at<double>(1, 0) += dst_map_origin_y;
          // pose_dst.at<double>(0, 0) -= src_map_origin_x;
          // pose_dst.at<double>(1, 0) -= src_map_origin_y;

          boost::shared_lock<boost::shared_mutex> lock(robots_to_current_it_mutex_[traced_robot]);
          double threshold_distance = abort_threshold_distance_; // Go to a location at least and minimally threshold_distance far.
          robots_to_current_it_[tracer_robot] = findMinIndex(camera_image_processor_.robots_to_all_pose_image_pairs_[traced_robot], threshold_distance, traced_robot, pose_dst);
        }
        /** just for finding min_it END */

        continueTraceback(tracer_robot, traced_robot, src_map_origin_x, src_map_origin_y, dst_map_origin_x, dst_map_origin_y, true);
        // 2. abort without enough count END
        return;
      }
    }
    // 3 or 4 or 5 or 6 or 7
    else
    {
      // pairwise_abort_[tracer_robot][traced_robot] = 0;

      std::string current_time = std::to_string(round(ros::Time::now().toSec() * 100.0) / 100.0);

      // get cv images
      cv_bridge::CvImageConstPtr cv_ptr_tracer = sensorImageToCvImagePtr(msg->tracer_image);
      cv_bridge::CvImageConstPtr cv_ptr_traced = sensorImageToCvImagePtr(msg->traced_image);
      cv_bridge::CvImageConstPtr cv_ptr_depth_tracer = sensorImageToCvImagePtr(msg->tracer_depth_image);
      cv_bridge::CvImageConstPtr cv_ptr_depth_traced = sensorImageToCvImagePtr(msg->traced_depth_image);
      //

      geometry_msgs::Quaternion goal_q = arrived_pose.orientation;
      double yaw = quaternionToYaw(goal_q);
      TransformNeeded transform_needed;
      transform_needed.arrived_x = arrived_pose.position.x;
      transform_needed.arrived_y = arrived_pose.position.y;

      MatchAndSolveResult result = camera_image_processor_.matchAndSolve(cv_ptr_tracer->image, cv_ptr_traced->image, cv_ptr_depth_tracer->image, cv_ptr_depth_traced->image, FeatureType::SURF, essential_mat_confidence_threshold_, yaw, transform_needed, tracer_robot, traced_robot, current_time);

      {
        std::ofstream fw(tracer_robot.substr(1) + "_" + traced_robot.substr(1) + "/" + current_time + "_transform_needed_" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot.txt", std::ofstream::app);
        if (fw.is_open())
        {
          fw << "transform_needed in global coordinates (x, y, r) = (" << transform_needed.tx << ", " << transform_needed.ty << ", " << transform_needed.r << ")" << std::endl;
          fw.close();
        }
      }

      cv::imwrite(tracer_robot.substr(1) + "_" + traced_robot.substr(1) + "/" + current_time + tracer_robot.substr(1) + "_tracer.png",
                  cv_ptr_tracer->image);
      cv::imwrite(tracer_robot.substr(1) + "_" + traced_robot.substr(1) + "/" + current_time + traced_robot.substr(1) + "_traced.png",
                  cv_ptr_traced->image);
      cv::Mat colorized_depth_tracer;
      cv_ptr_depth_tracer->image.convertTo(colorized_depth_tracer, CV_8U, 255.0 / 5.0);
      cv::cvtColor(colorized_depth_tracer, colorized_depth_tracer, cv::COLOR_GRAY2BGR);
      cv::Mat colorized_depth_traced;
      cv_ptr_depth_traced->image.convertTo(colorized_depth_traced, CV_8U, 255.0 / 5.0);
      cv::cvtColor(colorized_depth_traced, colorized_depth_traced, cv::COLOR_GRAY2BGR);
      cv::imwrite(tracer_robot.substr(1) + "_" + traced_robot.substr(1) + "/" + current_time + tracer_robot.substr(1) + "_tracer_depth.png",
                  colorized_depth_tracer);
      cv::imwrite(tracer_robot.substr(1) + "_" + traced_robot.substr(1) + "/" + current_time + traced_robot.substr(1) + "_traced_depth.png",
                  colorized_depth_traced);

      // 3 or 4 or 5
      if (result.match)
      {
        // 3 or 4
        if (result.solved)
        {
          // Compute loop closure constraint
          size_t tracer_robot_index;
          size_t traced_robot_index;
          for (auto it = transforms_indexes_.begin(); it != transforms_indexes_.end(); ++it)
          {
            if (it->second == tracer_robot)
            {
              tracer_robot_index = it->first;
            }
            else if (it->second == traced_robot)
            {
              traced_robot_index = it->first;
            }
          }

          cv::Mat world_transform = robot_to_robot_traceback_in_progress_transform_[tracer_robot][traced_robot];

          cv::Mat adjusted_transform;
          findAdjustedTransformation(world_transform, adjusted_transform, transform_needed.tx, transform_needed.ty, transform_needed.r, arrived_pose.position.x, arrived_pose.position.y, resolutions_[tracer_robot_index]);

          // TransformAdjustmentResult result;
          // result.current_time = current_time;
          // result.transform_needed = transform_needed;
          // result.world_transform = world_transform;
          // result.adjusted_transform = adjusted_transform;

          addTracebackLoopClosureConstraint(adjusted_transform, transform_needed.arrived_x / resolutions_[tracer_robot_index], transform_needed.arrived_y / resolutions_[tracer_robot_index], tracer_robot, traced_robot);
        }

        // 3. match and solved and accept
        if (++pairwise_accept_reject_status_[tracer_robot][traced_robot].accept_count >= accept_count_needed_)
        {
          if (result.solved)
          {
            writeTracebackFeedbackHistory(tracer_robot, traced_robot, "3. match and solved and accept");
          }
          else
          {
            writeTracebackFeedbackHistory(tracer_robot, traced_robot, "5. match but cannot solved and accept");
          }
          pairwise_accept_reject_status_[tracer_robot][traced_robot].accepted = true;
          pairwise_accept_reject_status_[tracer_robot][traced_robot].accept_count = 0;
          pairwise_accept_reject_status_[tracer_robot][traced_robot].reject_count = 0;
          robots_to_in_traceback_[tracer_robot] = false;

          {
            boost::lock_guard<boost::shared_mutex> lock(loop_constraints_mutex_);

            std::string src_robot = tracer_robot < traced_robot ? tracer_robot : traced_robot;
            std::string dst_robot = tracer_robot < traced_robot ? traced_robot : tracer_robot;
            size_t i = 0;
            for (auto &constraint : robot_to_robot_traceback_loop_closure_constraints_[src_robot][dst_robot])
            {
              if (i == 0)
              {
                robot_to_robot_latest_accepted_loop_closure_constraint_[src_robot][dst_robot] = constraint;
              }
              else
              {
                robot_to_robot_loop_closure_constraints_[src_robot][dst_robot].push_back(constraint);
              }

              // Generate result
              if (i != 0)
              {
                cv::Mat adjusted_transform(3, 3, CV_64F);
                adjusted_transform.at<double>(0, 0) = cos(constraint.r);
                adjusted_transform.at<double>(0, 1) = -sin(constraint.r);
                adjusted_transform.at<double>(0, 2) = constraint.tx;
                adjusted_transform.at<double>(1, 0) = sin(constraint.r);
                adjusted_transform.at<double>(1, 1) = cos(constraint.r);
                adjusted_transform.at<double>(1, 2) = constraint.ty;
                adjusted_transform.at<double>(2, 0) = 0;
                adjusted_transform.at<double>(2, 1) = 0;
                adjusted_transform.at<double>(2, 2) = 1;
                cv::Mat predicted_pose = evaluateMatch(adjusted_transform, arrived_pose.position.x, arrived_pose.position.y, tracer_robot, traced_robot, current_time);

                Result result;
                result.current_time = std::to_string(round(ros::Time::now().toSec() * 100.0) / 100.0);
                if (tracer_robot < traced_robot)
                {
                  result.from_robot = tracer_robot;
                  result.to_robot = traced_robot;
                }
                else
                {
                  result.from_robot = traced_robot;
                  result.to_robot = tracer_robot;
                }
                result.x = constraint.x * 0.05;
                result.y = constraint.y * 0.05;
                result.tx = constraint.tx * 0.05;
                result.ty = constraint.ty * 0.05;
                result.r = constraint.r;
                result.match_score = 99.0;
                result.t_error = sqrt(pow(predicted_pose.at<double>(0, 0) - arrived_pose.position.x, 2) + pow(predicted_pose.at<double>(1, 0) - arrived_pose.position.y, 2));
                double truth_r;
                if (result.from_robot == "/tb3_0" && result.to_robot == "/tb3_1")
                {
                  truth_r = 0.0; // HARDCODE
                }
                else if (result.from_robot == "/tb3_0" && result.to_robot == "/tb3_2")
                {
                  truth_r = -0.785; // HARDCODE
                }
                else if (result.from_robot == "/tb3_1" && result.to_robot == "/tb3_2")
                {
                  truth_r = -0.785; // HARDCODE
                }
                double rot_0_0 = cos(constraint.r) * cos(-1.0 * truth_r) - sin(constraint.r) * sin(-1.0 * truth_r);
                double rot_1_0 = sin(constraint.r) * cos(-1.0 * truth_r) + cos(constraint.r) * sin(-1.0 * truth_r);
                result.r_error = abs(atan2(rot_1_0, rot_0_0));
                appendResultToFile(result);
              }
              ++i;
            }
            robot_to_robot_traceback_loop_closure_constraints_[src_robot][dst_robot].clear();
          }

          if (tracer_robot < traced_robot)
          {
            ++robot_to_robot_traceback_accept_count_[tracer_robot][traced_robot];
          }
          else
          {
            ++robot_to_robot_traceback_accept_count_[traced_robot][tracer_robot];
          }
          // 3. match and solved and accept END
          return;
        }
        // 4. match and solved but not yet accept
        else
        {
          if (result.solved)
          {
            writeTracebackFeedbackHistory(tracer_robot, traced_robot, "4. match and solved but not yet accept");
          }
          else
          {
            writeTracebackFeedbackHistory(tracer_robot, traced_robot, "5. match but cannot solved");
          }
          continueTraceback(tracer_robot, traced_robot, src_map_origin_x, src_map_origin_y, dst_map_origin_x, dst_map_origin_y);
          // 4. match and solved but not yet accept END
          return;
        }
      }
      // 6 or 7
      else
      {
        // 6. does not match and reject
        if (++pairwise_accept_reject_status_[tracer_robot][traced_robot].reject_count >= reject_count_needed_ && pairwise_accept_reject_status_[tracer_robot][traced_robot].reject_count + pairwise_abort_[tracer_robot][traced_robot] >= pairwise_accept_reject_status_[tracer_robot][traced_robot].accept_count)
        {
          writeTracebackFeedbackHistory(tracer_robot, traced_robot, "6. does not match and reject");
          pairwise_accept_reject_status_[tracer_robot][traced_robot].accept_count = 0;
          pairwise_accept_reject_status_[tracer_robot][traced_robot].reject_count = 0;
          robots_to_in_traceback_[tracer_robot] = false;

          {
            std::string src_robot = tracer_robot < traced_robot ? tracer_robot : traced_robot;
            std::string dst_robot = tracer_robot < traced_robot ? traced_robot : tracer_robot;
            robot_to_robot_traceback_loop_closure_constraints_[src_robot][dst_robot].clear();
          }
          // 6. does not match and reject END
          return;
        }
        // 7. does not match but not yet reject
        else
        {
          writeTracebackFeedbackHistory(tracer_robot, traced_robot, "7. does not match but not yet reject");
          continueTraceback(tracer_robot, traced_robot, src_map_origin_x, src_map_origin_y, dst_map_origin_x, dst_map_origin_y);
          // 7. does not match but not yet reject END
          return;
        }
      }
    }
  }

  void Traceback::continueTraceback(std::string tracer_robot, std::string traced_robot, double src_map_origin_x, double src_map_origin_y, double dst_map_origin_x, double dst_map_origin_y, bool is_middle_abort)
  {
    std::string robot_name_src = tracer_robot;
    std::string robot_name_dst = traced_robot;
    ROS_INFO("Continue traceback process for robot %s", robot_name_src.c_str());

    size_t temp = robots_to_current_it_[robot_name_src];

    bool whole_list_visited = false;
    bool pass_end = false;
    // ++temp;
    // if (temp == camera_image_processor_.robots_to_all_pose_image_pairs_[robot_name_dst].end())
    // {
    //   temp = camera_image_processor_.robots_to_all_pose_image_pairs_[robot_name_dst].begin();
    //   pass_end = true;
    // }
    // if (is_middle_abort)
    // {
    //   if (temp + camera_pose_image_queue_skip_count_ >= camera_image_processor_.robots_to_all_pose_image_pairs_[robot_name_dst].size())
    //   {
    //     temp += camera_pose_image_queue_skip_count_ - camera_image_processor_.robots_to_all_pose_image_pairs_[robot_name_dst].size();
    //   }
    //   else
    //   {
    //     temp += camera_pose_image_queue_skip_count_;
    //   }
    // }
    while (camera_image_processor_.robots_to_all_visited_pose_image_pair_indexes_[robot_name_dst].count(camera_image_processor_.robots_to_all_pose_image_pairs_[robot_name_dst][temp].stamp))
    {
      if (++temp == camera_image_processor_.robots_to_all_pose_image_pairs_[robot_name_dst].size())
      {
        temp = 0;

        if (pass_end)
        { // Pass the end() the second time
          whole_list_visited = true;
          break;
        }
        else
        { // Pass the end() the first time
          pass_end = true;
        }
      }
    }

    robots_to_current_it_[robot_name_src] = temp;

    startOrContinueTraceback(robot_name_src, robot_name_dst, src_map_origin_x, src_map_origin_y, dst_map_origin_x, dst_map_origin_y);
  }

  void Traceback::updateTargetPoses()
  {
    ROS_DEBUG("Update target poses started.");

    std::vector<cv::Point2d> map_origins;

    {
      boost::shared_lock<boost::shared_mutex> lock(map_subscriptions_mutex_);
      map_origins = map_origins_;
    }

    size_t number_of_robots = map_origins.size();

    for (size_t i = 0; i < number_of_robots; ++i)
    {
      std::string robot_name_src = transforms_indexes_[i];
      std::string robot_name_dst = "";
      size_t max_position;

      if (robots_to_in_traceback_[robot_name_src])
      {
        continue; // continue to next robot since the current robot is currently in traceback process
      }

      // Select robot_name_dst
      for (size_t j = 0; j < number_of_robots; ++j)
      {
        if (i == j)
        {
          continue;
        }
        std::string dst = transforms_indexes_[j];

        bool hasCandidate = robot_name_src < dst ? robot_to_robot_candidate_loop_closure_constraints_[robot_name_src][dst].size() > 0 : robot_to_robot_candidate_loop_closure_constraints_[dst][robot_name_src].size() > 0;

        if (hasCandidate)
        {
          if (robot_name_dst == "")
          {
            robot_name_dst = dst;
            max_position = j;
          }
          else
          {
            size_t new_count = robot_name_src < dst ? robot_to_robot_traceback_accept_count_[robot_name_src][dst] : robot_to_robot_traceback_accept_count_[dst][robot_name_src];
            size_t old_count = robot_name_src < robot_name_dst ? robot_to_robot_traceback_accept_count_[robot_name_src][robot_name_dst] : robot_to_robot_traceback_accept_count_[robot_name_dst][robot_name_src];
            if (new_count < old_count)
            {
              robot_name_dst = dst;
              max_position = j;
            }
          }
        }
      }

      if (robot_name_dst == "")
      {
        continue;
      }

      {
        std::ofstream fw("Traceback_" + robot_name_src.substr(1) + "_.txt", std::ofstream::app);
        if (fw.is_open())
        {
          std::string current_time = std::to_string(round(ros::Time::now().toSec() * 100.0) / 100.0);
          fw << current_time << " - Candidate loop closure with robot " << robot_name_dst << std::endl;
          fw.close();
        }
      }

      assert(transforms_indexes_[i] == robot_name_src);
      assert(transforms_indexes_[max_position] == robot_name_dst);
      // Select robot_name_dst END

      cv::Mat transform = cv::Mat(3, 3, CV_64F);
      cv::Mat inv_transform = cv::Mat(3, 3, CV_64F);

      LoopClosureConstraint constraint;
      if (robot_name_src < robot_name_dst)
      {
        constraint = robot_to_robot_candidate_loop_closure_constraints_[robot_name_src][robot_name_dst].front();
        LoopClosureConstraint constraint_copy;
        constraint_copy.x = constraint.x;
        constraint_copy.y = constraint.y;
        constraint_copy.tx = constraint.tx;
        constraint_copy.ty = constraint.ty;
        constraint_copy.r = constraint.r;
        robot_to_robot_traceback_loop_closure_constraints_[robot_name_src][robot_name_dst].push_back(constraint_copy);

        transform.at<double>(0, 0) = cos(constraint.r);
        transform.at<double>(0, 1) = -sin(constraint.r);
        transform.at<double>(0, 2) = constraint.tx;
        transform.at<double>(1, 0) = sin(constraint.r);
        transform.at<double>(1, 1) = cos(constraint.r);
        transform.at<double>(1, 2) = constraint.ty;
        transform.at<double>(2, 0) = 0;
        transform.at<double>(2, 1) = 0;
        transform.at<double>(2, 2) = 1;
        inv_transform = transform.inv();
      }
      else
      {
        constraint = robot_to_robot_candidate_loop_closure_constraints_[robot_name_dst][robot_name_src].front();
        LoopClosureConstraint constraint_copy;
        constraint_copy.x = constraint.x;
        constraint_copy.y = constraint.y;
        constraint_copy.tx = constraint.tx;
        constraint_copy.ty = constraint.ty;
        constraint_copy.r = constraint.r;
        robot_to_robot_traceback_loop_closure_constraints_[robot_name_dst][robot_name_src].push_back(constraint_copy);

        inv_transform.at<double>(0, 0) = cos(constraint.r);
        inv_transform.at<double>(0, 1) = -sin(constraint.r);
        inv_transform.at<double>(0, 2) = constraint.tx;
        inv_transform.at<double>(1, 0) = sin(constraint.r);
        inv_transform.at<double>(1, 1) = cos(constraint.r);
        inv_transform.at<double>(1, 2) = constraint.ty;
        inv_transform.at<double>(2, 0) = 0;
        inv_transform.at<double>(2, 1) = 0;
        inv_transform.at<double>(2, 2) = 1;
        transform = inv_transform.inv();
      }

      // One candidate is only consumed by one traceback process
      if (robot_name_src < robot_name_dst)
      {
        robot_to_robot_candidate_loop_closure_constraints_[robot_name_src][robot_name_dst].erase(robot_to_robot_candidate_loop_closure_constraints_[robot_name_src][robot_name_dst].begin());
      }
      else
      {
        robot_to_robot_candidate_loop_closure_constraints_[robot_name_dst][robot_name_src].erase(robot_to_robot_candidate_loop_closure_constraints_[robot_name_dst][robot_name_src].begin());
      }

      // Keep this traceback transform for traceback next goals and later calculating the loop closure constraint correctly
      robot_to_robot_traceback_in_progress_transform_[robot_name_src][robot_name_dst] = transform.clone();

      // Get current pose
      geometry_msgs::Pose pose = getRobotPose(robot_name_src);

      // ROS_INFO("{%s} pose %zu (x, y) = (%f, %f)", robot_name_src.c_str(), i, pose.position.x, pose.position.y);

      // Transform current pose from src frame to dst frame
      // Since OpenCV transform estimation rotates about the top-left corner,
      // which is the bottom-left corner in the world coordinates.
      // However, the (0, 0) of each map is at center rather than at bottom-left corner,
      // making the coordinate transformation by default rotates about the center of the map.
      // Therefore, it is required to manually rotate about the bottom-left corner, which
      // is (-20m, -20m) or (-400px, -400px) when the resolution is 0.05.
      // This is achieved by translating by (20, 20) first, then rotate as usual,
      // then translate by (-20, -20).
      double src_map_origin_x = map_origins[i].x;
      double src_map_origin_y = map_origins[i].y;
      double dst_map_origin_x = map_origins[max_position].x;
      double dst_map_origin_y = map_origins[max_position].y;

      cv::Mat pose_src(3, 1, CV_64F);
      pose_src.at<double>(0, 0) = pose.position.x / resolutions_[i];
      pose_src.at<double>(1, 0) = pose.position.y / resolutions_[i];
      pose_src.at<double>(2, 0) = 1.0;

      cv::Mat pose_dst = transform * pose_src;
      pose_dst.at<double>(0, 0) *= resolutions_[max_position];
      pose_dst.at<double>(1, 0) *= resolutions_[max_position];
      // pose_dst.at<double>(0, 0) += src_map_origin_x;
      // pose_dst.at<double>(1, 0) += src_map_origin_y;
      // // Also adjust the difference between the origins
      // pose_dst.at<double>(0, 0) += dst_map_origin_x;
      // pose_dst.at<double>(1, 0) += dst_map_origin_y;
      // pose_dst.at<double>(0, 0) -= src_map_origin_x;
      // pose_dst.at<double>(1, 0) -= src_map_origin_y;

      // ROS_INFO("transformed pose (x, y) = (%f, %f)", pose_dst.at<double>(0, 0), pose_dst.at<double>(1, 0));

      robots_to_in_traceback_[robot_name_src] = true;

      ROS_INFO("Start traceback process for robot %s", robot_name_src.c_str());
      ROS_INFO("{%s} pose %zu (x, y) = (%f, %f)", robot_name_src.c_str(), i, pose.position.x, pose.position.y);

      ROS_INFO("transforms[%s][%s] (width, height) = (%d, %d)", robot_name_src.c_str(), robot_name_dst.c_str(), transform.cols, transform.rows);

      int width = transform.cols;
      int height = transform.rows;
      std::string s = "";
      for (int y = 0; y < height; y++)
      {
        for (int x = 0; x < width; x++)
        {
          double val = transform.at<double>(y, x);
          if (x == width - 1)
          {
            s += std::to_string(val) + "\n";
          }
          else
          {
            s += std::to_string(val) + ", ";
          }
        }
      }
      ROS_INFO("matrix:\n%s", s.c_str());
      ROS_INFO("transformed pose (x, y) = (%f, %f)", pose_dst.at<double>(0, 0), pose_dst.at<double>(1, 0));

      /** just for finding min_it */
      {
        boost::shared_lock<boost::shared_mutex> lock(robots_to_current_it_mutex_[robot_name_dst]);
        robots_to_current_it_[robot_name_src] = findMinIndex(camera_image_processor_.robots_to_all_pose_image_pairs_[robot_name_dst], 0.0, robot_name_dst, pose_dst);
      }
      /** just for finding min_it END */

      startOrContinueTraceback(robot_name_src, robot_name_dst, src_map_origin_x, src_map_origin_y, dst_map_origin_x, dst_map_origin_y);
    }
  }

  size_t Traceback::findMinIndex(std::vector<PoseImagePair> &pose_image_pairs, double threshold_distance, std::string robot_name_dst, cv::Mat pose_dst)
  {
    size_t min_index;
    double min_distance = DBL_MAX;

    size_t i = 0;
    for (auto it = pose_image_pairs.begin(); it != pose_image_pairs.end(); ++it)
    {
      if (camera_image_processor_.robots_to_all_visited_pose_image_pair_indexes_[robot_name_dst].count(it->stamp))
      {
        ++i;
        continue;
      }

      double dst_x = it->pose.position.x;
      double dst_y = it->pose.position.y;
      double src_x = pose_dst.at<double>(0, 0);
      double src_y = pose_dst.at<double>(1, 0);
      double distance = sqrt(pow(dst_x - src_x, 2) + pow(dst_y - src_y, 2));
      if (distance < min_distance && distance > threshold_distance)
      {
        min_distance = distance;
        min_index = i;
      }

      ++i;
    }

    return min_index;
  }

  bool Traceback::hasObstacleNearby(MapSubscription &subscription, int distance)
  {
    if (!subscription.readonly_map)
    {
      return true;
    }
    std::string robot_name = subscription.robot_namespace;
    // ROS_DEBUG("%s", robot_name.c_str());
    auto current = camera_image_processor_.robots_to_current_pose_.find(robot_name);
    if (current != camera_image_processor_.robots_to_current_pose_.end())
    {
      geometry_msgs::Pose pose = current->second;
      int width = subscription.readonly_map->info.width;
      int height = subscription.readonly_map->info.height;
      double origin_x = subscription.readonly_map->info.origin.position.x;
      double origin_y = subscription.readonly_map->info.origin.position.y;
      float resolution = subscription.readonly_map->info.resolution;

      int cx = (pose.position.x - origin_x) / resolution;
      int cy = (pose.position.y - origin_y) / resolution;

      for (int dy = 0 - distance; dy <= 0 + distance; ++dy)
      {
        int y = cy + dy;
        if (y < 0 || y > height)
        {
          return true;
        }
        for (int dx = 0 - distance; dx <= 0 + distance; ++dx)
        {
          int x = cx + dx;
          if (x < 0 || x > width)
          {
            return true;
          }
          int cell = subscription.readonly_map->data.at(y * width + x);
          // ROS_DEBUG("cell (%d, %d) = %d", x, y, cell);
          if (cell != 0)
          {
            return true;
          }
        }
      }
      return false;
    }
    return true;
  }

  void Traceback::startOrContinueTraceback(std::string robot_name_src, std::string robot_name_dst, double src_map_origin_x, double src_map_origin_y, double dst_map_origin_x, double dst_map_origin_y)
  {
    size_t i;
    for (auto it = transforms_indexes_.begin(); it != transforms_indexes_.end(); ++it)
    {
      if (it->second == robot_name_src)
        i = it->first;
    }

    size_t max_position;
    for (auto it = transforms_indexes_.begin(); it != transforms_indexes_.end(); ++it)
    {
      if (it->second == robot_name_dst)
        max_position = it->first;
    }

    ROS_DEBUG("startOrContinueTraceback i = %zu", i);
    ROS_DEBUG("startOrContinueTraceback max_position = %zu", max_position);

    cv::Mat transform = robot_to_robot_traceback_in_progress_transform_[robot_name_src][robot_name_dst].clone();
    cv::Mat inv_transform = transform.inv();

    PoseImagePair current_it = camera_image_processor_.robots_to_all_pose_image_pairs_[robot_name_dst][robots_to_current_it_[robot_name_src]];

    double goal_x = current_it.pose.position.x;
    double goal_y = current_it.pose.position.y;

    // Transform goal from dst frame to src (robot i) frame
    // Same as above, it is required to manually rotate about the bottom-left corner, which
    // is (-20m, -20m) or (-400px, -400px) when the resolution is 0.05.
    cv::Mat goal_dst(3, 1, CV_64F);
    // Change to not consider map origin
    goal_dst.at<double>(0, 0) = goal_x / resolutions_[max_position];
    goal_dst.at<double>(1, 0) = goal_y / resolutions_[max_position];
    goal_dst.at<double>(2, 0) = 1.0;

    cv::Mat goal_src = inv_transform * goal_dst;
    goal_src.at<double>(0, 0) *= resolutions_[i];
    goal_src.at<double>(1, 0) *= resolutions_[i];
    // goal_src.at<double>(0, 0) += dst_map_origin_x;
    // goal_src.at<double>(1, 0) += dst_map_origin_y;
    // // Also adjust the difference between the origins
    // goal_src.at<double>(0, 0) += src_map_origin_x;
    // goal_src.at<double>(1, 0) += src_map_origin_y;
    // goal_src.at<double>(0, 0) -= dst_map_origin_x;
    // goal_src.at<double>(1, 0) -= dst_map_origin_y;

    ROS_INFO("transformed goal_src (x, y) = (%f, %f)", goal_src.at<double>(0, 0), goal_src.at<double>(1, 0));

    geometry_msgs::Point target_position;
    target_position.x = goal_src.at<double>(0, 0);
    target_position.y = goal_src.at<double>(1, 0);
    target_position.z = 0.0f;

    // abort timeout and distance to goal for reject
    double distance_to_goal;
    int abort_timeout;
    {
      geometry_msgs::Pose pose = getRobotPose(robot_name_src);
      double current_x = pose.position.x;
      double current_y = pose.position.y;
      distance_to_goal = sqrt(pow(current_x - target_position.x, 2) + pow(current_y - target_position.y, 2));
    }
    // Abort timeout allowed based on distance to goal
    // in seconds
    // HARDCODE the formula currently
    abort_timeout = 30 + 5 * distance_to_goal;
    // abort timeout END

    // Reject when the goal is too far only when the traceback process
    // is about to started
    // a bit hacky way to check for this
    bool just_about_to_start = pairwise_accept_reject_status_[robot_name_src][robot_name_dst].accept_count == 0 && pairwise_accept_reject_status_[robot_name_src][robot_name_dst].reject_count == 0 && pairwise_abort_[robot_name_src][robot_name_dst] == 0;
    if (just_about_to_start)
    {
      // S1. reject since the goal is unreasonably far
      if (distance_to_goal >= unreasonable_goal_distance_)
      {
        writeTracebackFeedbackHistory(robot_name_src, robot_name_dst, "S1. reject since the goal is unreasonably far");
        robots_to_in_traceback_[robot_name_src] = false;
        if (robot_name_src < robot_name_dst)
        {
          robot_to_robot_traceback_loop_closure_constraints_[robot_name_src][robot_name_dst].clear();
        }
        else
        {
          robot_to_robot_traceback_loop_closure_constraints_[robot_name_dst][robot_name_src].clear();
        }

        return;
        // S1. reject since the goal is unreasonably far END
      }
    }

    // TODO reject goal if it is sure that it is occupied,
    // may also be more restrictive to disallow obstacles nearby within distance d_goal_near
    // end

    // Transform rotation
    // Note that due to scaling, the "rotation matrix" values can exceed 1, and therefore need to normalize it.
    geometry_msgs::Quaternion goal_q = current_it.pose.orientation;
    geometry_msgs::Quaternion transform_q;
    matToQuaternion(inv_transform, transform_q);
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

    // Rotate by -30 degree
    // tf2_transform_q.setW(0.9659258);
    // tf2_transform_q.setX(0.0);
    // tf2_transform_q.setY(0.0);
    // tf2_transform_q.setZ(-0.258819);
    // tf2_new_q = tf2_transform_q * tf2_new_q;
    //

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
    goal.target_pose.header.frame_id = robot_name_src.substr(1) + robot_name_src + "/map";
    goal.target_pose.header.stamp = ros::Time::now();

    traceback_msgs::GoalAndImage goal_and_image;
    goal_and_image.goal = goal;
    goal_and_image.image = current_it.image;
    goal_and_image.depth_image = current_it.depth_image;
    // goal_and_image.point_cloud = current_it.point_cloud;
    goal_and_image.tracer_robot = robot_name_src;
    goal_and_image.traced_robot = robot_name_dst;
    goal_and_image.src_map_origin_x = src_map_origin_x;
    goal_and_image.src_map_origin_y = src_map_origin_y;
    goal_and_image.dst_map_origin_x = dst_map_origin_x;
    goal_and_image.dst_map_origin_y = dst_map_origin_y;
    goal_and_image.abort_timeout = abort_timeout;
    goal_and_image.stamp = current_it.stamp;
    ROS_DEBUG("Goal and image to be sent");

    /** Visualize goal in dst robot frame */
    {
      geometry_msgs::PoseStamped pose_stamped;
      pose_stamped.pose = current_it.pose;
      pose_stamped.header.frame_id = robot_name_dst.substr(1) + robot_name_dst + "/map";
      pose_stamped.header.stamp = ros::Time::now();
      visualizeGoal(pose_stamped, robot_name_src, false);
    }
    /** Visualize goal in dst robot frame END */

    /** Visualize goal in src robot frame */
    visualizeGoal(goal.target_pose, robot_name_src, true);
    /** Visualize goal in src robot frame END */
    robots_to_goal_and_image_publisher_[robot_name_src].publish(goal_and_image);
  }

  void Traceback::visualizeGoal(geometry_msgs::PoseStamped pose_stamped, std::string robot_name, bool is_src)
  {
    std_msgs::ColorRGBA blue;
    blue.r = 0;
    blue.g = 0;
    blue.b = 1.0;
    blue.a = 1.0;
    std_msgs::ColorRGBA red;
    red.r = 1.0;
    red.g = 0;
    red.b = 0;
    red.a = 1.0;
    std_msgs::ColorRGBA green;
    green.r = 0;
    green.g = 1.0;
    green.b = 0;
    green.a = 1.0;

    visualization_msgs::Marker m;

    m.header.frame_id = pose_stamped.header.frame_id;
    m.header.stamp = pose_stamped.header.stamp;
    if (is_src)
    {
      m.ns = "src_frame";
    }
    else
    {
      m.ns = "dst_frame";
    }
    if (is_src)
    {
      m.scale.x = 0.4;
      m.scale.y = 0.4;
      m.scale.z = 0.4;
    }
    else
    {
      m.scale.x = 0.8;
      m.scale.y = 0.8;
      m.scale.z = 0.8;
    }

    // lives forever
    m.lifetime = ros::Duration(0);
    m.frame_locked = true;

    m.action = visualization_msgs::Marker::ADD;
    m.type = visualization_msgs::Marker::ARROW;
    size_t id = 0;
    m.id = int(id);
    if (robot_name == "/tb3_0")
    {
      m.color = red;
    }
    else if (robot_name == "/tb3_1")
    {
      m.color = green;
    }
    else if (robot_name == "/tb3_2")
    {
      m.color = blue;
    }
    m.pose.position = pose_stamped.pose.position;
    m.pose.orientation = pose_stamped.pose.orientation;
    // delete previous markers, which are now unused
    // m.action = visualization_msgs::Marker::DELETE;
    // for (; id < last_markers_count_; ++id)
    // {
    //   m.id = int(id);
    //   markers.push_back(m);
    // }

    // last_markers_count_ = current_markers_count;
    robots_to_visualize_marker_publisher_[robot_name].publish(m);
  }

  geometry_msgs::Pose Traceback::getRobotPose(std::string robot_name)
  {
    std::string global_frame = ros::names::append(ros::names::append(robot_name.substr(1), robot_name), "map");
    std::string robot_base_frame = ros::names::append(robot_name.substr(1), "base_link");

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
    std::string current_time = std::to_string(round(ros::Time::now().toSec() * 100.0) / 100.0);

    if (is_first_match_and_collect_)
    {
      is_first_match_and_collect_ = false;
      {
        std::ofstream fw(current_time + ".txt", std::ofstream::app);
        if (fw.is_open())
        {
          fw << "Start matching at " << current_time << std::endl;
          fw.close();
        }
      }
    }
    // ROS_DEBUG("Receive updated camera image started.");

    ////
    // return if in "map" mode, proceed if in "image" mode
    ////
    if (estimation_mode_ == "map")
    {
      return;
    }

    for (auto current : camera_image_processor_.robots_to_current_image_)
    {
      std::string robot_name = current.first;

      sensor_msgs::Image depth_image = camera_image_processor_.robots_to_current_depth_image_[robot_name];

      // get cv images
      cv_bridge::CvImageConstPtr cv_ptr = sensorImageToCvImagePtr(current.second);
      cv_bridge::CvImageConstPtr cv_ptr_depth = sensorImageToCvImagePtr(depth_image);
      //

      FeaturesDepthsPose features_depths_pose;
      bool hasFeaturesAndDepths = camera_image_processor_.computeFeaturesAndDepths(features_depths_pose.features, features_depths_pose.depths, cv_ptr->image, FeatureType::ORB, cv_ptr_depth->image);
      if (!hasFeaturesAndDepths)
      {
        continue;
      }
      features_depths_pose.pose = getRobotPose(robot_name);

      if (robots_to_image_features_depths_pose_[robot_name].size() >= features_depths_max_queue_size_)
      {
        robots_to_image_features_depths_pose_[robot_name].erase(robots_to_image_features_depths_pose_[robot_name].begin());
      }

      robots_to_image_features_depths_pose_[robot_name].push_back(features_depths_pose);

      for (auto &pair : robots_to_image_features_depths_pose_)
      {
        std::string second_robot_name = pair.first;
        // Ensure robot_name < second_robot_name so that the same pair will not be done twice
        // Later only need to insert the transform (loop closure constraint) for this pair (no need inv_transform)
        if (robot_name >= second_robot_name)
        {
          continue;
        }

        for (size_t i = 0; i < pair.second.size(); ++i)
        {
          cv::detail::ImageFeatures features1 = robots_to_image_features_depths_pose_[robot_name].back().features;
          cv::detail::ImageFeatures features2 = pair.second[i].features;

          double confidence_output = transform_estimator_.matchTwoFeatures(features1, features2, confidence_threshold_);
          if (confidence_output > 0.0)
          {
            geometry_msgs::Pose pose1 = robots_to_image_features_depths_pose_[robot_name].back().pose;
            geometry_msgs::Pose pose2 = pair.second[i].pose;
            std::vector<double> depths1 = robots_to_image_features_depths_pose_[robot_name].back().depths;
            std::vector<double> depths2 = pair.second[i].depths;
            // ROS_DEBUG("Match!");

            // if (test_mode_ != "collect")
            // {
            {
              std::ofstream fw(robot_name.substr(1) + "_" + second_robot_name.substr(1) + "/" + "Transform_proposed_" + robot_name.substr(1) + "_current_robot_" + second_robot_name.substr(1) + "_target_robot.txt", std::ofstream::app);
              if (fw.is_open())
              {
                fw << "Transform proposed at time " << current_time << " with confidence " << confidence_output << std::endl;
                fw.close();
              }
            }
            // }

            // TEST with ground truth
            // double init_0_x = -7.0;
            // double init_0_y = -1.0;
            // double init_0_r = 0.0;
            // double init_1_x = 7.0;
            // double init_1_y = -1.0;
            // double init_1_r = 0.0;
            // double init_2_x = 0.5;
            // double init_2_y = 3.0;
            // double init_2_r = 0.785;
            // geometry_msgs::Pose init_pose1 = pose1;
            // geometry_msgs::Pose init_pose2 = pose2;
            // if (robot_name == "/tb3_0")
            // {
            //   init_pose1.position.x = init_0_x;
            //   init_pose1.position.y = init_0_y;
            //   init_pose1.position.z = 0.0;
            //   init_pose1.orientation.z = 0.0;
            //   init_pose1.orientation.w = 1.0;
            // }
            // else if (robot_name == "/tb3_1")
            // {
            //   init_pose1.position.x = init_1_x;
            //   init_pose1.position.y = init_1_y;
            //   init_pose1.position.z = 0.0;
            //   init_pose1.orientation.z = 0.0;
            //   init_pose1.orientation.w = 1.0;
            // }
            // else if (robot_name == "/tb3_2")
            // {
            //   init_pose1.position.x = init_2_x;
            //   init_pose1.position.y = init_2_y;
            //   init_pose1.position.z = 0.0;
            //   init_pose1.orientation.z = 0.3826834;
            //   init_pose1.orientation.w = 0.9238795;
            // }
            // if (second_robot_name == "/tb3_0")
            // {
            //   init_pose2.position.x = init_0_x;
            //   init_pose2.position.y = init_0_y;
            //   init_pose2.position.z = 0.0;
            //   init_pose2.orientation.z = 0.0;
            //   init_pose2.orientation.w = 1.0;
            // }
            // else if (second_robot_name == "/tb3_1")
            // {
            //   init_pose2.position.x = init_1_x;
            //   init_pose2.position.y = init_1_y;
            //   init_pose2.position.z = 0.0;
            //   init_pose2.orientation.z = 0.0;
            //   init_pose2.orientation.w = 1.0;
            // }
            // else if (second_robot_name == "/tb3_2")
            // {
            //   init_pose2.position.x = init_2_x;
            //   init_pose2.position.y = init_2_y;
            //   init_pose2.position.z = 0.0;
            //   init_pose2.orientation.z = 0.3826834;
            //   init_pose2.orientation.w = 0.9238795;
            // }
            // TEST with ground truth END

            geometry_msgs::Pose init_pose1;
            init_pose1.position.x = pose1.position.x * -1;
            init_pose1.position.y = pose1.position.y * -1;
            init_pose1.position.z = 0.0;
            init_pose1.orientation.z = pose1.orientation.z * -1;
            init_pose1.orientation.x = 0.0;
            init_pose1.orientation.y = 0.0;
            init_pose1.orientation.w = pose1.orientation.w;
            geometry_msgs::Pose init_pose2;
            init_pose2.position.x = pose2.position.x * -1;
            init_pose2.position.y = pose2.position.y * -1;
            init_pose2.position.z = 0.0;
            init_pose2.orientation.z = pose2.orientation.z * -1;
            init_pose2.orientation.x = 0.0;
            init_pose2.orientation.y = 0.0;
            init_pose2.orientation.w = pose2.orientation.w;

            size_t self_robot_index;
            size_t second_robot_index;
            for (auto it = transforms_indexes_.begin(); it != transforms_indexes_.end(); ++it)
            {
              if (it->second == robot_name)
              {
                self_robot_index = it->first;
              }
              else if (it->second == second_robot_name)
              {
                second_robot_index = it->first;
              }
            }

            std::vector<geometry_msgs::Pose> transforms = {init_pose1, init_pose2};
            std::vector<cv::Mat> current_traceback_transforms;
            for (size_t i = 0; i < transforms.size(); ++i)
            {
              double x = transforms[i].position.x;
              double y = transforms[i].position.y;
              tf2::Quaternion tf_q;
              tf2::fromMsg(transforms[i].orientation, tf_q);
              tf2::Matrix3x3 m(tf_q);
              double roll, pitch, yaw;
              m.getRPY(roll, pitch, yaw);

              size_t index = i == 0 ? self_robot_index : second_robot_index;
              cv::Mat t_global_i(3, 3, CV_64F);
              t_global_i.at<double>(0, 0) = cos(-1 * yaw);
              t_global_i.at<double>(0, 1) = -sin(-1 * yaw);
              t_global_i.at<double>(0, 2) = -1 * x / resolutions_[index];
              t_global_i.at<double>(1, 0) = sin(-1 * yaw);
              t_global_i.at<double>(1, 1) = cos(-1 * yaw);
              t_global_i.at<double>(1, 2) = -1 * y / resolutions_[index];
              t_global_i.at<double>(2, 0) = 0.0;
              t_global_i.at<double>(2, 1) = 0.0;
              t_global_i.at<double>(2, 2) = 1;

              current_traceback_transforms.push_back(t_global_i);
            }

            cv::Mat world_transform = current_traceback_transforms[1] *
                                      current_traceback_transforms[0].inv();

            //
            // Up to now, world_transform is robot->second_robot world transform in pixels.
            //
            std::string current_time = std::to_string(round(ros::Time::now().toSec() * 100.0) / 100.0);

            geometry_msgs::Quaternion goal_q = pose1.orientation;
            double yaw = quaternionToYaw(goal_q);
            TransformNeeded transform_needed;
            transform_needed.arrived_x = pose1.position.x;
            transform_needed.arrived_y = pose1.position.y;
            MatchAndSolveResult result = camera_image_processor_.matchAndSolveWithFeaturesAndDepths(features1, features2, depths1, depths2, essential_mat_confidence_threshold_, yaw, transform_needed, robot_name, second_robot_name, current_time);

            if (!result.solved)
            {
              transform_needed.tx = 0.0;
              transform_needed.ty = 0.0;
              transform_needed.r = 0.0;
            }

            cv::Mat adjusted_transform;
            findAdjustedTransformation(world_transform, adjusted_transform, transform_needed.tx, transform_needed.ty, transform_needed.r, transform_needed.arrived_x, transform_needed.arrived_y, resolutions_[self_robot_index]);

            // std::vector<cv::Point2d> local_origins = {map_origins_[self_robot_index], map_origins_[second_robot_index]};
            // std::vector<float> local_resolutions = {resolutions_[self_robot_index], resolutions_[second_robot_index]};
            // std::vector<cv::Mat> modified_traceback_transforms;
            // modifyTransformsBasedOnOrigins(current_traceback_transforms,
            //                                modified_traceback_transforms,
            //                                local_origins, local_resolutions);

            std::string s = "";
            for (int y = 0; y < 3; y++)
            {
              for (int x = 0; x < 3; x++)
              {
                double val = world_transform.at<double>(y, x);
                if (x == 3 - 1)
                {
                  s += std::to_string(val) + "\n";
                }
                else
                {
                  s += std::to_string(val) + ", ";
                }
              }
            }
            ROS_INFO("matrix:\n%s", s.c_str());

            // Transform proposed here is often outlier
            addCandidateLoopClosureConstraint(adjusted_transform, transform_needed.arrived_x / resolutions_[self_robot_index], transform_needed.arrived_y / resolutions_[self_robot_index], robot_name, second_robot_name);

            addLoopClosureConstraint(adjusted_transform, transform_needed.arrived_x / resolutions_[self_robot_index], transform_needed.arrived_y / resolutions_[self_robot_index], robot_name, second_robot_name);

            // Evaluate match with current pose of current robot
            // Note that unmodified transform is used
            cv::Mat predicted_pose = evaluateMatch(adjusted_transform, pose1.position.x, pose1.position.y, robot_name, second_robot_name, current_time);
            //

            // Generate result
            {
              Result result;
              result.current_time = std::to_string(round(ros::Time::now().toSec() * 100.0) / 100.0);
              if (robot_name < second_robot_name)
              {
                result.from_robot = robot_name;
                result.to_robot = second_robot_name;
              }
              else
              {
                result.from_robot = second_robot_name;
                result.to_robot = robot_name;
              }
              result.x = pose1.position.x;
              result.y = pose1.position.y;
              result.tx = adjusted_transform.at<double>(0, 2) * resolutions_[self_robot_index];
              result.ty = adjusted_transform.at<double>(1, 2) * resolutions_[self_robot_index];
              result.r = atan2(adjusted_transform.at<double>(1, 0), adjusted_transform.at<double>(0, 0));
              result.match_score = confidence_output;
              result.t_error = sqrt(pow(predicted_pose.at<double>(0, 0) - pose1.position.x, 2) + pow(predicted_pose.at<double>(1, 0) - pose1.position.y, 2));
              double truth_r;
              if (result.from_robot == "/tb3_0" && result.to_robot == "/tb3_1")
              {
                truth_r = 0.0; // HARDCODE
              }
              else if (result.from_robot == "/tb3_0" && result.to_robot == "/tb3_2")
              {
                truth_r = -0.785; // HARDCODE
              }
              else if (result.from_robot == "/tb3_1" && result.to_robot == "/tb3_2")
              {
                truth_r = -0.785; // HARDCODE
              }
              double rot_0_0 = cos(result.r) * cos(-1.0 * truth_r) - sin(result.r) * sin(-1.0 * truth_r);
              double rot_1_0 = sin(result.r) * cos(-1.0 * truth_r) + cos(result.r) * sin(-1.0 * truth_r);
              result.r_error = abs(atan2(rot_1_0, rot_0_0));
              appendResultToFile(result);
            }

            for (int i = 0; i < 20; ++i)
            {
              double start = 1.0;
              double interval = 0.1;
              double threshold = start + i * interval;
              if (confidence_output >= threshold)
              {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(2) << threshold;
                std::string threshold_str = ss.str();
                collectProposingData(pose1.position.x, pose1.position.y, predicted_pose.at<double>(0, 0), predicted_pose.at<double>(1, 0), confidence_output, threshold_str, robot_name, second_robot_name, current_time, false);
              }
              if (confidence_output >= threshold && confidence_output < threshold + 0.1)
              {
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(2) << threshold;
                std::string threshold_str = ss.str();
                collectProposingData(pose1.position.x, pose1.position.y, predicted_pose.at<double>(0, 0), predicted_pose.at<double>(1, 0), confidence_output, threshold_str, robot_name, second_robot_name, current_time, true);
              }
            }
          }
        }

        ROS_DEBUG("Robot %s finishes matching with the second robot %s, which has %zu candidates.", robot_name.c_str(), second_robot_name.c_str(), pair.second.size());
      }
    }
  }

  void Traceback::appendResultToFile(Result result)
  {
    boost::lock_guard<boost::shared_mutex> lock(result_file_mutex_);
    std::ofstream fw("_result.csv", std::ofstream::app);
    if (fw.is_open())
    {
      fw << result_index_ << "," << result.current_time << "," << result.from_robot << "," << result.to_robot << "," << result.x << "," << result.y << "," << result.tx << "," << result.ty << "," << result.r << "," << result.match_score << "," << result.t_error << "," << result.r_error << std::endl;
      fw.close();
    }
    ++result_index_;
  }

  void Traceback::pushData()
  {
    // Skip when there are obstacles nearby
    std::unordered_map<std::string, bool> robots_to_skip;
    boost::shared_lock<boost::shared_mutex> lock(map_subscriptions_mutex_);
    for (auto &subscription : map_subscriptions_)
    {
      robots_to_skip[subscription.robot_namespace] = hasObstacleNearby(subscription, check_obstacle_nearby_pixel_distance_);
    }
    //

    for (auto current : camera_image_processor_.robots_to_current_image_)
    {
      std::string robot_name = current.first;
      // Skip when there are obstacles nearby
      if (robots_to_skip[robot_name])
      {
        continue;
      }
      //
      geometry_msgs::Pose pose = camera_image_processor_.robots_to_current_pose_[current.first];
      sensor_msgs::Image depth_image = camera_image_processor_.robots_to_current_depth_image_[current.first];
      // sensor_msgs::PointCloud2 point_cloud = camera_image_processor_.robots_to_current_point_cloud_[current.first];
      PoseImagePair pose_image_pair;
      pose_image_pair.pose = pose;
      pose_image_pair.image = current.second;
      pose_image_pair.depth_image = depth_image;
      pose_image_pair.stamp = ros::Time::now().toNSec();

      auto all = camera_image_processor_.robots_to_all_pose_image_pairs_.find(current.first);
      if (all != camera_image_processor_.robots_to_all_pose_image_pairs_.end())
      {
        if (all->second.size() >= camera_pose_image_max_queue_size_)
        {
          {
            boost::lock_guard<boost::shared_mutex> lock(robots_to_current_it_mutex_[current.first]);

            auto current_it = robots_to_current_it_.find(current.first);
            if (current_it != robots_to_current_it_.end())
            {
              // ROS_INFO("Before erase current_it stamp: %ld", all->second[current_it->second].stamp);
              if (current_it->second == 0)
              {
                all->second.erase(all->second.begin() + camera_pose_image_queue_skip_count_);
              }
              else
              {
                if (robots_to_current_it_[current.first] + camera_pose_image_queue_skip_count_ >= all->second.size())
                {
                  all->second.erase(all->second.begin() + robots_to_current_it_[current.first] + camera_pose_image_queue_skip_count_ - all->second.size());
                }
                else
                {
                  all->second.erase(all->second.begin());
                }
                --robots_to_current_it_[current.first];
              }
              // ROS_INFO("After erase current_it stamp: %ld", all->second[current_it->second].stamp);
            }
          }
        }
        //
        all->second.emplace_back(pose_image_pair);
        PoseImagePair latestPair = *std::max_element(camera_image_processor_.robots_to_all_pose_image_pairs_[current.first].begin(), camera_image_processor_.robots_to_all_pose_image_pairs_[current.first].end());
        // ROS_INFO("latestPair.stamp: %ld", latestPair.stamp);
      }
      else
      {
        camera_image_processor_.robots_to_all_pose_image_pairs_.insert({current.first, {pose_image_pair}});
      }
      // ROS_DEBUG("camera_image_processor_.robots_to_all_pose_image_pairs_[%s] size: %zu", all->first.c_str(), all->second.size());
    }
  }

  void Traceback::addTracebackLoopClosureConstraint(cv::Mat &adjusted_transform, double src_x, double src_y, std::string src_robot, std::string dst_robot)
  {
    if (src_robot < dst_robot)
    {
      LoopClosureConstraint loop_closure_constraint;
      loop_closure_constraint.x = src_x;
      loop_closure_constraint.y = src_y;
      loop_closure_constraint.tx = adjusted_transform.at<double>(0, 2);
      loop_closure_constraint.ty = adjusted_transform.at<double>(1, 2);
      loop_closure_constraint.r = atan2(adjusted_transform.at<double>(1, 0), adjusted_transform.at<double>(0, 0));
      robot_to_robot_traceback_loop_closure_constraints_[src_robot][dst_robot].push_back(loop_closure_constraint);
    }
    else
    {
      LoopClosureConstraint loop_closure_constraint;
      cv::Mat src(3, 1, CV_64F);
      src.at<double>(0, 0) = src_x;
      src.at<double>(1, 0) = src_y;
      src.at<double>(2, 0);
      cv::Mat dst = adjusted_transform * src;
      double inv_x = dst.at<double>(0, 0);
      double inv_y = dst.at<double>(1, 0);
      loop_closure_constraint.x = inv_x;
      loop_closure_constraint.y = inv_y;
      cv::Mat inv_adjusted_transform = adjusted_transform.inv();
      loop_closure_constraint.tx = inv_adjusted_transform.at<double>(0, 2);
      loop_closure_constraint.ty = inv_adjusted_transform.at<double>(1, 2);
      loop_closure_constraint.r = atan2(inv_adjusted_transform.at<double>(1, 0), inv_adjusted_transform.at<double>(0, 0));
      robot_to_robot_traceback_loop_closure_constraints_[dst_robot][src_robot].push_back(loop_closure_constraint);
    }
  }

  void Traceback::addCandidateLoopClosureConstraint(cv::Mat &adjusted_transform, double src_x, double src_y, std::string src_robot, std::string dst_robot)
  {
    if (src_robot < dst_robot)
    {
      LoopClosureConstraint loop_closure_constraint;
      loop_closure_constraint.x = src_x;
      loop_closure_constraint.y = src_y;
      loop_closure_constraint.tx = adjusted_transform.at<double>(0, 2);
      loop_closure_constraint.ty = adjusted_transform.at<double>(1, 2);
      loop_closure_constraint.r = atan2(adjusted_transform.at<double>(1, 0), adjusted_transform.at<double>(0, 0));
      robot_to_robot_candidate_loop_closure_constraints_[src_robot][dst_robot].push_back(loop_closure_constraint);
    }
    else
    {
      LoopClosureConstraint loop_closure_constraint;
      cv::Mat src(3, 1, CV_64F);
      src.at<double>(0, 0) = src_x;
      src.at<double>(1, 0) = src_y;
      src.at<double>(2, 0);
      cv::Mat dst = adjusted_transform * src;
      double inv_x = dst.at<double>(0, 0);
      double inv_y = dst.at<double>(1, 0);
      loop_closure_constraint.x = inv_x;
      loop_closure_constraint.y = inv_y;
      cv::Mat inv_adjusted_transform = adjusted_transform.inv();
      loop_closure_constraint.tx = inv_adjusted_transform.at<double>(0, 2);
      loop_closure_constraint.ty = inv_adjusted_transform.at<double>(1, 2);
      loop_closure_constraint.r = atan2(inv_adjusted_transform.at<double>(1, 0), inv_adjusted_transform.at<double>(0, 0));
      robot_to_robot_candidate_loop_closure_constraints_[dst_robot][src_robot].push_back(loop_closure_constraint);
    }
  }

  void Traceback::addLoopClosureConstraint(cv::Mat &adjusted_transform, double src_x, double src_y, std::string src_robot, std::string dst_robot)
  {
    if (src_robot < dst_robot)
    {
      LoopClosureConstraint loop_closure_constraint;
      loop_closure_constraint.x = src_x;
      loop_closure_constraint.y = src_y;
      loop_closure_constraint.tx = adjusted_transform.at<double>(0, 2);
      loop_closure_constraint.ty = adjusted_transform.at<double>(1, 2);
      loop_closure_constraint.r = atan2(adjusted_transform.at<double>(1, 0), adjusted_transform.at<double>(0, 0));
      robot_to_robot_loop_closure_constraints_[src_robot][dst_robot].push_back(loop_closure_constraint);
    }
    else
    {
      LoopClosureConstraint loop_closure_constraint;
      cv::Mat src(3, 1, CV_64F);
      src.at<double>(0, 0) = src_x;
      src.at<double>(1, 0) = src_y;
      src.at<double>(2, 0);
      cv::Mat dst = adjusted_transform * src;
      double inv_x = dst.at<double>(0, 0);
      double inv_y = dst.at<double>(1, 0);
      loop_closure_constraint.x = inv_x;
      loop_closure_constraint.y = inv_y;
      cv::Mat inv_adjusted_transform = adjusted_transform.inv();
      loop_closure_constraint.tx = inv_adjusted_transform.at<double>(0, 2);
      loop_closure_constraint.ty = inv_adjusted_transform.at<double>(1, 2);
      loop_closure_constraint.r = atan2(inv_adjusted_transform.at<double>(1, 0), inv_adjusted_transform.at<double>(0, 0));
      robot_to_robot_loop_closure_constraints_[dst_robot][src_robot].push_back(loop_closure_constraint);
    }
  }

  void Traceback::transformOptimization()
  {
    ROS_DEBUG("Transform optimization started.");
    {
      boost::shared_lock<boost::shared_mutex> lock(loop_constraints_mutex_);

      /** Remove loop closure constraints that are too different from the latest accepted loop closure constraint */
      for (auto &src : robot_to_robot_loop_closure_constraints_)
      {
        for (auto &dst : src.second)
        {
          auto it = robot_to_robot_latest_accepted_loop_closure_constraint_.find(src.first);
          if (it != robot_to_robot_latest_accepted_loop_closure_constraint_.end())
          {
            auto it2 = robot_to_robot_latest_accepted_loop_closure_constraint_[src.first].find(dst.first);
            if (it2 != robot_to_robot_latest_accepted_loop_closure_constraint_[src.first].end())
            {
              LoopClosureConstraint latest_accepted_constraint = it2->second;
              std::vector<LoopClosureConstraint>::iterator it3 = dst.second.begin();
              while (it3 != dst.second.end())
              {
                LoopClosureConstraint constraint = *it3;
                double RESOLUTION = 0.05;
                constraint.x *= RESOLUTION;
                constraint.y *= RESOLUTION;
                constraint.tx *= RESOLUTION;
                constraint.ty *= RESOLUTION;
                latest_accepted_constraint.x *= RESOLUTION;
                latest_accepted_constraint.y *= RESOLUTION;
                latest_accepted_constraint.tx *= RESOLUTION;
                latest_accepted_constraint.ty *= RESOLUTION;

                // cv::Mat pose(3, 1, CV_64F);
                // pose.at<double>(0, 0) = constraint.x;
                // pose.at<double>(1, 0) = constraint.y;
                // pose.at<double>(2, 0) = 1.0;
                // cv::Mat transform(3, 3, CV_64F);
                // transform.at<double>(0, 0) = cos(constraint.r);
                // transform.at<double>(0, 1) = -sin(constraint.r);
                // transform.at<double>(0, 2) = constraint.tx;
                // transform.at<double>(1, 0) = sin(constraint.r);
                // transform.at<double>(1, 1) = cos(constraint.r);
                // transform.at<double>(1, 2) = constraint.ty;
                // transform.at<double>(2, 0) = 0;
                // transform.at<double>(2, 1) = 0;
                // transform.at<double>(2, 2) = 1;
                // cv::Mat accepted_transform(3, 3, CV_64F);
                // accepted_transform.at<double>(0, 0) = cos(latest_accepted_constraint.r);
                // accepted_transform.at<double>(0, 1) = -sin(latest_accepted_constraint.r);
                // accepted_transform.at<double>(0, 2) = latest_accepted_constraint.tx;
                // accepted_transform.at<double>(1, 0) = sin(latest_accepted_constraint.r);
                // accepted_transform.at<double>(1, 1) = cos(latest_accepted_constraint.r);
                // accepted_transform.at<double>(1, 2) = latest_accepted_constraint.ty;
                // accepted_transform.at<double>(2, 0) = 0;
                // accepted_transform.at<double>(2, 1) = 0;
                // accepted_transform.at<double>(2, 2) = 1;
                // cv::Mat predicted_pose = accepted_transform.inv() * transform * pose;
                // double error = sqrt(pow(predicted_pose.at<double>(0, 0) - pose.at<double>(0, 0), 2) + pow(predicted_pose.at<double>(1, 0) - pose.at<double>(1, 0), 2));

                double error_tx = abs(constraint.tx - latest_accepted_constraint.tx);
                double error_ty = abs(constraint.ty - latest_accepted_constraint.ty); 
                double error_r = abs(constraint.tx - latest_accepted_constraint.tx); 

                if (error_tx >= far_from_accepted_transform_threshold_ || error_ty >= far_from_accepted_transform_threshold_ || error_r >= far_from_accepted_transform_threshold_ / 4.0)
                {
                  //
                  size_t result_loop_index = it3 - dst.second.begin();
                  size_t smaller_or_equal_count = 0;
                  for (size_t index : result_loop_indexes_)
                  {
                    if (index <= result_loop_index)
                    {
                      ++smaller_or_equal_count;
                    }
                  }
                  size_t result_index = result_loop_index + smaller_or_equal_count;
                  //
                  result_loop_indexes_.push_back(result_loop_index);
                  //

                  it3 = dst.second.erase(it3);

                  {
                    std::string current_time = std::to_string(round(ros::Time::now().toSec() * 100.0) / 100.0);
                    std::ofstream fw("_erased_loop_closure_" + src.first.substr(1) + "_to_" + dst.first.substr(1) + ".csv", std::ofstream::app);
                    if (fw.is_open())
                    {
                      fw << result_index << "," << current_time << "," << constraint.x << "," << constraint.y << "," << constraint.tx << "," << constraint.ty << "," << constraint.r << std::endl;
                      fw.close();
                    }
                  }
                  continue;
                }

                ++it3;
              }
            }
          }
        }
      }
      /** Remove loop closure constraints that are too different from the latest accepted loop closure constraint END */

      int total_loop_constraint_count = 0;
      for (auto &src : robot_to_robot_loop_closure_constraints_)
      {
        for (auto &dst : src.second)
        {
          total_loop_constraint_count += dst.second.size();
        }
      }

      // Only increment, so do only when there are updates
      if (total_loop_constraint_count != last_total_loop_constraint_count_)
      {
        last_total_loop_constraint_count_ = total_loop_constraint_count;

        // Write to file when total loop constraint number has changed
        {
          std::string current_time = std::to_string(round(ros::Time::now().toSec() * 100.0) / 100.0);

          std::ofstream fw("Loop_closure_count_" + current_time + ".txt", std::ofstream::app);
          if (fw.is_open())
          {
            for (auto &src : robot_to_robot_loop_closure_constraints_)
            {
              for (auto &dst : src.second)
              {
                fw << "Count from " << src.first << " to " << dst.first << " is " << dst.second.size() << std::endl;
              }
            }
            fw.close();
          }
        }

        // Do optimization
        std::string current_time = std::to_string(round(ros::Time::now().toSec() * 100.0) / 100.0);

        for (auto &src : robot_to_robot_loop_closure_constraints_)
        {
          for (auto &dst : src.second)
          {
            std::string robot_name = src.first;
            std::string second_robot_name = dst.first;

            if (robot_name >= second_robot_name)
            {
              continue;
            }

            std::vector<LoopClosureConstraint> loop_closure_constraints = dst.second;

            if (loop_closure_constraints.size() == 0)
            {
              continue;
            }

            std::vector<double> x_values;
            std::vector<double> y_values;
            std::vector<double> tx_values;
            std::vector<double> ty_values;
            std::vector<double> r_values;

            for (LoopClosureConstraint &constraint : loop_closure_constraints)
            {
              x_values.push_back(constraint.x);
              y_values.push_back(constraint.y);
              tx_values.push_back(constraint.tx);
              ty_values.push_back(constraint.ty);
              r_values.push_back(constraint.r);
            }

            double init_tx, init_ty, init_r;

            cv::Mat init_transform = robot_to_robot_optimized_transform_[robot_name][second_robot_name];
            if (!init_transform.empty())
            {
              init_tx = init_transform.at<double>(0, 2);
              init_ty = init_transform.at<double>(1, 2);
              init_r = atan2(init_transform.at<double>(1, 0), init_transform.at<double>(0, 0));
            }
            else
            {
              assert(loop_closure_constraints.size() > 0);
              init_tx = loop_closure_constraints[0].tx;
              init_ty = loop_closure_constraints[0].ty;
              init_r = loop_closure_constraints[0].r;

              // Just for later writing to file without error
              init_transform = cv::Mat::eye(3, 3, CV_64F);
            }

            std::vector<double> optimized_tx_ty_r = camera_image_processor_.LMOptimize(x_values, y_values, tx_values, ty_values, r_values, init_tx, init_ty, init_r);

            double optimized_tx = optimized_tx_ty_r[0];
            double optimized_ty = optimized_tx_ty_r[1];
            double optimized_r = optimized_tx_ty_r[2];

            // Compute transformation from optimized (tx, ty, r)
            cv::Mat optimized_transform(3, 3, CV_64F);
            optimized_transform.at<double>(0, 0) = cos(optimized_r);
            optimized_transform.at<double>(0, 1) = -sin(optimized_r);
            optimized_transform.at<double>(0, 2) = optimized_tx;
            optimized_transform.at<double>(1, 0) = sin(optimized_r);
            optimized_transform.at<double>(1, 1) = cos(optimized_r);
            optimized_transform.at<double>(1, 2) = optimized_ty;
            optimized_transform.at<double>(2, 0) = 0;
            optimized_transform.at<double>(2, 1) = 0;
            optimized_transform.at<double>(2, 2) = 1;

            robot_to_robot_optimized_transform_[robot_name][second_robot_name] = optimized_transform;

            //
            transform_estimator_.updateBestTransforms(optimized_transform, robot_name, second_robot_name, best_transforms_, has_best_transforms_);
            //

            evaluateWithGroundTruth(init_transform, optimized_transform, robot_name, second_robot_name, current_time);

            if (has_best_transforms_.size() == resolutions_.size())
            {
              std::vector<std::string> robot_names;
              std::vector<double> m_00;
              std::vector<double> m_01;
              std::vector<double> m_02;
              std::vector<double> m_10;
              std::vector<double> m_11;
              std::vector<double> m_12;
              std::vector<double> m_20;
              std::vector<double> m_21;
              std::vector<double> m_22;
              for (auto it = best_transforms_[robot_name].begin(); it != best_transforms_[robot_name].end(); ++it)
              {
                std::string dst_robot = it->first;
                cv::Mat dst_transform = it->second;
                robot_names.push_back(dst_robot);
                m_00.push_back(dst_transform.at<double>(0, 0));
                m_01.push_back(dst_transform.at<double>(0, 1));
                m_02.push_back(dst_transform.at<double>(0, 2));
                m_10.push_back(dst_transform.at<double>(1, 0));
                m_11.push_back(dst_transform.at<double>(1, 1));
                m_12.push_back(dst_transform.at<double>(1, 2));
                m_20.push_back(dst_transform.at<double>(2, 0));
                m_21.push_back(dst_transform.at<double>(2, 1));
                m_22.push_back(dst_transform.at<double>(2, 2));

                {
                  std::ofstream fw("Best_transforms_" + current_time + "_" + robot_name.substr(1) + "_tracer_robot.txt", std::ofstream::app);
                  if (fw.is_open())
                  {
                    fw << "Best transform from " << robot_name << " to " << dst_robot << " :" << std::endl;
                    fw << dst_transform.at<double>(0, 0) << "\t" << dst_transform.at<double>(0, 1) << "\t" << dst_transform.at<double>(0, 2) << std::endl;
                    fw << dst_transform.at<double>(1, 0) << "\t" << dst_transform.at<double>(1, 1) << "\t" << dst_transform.at<double>(1, 2) << std::endl;
                    fw << dst_transform.at<double>(2, 0) << "\t" << dst_transform.at<double>(2, 1) << "\t" << dst_transform.at<double>(2, 2) << std::endl;
                    fw.close();
                  }
                }
              }

              traceback_msgs::TracebackTransforms traceback_transforms;
              traceback_transforms.robot_names = robot_names;
              traceback_transforms.m_00 = m_00;
              traceback_transforms.m_01 = m_01;
              traceback_transforms.m_02 = m_02;
              traceback_transforms.m_10 = m_10;
              traceback_transforms.m_11 = m_11;
              traceback_transforms.m_12 = m_12;
              traceback_transforms.m_20 = m_20;
              traceback_transforms.m_21 = m_21;
              traceback_transforms.m_22 = m_22;

              traceback_transforms_publisher_.publish(traceback_transforms);
            }
          }
        }
      }
    }
  }

  void Traceback::poseEstimation()
  {
    ROS_DEBUG("Grid pose estimation started.");
    std::vector<nav_msgs::OccupancyGridConstPtr> grids;
    grids.reserve(map_subscriptions_size_);
    {
      resolutions_.clear();
      map_origins_.clear();

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
        resolutions_.emplace_back(subscription.readonly_map->info.resolution);
        map_origins_.emplace_back(cv::Point2d(subscription.readonly_map->info.origin.position.x, subscription.readonly_map->info.origin.position.y));
        ++i;
      }
    }

    transform_estimator_.feed(grids.begin(), grids.end());

    if (estimation_mode_ == "map")
    {
      transform_estimator_.estimateTransforms(FeatureType::AKAZE,
                                              confidence_threshold_);
    }
  }

  void Traceback::CameraImageUpdate(const sensor_msgs::ImageConstPtr &msg)
  {
    std::string frame_id = msg->header.frame_id;
    std::string robot_name = "/" + frame_id.substr(0, frame_id.find("/"));
    camera_image_processor_.robots_to_temp_image_[robot_name] = *msg;
  }

  // rgb and depth images may not be synchronized
  void Traceback::CameraDepthImageUpdate(const sensor_msgs::ImageConstPtr &msg)
  {
    std::string frame_id = msg->header.frame_id;
    std::string robot_name = "/" + frame_id.substr(0, frame_id.find("/"));
    camera_image_processor_.robots_to_current_depth_image_[robot_name] = *msg;
    camera_image_processor_.robots_to_current_pose_[robot_name] = getRobotPose(robot_name);
    camera_image_processor_.robots_to_current_image_[robot_name] = camera_image_processor_.robots_to_temp_image_[robot_name];
  }

  void Traceback::modifyTransformsBasedOnOrigins(
      std::vector<cv::Mat> &transforms, std::vector<cv::Mat> &out,
      std::vector<cv::Point2d> &map_origins, std::vector<float> &resolutions)
  {
    size_t identity_index = -1;
    const double ZERO_ERROR = 0.0001;
    for (size_t k = 0; k < transforms.size(); ++k)
    {
      if (abs(transforms[k].at<double>(0, 0) - 1.0) < ZERO_ERROR &&
          abs(transforms[k].at<double>(0, 1) - 0.0) < ZERO_ERROR &&
          abs(transforms[k].at<double>(0, 2) - 0.0) < ZERO_ERROR &&
          abs(transforms[k].at<double>(1, 0) - 0.0) < ZERO_ERROR &&
          abs(transforms[k].at<double>(1, 1) - 1.0) < ZERO_ERROR &&
          abs(transforms[k].at<double>(1, 2) - 0.0) < ZERO_ERROR &&
          abs(transforms[k].at<double>(2, 0) - 0.0) < ZERO_ERROR &&
          abs(transforms[k].at<double>(2, 1) - 0.0) < ZERO_ERROR &&
          abs(transforms[k].at<double>(2, 2) - 1.0) < ZERO_ERROR)
      {
        identity_index = k;
      }
    }

    if (identity_index == -1)
    {
      return;
    }

    out.clear();
    out.reserve(transforms.size());

    double src_map_origin_x = map_origins[identity_index].x;
    double src_map_origin_y = map_origins[identity_index].y;
    float src_resolution = resolutions[identity_index];
    for (size_t i = 0; i < transforms.size(); ++i)
    {
      double dst_map_origin_x = map_origins[i].x;
      double dst_map_origin_y = map_origins[i].y;
      float dst_resolution = resolutions[i];

      cv::Mat t1(3, 3, CV_64F);
      t1.at<double>(0, 0) = 1.0;
      t1.at<double>(0, 1) = 0.0;
      t1.at<double>(0, 2) = src_map_origin_x / src_resolution;
      t1.at<double>(1, 0) = 0.0;
      t1.at<double>(1, 1) = 1.0;
      t1.at<double>(1, 2) = src_map_origin_y / src_resolution;
      t1.at<double>(2, 0) = 0.0;
      t1.at<double>(2, 1) = 0.0;
      t1.at<double>(2, 2) = 1.0;

      cv::Mat t2(3, 3, CV_64F);
      t2.at<double>(0, 0) = 1.0;
      t2.at<double>(0, 1) = 0.0;
      t2.at<double>(0, 2) = -1 * dst_map_origin_x / dst_resolution;
      t2.at<double>(1, 0) = 0.0;
      t2.at<double>(1, 1) = 1.0;
      t2.at<double>(1, 2) = -1 * dst_map_origin_y / dst_resolution;
      t2.at<double>(2, 0) = 0.0;
      t2.at<double>(2, 1) = 0.0;
      t2.at<double>(2, 2) = 1.0;

      out.emplace_back(t2 * transforms[i] * t1);
    }
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
          transforms_indexes_.insert({map_subscriptions_size_, robot_name});
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
        std::string camera_rgb_topic;
        std::string camera_depth_topic;
        std::string camera_point_cloud_topic;

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
        camera_rgb_topic = ros::names::append(robot_name, robot_camera_image_topic_);
        camera_depth_topic = ros::names::append(robot_name, robot_camera_depth_image_topic_);

        subscription.robot_namespace = robot_name;

        ROS_INFO("Subscribing to CAMERA rgb topic: %s.", camera_rgb_topic.c_str());
        ROS_INFO("Subscribing to CAMERA depth topic: %s.", camera_depth_topic.c_str());
        // ROS_INFO("Subscribing to CAMERA point cloud topic: %s.", camera_point_cloud_topic.c_str());

        // Insert empty std::vector to the map to prevent future error when accessing the map by robot name.
        auto it = camera_image_processor_.robots_to_all_pose_image_pairs_.find(subscription.robot_namespace);
        if (it == camera_image_processor_.robots_to_all_pose_image_pairs_.end())
        {
          camera_image_processor_.robots_to_all_pose_image_pairs_.insert({subscription.robot_namespace, {}});
        }

        subscription.camera_rgb_sub.subscribe(node_,
                                              camera_rgb_topic, 10);

        subscription.camera_depth_sub.subscribe(node_,
                                                camera_depth_topic, 10);

        // subscription.camera_point_cloud_sub.subscribe(node_,
        //                                               camera_point_cloud_topic, 10);

        subscription.camera_rgb_sub.registerCallback(boost::bind(&Traceback::CameraImageUpdate, this, _1));
        subscription.camera_depth_sub.registerCallback(boost::bind(&Traceback::CameraDepthImageUpdate, this, _1));
        // subscription.camera_point_cloud_sub.registerCallback(boost::bind(&Traceback::CameraPointCloudUpdate, this, _1));

        // Synchronizer does not callback for unknown reason.
        // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;

        // message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(50), subscription.camera_rgb_sub, subscription.camera_point_cloud_sub);
        // boost::function<void(sensor_msgs::ImageConstPtr, sensor_msgs::PointCloud2ConstPtr)> callback(boost::bind(&Traceback::CameraImageUpdate, this, _1, _2));
        // sync.registerCallback(boost::bind(callback, _1, _2));

        // Create goal publisher for this robot
        robots_to_goal_and_image_publisher_.emplace(robot_name, node_.advertise<traceback_msgs::GoalAndImage>(ros::names::append(robot_name, traceback_goal_and_image_topic_), 10));

        robots_to_image_and_image_subscriber_.emplace(robot_name, node_.subscribe<traceback_msgs::ImageAndImage>(
                                                                      ros::names::append(robot_name, traceback_image_and_image_topic_), 10,
                                                                      [this](const traceback_msgs::ImageAndImage::ConstPtr &msg)
                                                                      {
                                                                        tracebackImageAndImageUpdate(msg);
                                                                      }));
        robots_to_visualize_marker_publisher_.emplace(robot_name, node_.advertise<visualization_msgs::Marker>(ros::names::append(robot_name, visualize_goal_topic_), 10));

        robots_to_in_traceback_.emplace(robot_name, false);
        robots_to_current_it_.emplace(robot_name, 0);

        traceback_transforms_publisher_ = node_.advertise<traceback_msgs::TracebackTransforms>(traceback_transforms_topic_, 10);

        robots_to_image_features_depths_pose_[robot_name] = {};

        for (auto it = robots_to_camera_subscriptions_.begin(); it != robots_to_camera_subscriptions_.end(); ++it)
        {
          AcceptRejectStatus status;
          status.accept_count = 0;
          status.reject_count = 0;
          status.accepted = false;
          pairwise_accept_reject_status_[it->first][robot_name] = status;
          pairwise_accept_reject_status_[robot_name][it->first] = status;
          pairwise_abort_[it->first][robot_name] = 0;
          pairwise_abort_[robot_name][it->first] = 0;
          pairwise_paused_[it->first][robot_name] = false;
          pairwise_paused_[robot_name][it->first] = false;

          robot_to_robot_traceback_accept_count_[it->first][robot_name] = 0;
          robot_to_robot_traceback_accept_count_[robot_name][it->first] = 0;

          robot_to_robot_traceback_loop_closure_constraints_[it->first][robot_name] = {};
          robot_to_robot_traceback_loop_closure_constraints_[robot_name][it->first] = {};
          robot_to_robot_candidate_loop_closure_constraints_[it->first][robot_name] = {};
          robot_to_robot_candidate_loop_closure_constraints_[robot_name][it->first] = {};
          robot_to_robot_loop_closure_constraints_[it->first][robot_name] = {};
          robot_to_robot_loop_closure_constraints_[robot_name][it->first] = {};

          if (it->first < robot_name || it->first > robot_name)
          {
            std::string src_robot = it->first < robot_name ? it->first : robot_name;
            std::string dst_robot = it->first < robot_name ? robot_name : it->first;
            std::string current_time = std::to_string(round(ros::Time::now().toSec() * 100.0) / 100.0);
            std::ofstream fw("_erased_loop_closure_" + src_robot.substr(1) + "_to_" + dst_robot.substr(1) + ".csv", std::ofstream::app);
            if (fw.is_open())
            {
              fw << "result_index"
                 << ","
                 << "timestamp"
                 << ","
                 << "x"
                 << ","
                 << "y"
                 << ","
                 << "tx"
                 << ","
                 << "ty"
                 << ","
                 << "r"
                 << std::endl;
              fw.close();
            }
          }
        }
      }
    }
  }

  bool Traceback::readOptimizedTransform(cv::Mat &transform, cv::Mat &inv_transform, std::string from, std::string to)
  {
    if (from < to)
    {
      transform = robot_to_robot_optimized_transform_[from][to];
      if (transform.empty())
      {
        return false;
      }
      inv_transform = transform.inv();
    }
    else
    {
      inv_transform = robot_to_robot_optimized_transform_[to][from];
      if (inv_transform.empty())
      {
        return false;
      }
      transform = inv_transform.inv();
    }

    if (transform.empty() || inv_transform.empty())
    {
      return false;
    }

    return true;
  }

  cv_bridge::CvImageConstPtr Traceback::sensorImageToCvImagePtr(const sensor_msgs::Image &image)
  {
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
      if (sensor_msgs::image_encodings::isColor(image.encoding))
        cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
      else
        cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::TYPE_32FC1);
    }
    catch (cv_bridge::Exception &e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return nullptr;
    }
    return cv_ptr;
  }

  void Traceback::matToQuaternion(cv::Mat &mat, geometry_msgs::Quaternion &q)
  {
    double a = mat.at<double>(0, 0);
    double b = mat.at<double>(1, 0);
    double mag = sqrt(a * a + b * b);
    if (mag != 0)
    {
      a /= mag;
      b /= mag;
    }
    if (a > 1)
      a = 0.9999;
    if (a < -0.9999)
      a = -1;
    if (b > 1)
      b = 0.9999;
    if (b < -1)
      b = -0.9999;
    q.w = std::sqrt(2. + 2. * a) * 0.5;
    q.x = 0.;
    q.y = 0.;
    q.z = std::copysign(std::sqrt(2. - 2. * a) * 0.5, b);
  }

  double Traceback::quaternionToYaw(geometry_msgs::Quaternion &q)
  {
    tf2::Quaternion tf_q;
    tf_q.setW(q.w);
    tf_q.setX(q.x);
    tf_q.setY(q.y);
    tf_q.setZ(q.z);
    tf2::Matrix3x3 m(tf_q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    return yaw;
  }

  // Input 3x3, output 3x3.
  void Traceback::imageTransformToMapTransform(cv::Mat &image, cv::Mat &map, float src_resolution, float dst_resolution, double src_map_origin_x, double src_map_origin_y, double dst_map_origin_x, double dst_map_origin_y)
  {
    cv::Mat t1(3, 3, CV_64F);
    t1.at<double>(0, 0) = 1.0;
    t1.at<double>(0, 1) = 0.0;
    t1.at<double>(0, 2) = -1 * src_map_origin_x / src_resolution;
    t1.at<double>(1, 0) = 0.0;
    t1.at<double>(1, 1) = 1.0;
    t1.at<double>(1, 2) = -1 * src_map_origin_y / src_resolution;
    t1.at<double>(2, 0) = 0.0;
    t1.at<double>(2, 1) = 0.0;
    t1.at<double>(2, 2) = 1.0;

    cv::Mat t2(3, 3, CV_64F);
    t2.at<double>(0, 0) = 1.0;
    t2.at<double>(0, 1) = 0.0;
    t2.at<double>(0, 2) = dst_map_origin_x / dst_resolution;
    t2.at<double>(1, 0) = 0.0;
    t2.at<double>(1, 1) = 1.0;
    t2.at<double>(1, 2) = dst_map_origin_y / dst_resolution;
    t2.at<double>(2, 0) = 0.0;
    t2.at<double>(2, 1) = 0.0;
    t2.at<double>(2, 2) = 1.0;

    map = t2 * image * t1;
  }

  void Traceback::findAdjustedTransformation(cv::Mat &original, cv::Mat &adjusted, double transform_needed_tx, double transform_needed_ty, double transform_needed_r, double arrived_x, double arrived_y, float src_resolution)
  {
    cv::Mat t1(3, 3, CV_64F);
    t1.at<double>(0, 0) = 1.0;
    t1.at<double>(0, 1) = 0.0;
    t1.at<double>(0, 2) = -1 * arrived_x / src_resolution;
    t1.at<double>(1, 0) = 0.0;
    t1.at<double>(1, 1) = 1.0;
    t1.at<double>(1, 2) = -1 * arrived_y / src_resolution;
    t1.at<double>(2, 0) = 0.0;
    t1.at<double>(2, 1) = 0.0;
    t1.at<double>(2, 2) = 1.0;

    /** Need to invert all transform needed */
    // Ensure that they are passed by value
    transform_needed_tx *= -1;
    transform_needed_ty *= -1;
    transform_needed_r *= -1;
    cv::Mat adjustment(3, 3, CV_64F);
    adjustment.at<double>(0, 0) = cos(transform_needed_r);
    adjustment.at<double>(0, 1) = -sin(transform_needed_r);
    adjustment.at<double>(0, 2) = transform_needed_tx / src_resolution;
    adjustment.at<double>(1, 0) = sin(transform_needed_r);
    adjustment.at<double>(1, 1) = cos(transform_needed_r);
    adjustment.at<double>(1, 2) = transform_needed_ty / src_resolution;
    adjustment.at<double>(2, 0) = 0.0;
    adjustment.at<double>(2, 1) = 0.0;
    adjustment.at<double>(2, 2) = 1.0;

    cv::Mat t2(3, 3, CV_64F);
    t2.at<double>(0, 0) = 1.0;
    t2.at<double>(0, 1) = 0.0;
    t2.at<double>(0, 2) = arrived_x / src_resolution;
    t2.at<double>(1, 0) = 0.0;
    t2.at<double>(1, 1) = 1.0;
    t2.at<double>(1, 2) = arrived_y / src_resolution;
    t2.at<double>(2, 0) = 0.0;
    t2.at<double>(2, 1) = 0.0;
    t2.at<double>(2, 2) = 1.0;

    adjusted = t2 * adjustment * t1 * original;
  }

  void Traceback::evaluateWithGroundTruth(cv::Mat &original, cv::Mat &adjusted, std::string tracer_robot, std::string traced_robot, std::string current_time)
  {
    // initial poses w.r.t. global frame
    double resolution = 0.05;
    double init_0_x = -7.0;
    double init_0_y = -1.0;
    double init_0_r = 0.0;
    double init_1_x = 7.0;
    double init_1_y = -1.0;
    double init_1_r = 0.0;
    double init_2_x = 0.5;
    double init_2_y = 3.0;
    double init_2_r = 0.785;

    // global origin object w.r.t. frame 0
    cv::Mat truth_0_xy(3, 1, CV_64F);
    truth_0_xy.at<double>(0, 0) = -1 * init_0_x / resolution;
    truth_0_xy.at<double>(1, 0) = -1 * init_0_y / resolution;
    truth_0_xy.at<double>(2, 0) = 1.0;
    double truth_0_r = 0.0;
    // global origin object w.r.t. frame 1
    cv::Mat truth_1_xy(3, 1, CV_64F);
    truth_1_xy.at<double>(0, 0) = -1 * init_1_x / resolution;
    truth_1_xy.at<double>(1, 0) = -1 * init_1_y / resolution;
    truth_1_xy.at<double>(2, 0) = 1.0;
    double truth_1_r = 0.0;
    // global origin object w.r.t. frame 2
    cv::Mat truth_2_xy(3, 1, CV_64F);
    truth_2_xy.at<double>(0, 0) = -1 * init_2_x / resolution;
    truth_2_xy.at<double>(1, 0) = -1 * init_2_y / resolution;
    truth_2_xy.at<double>(2, 0) = 1.0;
    double truth_2_r = -1 * init_2_r;

    cv::Mat test_xy;
    double test_r;
    if (tracer_robot == "/tb3_0")
    {
      test_xy = truth_0_xy;
      test_r = truth_0_r;
    }
    else if (tracer_robot == "/tb3_1")
    {
      test_xy = truth_1_xy;
      test_r = truth_1_r;
    }
    else if (tracer_robot == "/tb3_2")
    {
      test_xy = truth_2_xy;
      test_r = truth_2_r;
    }

    double expected_x, expected_y, expected_r;
    if (traced_robot == "/tb3_0")
    {
      expected_x = truth_0_xy.at<double>(0, 0);
      expected_y = truth_0_xy.at<double>(1, 0);
      expected_r = truth_0_r;
    }
    else if (traced_robot == "/tb3_1")
    {
      expected_x = truth_1_xy.at<double>(0, 0);
      expected_y = truth_1_xy.at<double>(1, 0);
      expected_r = truth_1_r;
    }
    else if (traced_robot == "/tb3_2")
    {
      expected_x = truth_2_xy.at<double>(0, 0);
      expected_y = truth_2_xy.at<double>(1, 0);
      expected_r = truth_2_r;
    }

    cv::Mat original_estimated_xy, adjusted_estimated_xy;
    double original_estimated_r, adjusted_estimated_r;
    original_estimated_xy = original * test_xy; //
    original_estimated_r = atan2(original.at<double>(1, 0), original.at<double>(0, 0)) + test_r;
    adjusted_estimated_xy = adjusted * test_xy;
    adjusted_estimated_r = atan2(adjusted.at<double>(1, 0), adjusted.at<double>(0, 0)) + test_r;

    // Compute ground truth pairwise matrix
    // e.g. from w.r.t frame 0 to w.r.t frame 2:
    // 0->2 = global->2 * 0->global
    //      = global->2 * inv(global->0)
    // Using the above order, initial pose values can be directly used.
    cv::Mat t_global_0(3, 3, CV_64F);
    t_global_0.at<double>(0, 0) = cos(-1 * init_0_r);
    t_global_0.at<double>(0, 1) = -sin(-1 * init_0_r);
    t_global_0.at<double>(0, 2) = -1 * init_0_x / resolution;
    t_global_0.at<double>(1, 0) = sin(-1 * init_0_r);
    t_global_0.at<double>(1, 1) = cos(-1 * init_0_r);
    t_global_0.at<double>(1, 2) = -1 * init_0_y / resolution;
    t_global_0.at<double>(2, 0) = 0.0;
    t_global_0.at<double>(2, 1) = 0.0;
    t_global_0.at<double>(2, 2) = 1;

    cv::Mat t_global_1(3, 3, CV_64F);
    t_global_1.at<double>(0, 0) = cos(-1 * init_1_r);
    t_global_1.at<double>(0, 1) = -sin(-1 * init_1_r);
    t_global_1.at<double>(0, 2) = -1 * init_1_x / resolution;
    t_global_1.at<double>(1, 0) = sin(-1 * init_1_r);
    t_global_1.at<double>(1, 1) = cos(-1 * init_1_r);
    t_global_1.at<double>(1, 2) = -1 * init_1_y / resolution;
    t_global_1.at<double>(2, 0) = 0.0;
    t_global_1.at<double>(2, 1) = 0.0;
    t_global_1.at<double>(2, 2) = 1;

    cv::Mat t_global_2(3, 3, CV_64F);
    t_global_2.at<double>(0, 0) = cos(-1 * init_2_r);
    t_global_2.at<double>(0, 1) = -sin(-1 * init_2_r);
    t_global_2.at<double>(0, 2) = -1 * init_2_x / resolution;
    t_global_2.at<double>(1, 0) = sin(-1 * init_2_r);
    t_global_2.at<double>(1, 1) = cos(-1 * init_2_r);
    t_global_2.at<double>(1, 2) = -1 * init_2_y / resolution;
    t_global_2.at<double>(2, 0) = 0.0;
    t_global_2.at<double>(2, 1) = 0.0;
    t_global_2.at<double>(2, 2) = 1;

    cv::Mat t_0_1 = t_global_1 * t_global_0.inv();
    cv::Mat t_0_2 = t_global_2 * t_global_0.inv();
    cv::Mat t_1_0 = t_0_1.inv();
    cv::Mat t_1_2 = t_global_2 * t_global_1.inv();
    cv::Mat t_2_0 = t_0_2.inv();
    cv::Mat t_2_1 = t_1_2.inv();

    cv::Mat ground_truth_transform, inverse_ground_truth_transform;
    if (tracer_robot == "/tb3_0" && traced_robot == "/tb3_1")
    {
      ground_truth_transform = t_0_1;
      inverse_ground_truth_transform = t_1_0;
    }
    else if (tracer_robot == "/tb3_0" && traced_robot == "/tb3_2")
    {
      ground_truth_transform = t_0_2;
      inverse_ground_truth_transform = t_2_0;
    }
    else if (tracer_robot == "/tb3_1" && traced_robot == "/tb3_0")
    {
      ground_truth_transform = t_1_0;
      inverse_ground_truth_transform = t_0_1;
    }
    else if (tracer_robot == "/tb3_1" && traced_robot == "/tb3_2")
    {
      ground_truth_transform = t_1_2;
      inverse_ground_truth_transform = t_2_1;
    }
    else if (tracer_robot == "/tb3_2" && traced_robot == "/tb3_0")
    {
      ground_truth_transform = t_2_0;
      inverse_ground_truth_transform = t_0_2;
    }
    else if (tracer_robot == "/tb3_2" && traced_robot == "/tb3_1")
    {
      ground_truth_transform = t_2_1;
      inverse_ground_truth_transform = t_1_2;
    }
    //
    {
      std::ofstream fw("Accepted_transform_" + current_time + "_" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot.txt", std::ofstream::app);
      if (fw.is_open())
      {
        double optimized_tx = adjusted.at<double>(0, 2);
        double optimized_ty = adjusted.at<double>(1, 2);
        double optimized_r = atan2(adjusted.at<double>(1, 0), adjusted.at<double>(0, 0));
        fw << "Unadjusted transform:" << std::endl;
        fw << original.at<double>(0, 0) << "\t" << original.at<double>(0, 1) << "\t" << original.at<double>(0, 2) << std::endl;
        fw << original.at<double>(1, 0) << "\t" << original.at<double>(1, 1) << "\t" << original.at<double>(1, 2) << std::endl;
        fw << original.at<double>(2, 0) << "\t" << original.at<double>(2, 1) << "\t" << original.at<double>(2, 2) << std::endl;
        fw << "Optimized (tx, ty, r) = (" << optimized_tx << ", " << optimized_ty << ", " << optimized_r << ")" << std::endl;
        fw << "Optimized transform:" << std::endl;
        fw << adjusted.at<double>(0, 0) << "\t" << adjusted.at<double>(0, 1) << "\t" << adjusted.at<double>(0, 2) << std::endl;
        fw << adjusted.at<double>(1, 0) << "\t" << adjusted.at<double>(1, 1) << "\t" << adjusted.at<double>(1, 2) << std::endl;
        fw << adjusted.at<double>(2, 0) << "\t" << adjusted.at<double>(2, 1) << "\t" << adjusted.at<double>(2, 2) << std::endl;
        fw.close();
      }
    }

    {
      std::ofstream fw("Accepted_transform_" + current_time + "_" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot.txt", std::ofstream::app);
      if (fw.is_open())
      {
        fw << std::endl;
        fw << "Ground truth transform from robot " << tracer_robot.substr(1) << "'s frame to robot " << traced_robot.substr(1) << "'s frame:" << std::endl;
        fw << ground_truth_transform.at<double>(0, 0) << "\t" << ground_truth_transform.at<double>(0, 1) << "\t" << ground_truth_transform.at<double>(0, 2) << std::endl;
        fw << ground_truth_transform.at<double>(1, 0) << "\t" << ground_truth_transform.at<double>(1, 1) << "\t" << ground_truth_transform.at<double>(1, 2) << std::endl;
        fw << ground_truth_transform.at<double>(2, 0) << "\t" << ground_truth_transform.at<double>(2, 1) << "\t" << ground_truth_transform.at<double>(2, 2) << std::endl;
        fw << "Inverse ground truth transform, that is, from robot " << traced_robot.substr(1) << "'s frame to robot " << tracer_robot.substr(1) << "'s frame:" << std::endl;
        fw << inverse_ground_truth_transform.at<double>(0, 0) << "\t" << inverse_ground_truth_transform.at<double>(0, 1) << "\t" << inverse_ground_truth_transform.at<double>(0, 2) << std::endl;
        fw << inverse_ground_truth_transform.at<double>(1, 0) << "\t" << inverse_ground_truth_transform.at<double>(1, 1) << "\t" << inverse_ground_truth_transform.at<double>(1, 2) << std::endl;
        fw << inverse_ground_truth_transform.at<double>(2, 0) << "\t" << inverse_ground_truth_transform.at<double>(2, 1) << "\t" << inverse_ground_truth_transform.at<double>(2, 2) << std::endl;
        fw.close();
      }
    }

    {
      std::ofstream fw("Accepted_transform_" + current_time + "_" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot.txt", std::ofstream::app);
      if (fw.is_open())
      {
        fw << std::endl;
        fw << "Evaluation:" << std::endl;
        fw << "Expected transformed (x, y, r) of the global origin object" << std::endl;
        fw << "   from robot " << tracer_robot.substr(1) << "'s frame to robot " << traced_robot.substr(1) << "'s frame:" << std::endl;
        fw << "Expected (x, y, r) = (" << expected_x << ", " << expected_y << ", " << expected_r << ")" << std::endl;
        fw << "Original estimated (x, y, r) = (" << original_estimated_xy.at<double>(0, 0) << ", " << original_estimated_xy.at<double>(1, 0) << ", " << original_estimated_r << ")" << std::endl;
        fw << "Adjusted estimated (x, y, r) = (" << adjusted_estimated_xy.at<double>(0, 0) << ", " << adjusted_estimated_xy.at<double>(1, 0) << ", " << adjusted_estimated_r << ")" << std::endl;
        fw << std::endl;
        fw << "Error of original is " << sqrt(pow(expected_x - original_estimated_xy.at<double>(0, 0), 2) + pow(expected_y - original_estimated_xy.at<double>(1, 0), 2)) << " pixels translation and " << abs(expected_r - original_estimated_r) << " radians rotation" << std::endl;
        fw << "Error of adjusted is " << sqrt(pow(expected_x - adjusted_estimated_xy.at<double>(0, 0), 2) + pow(expected_y - adjusted_estimated_xy.at<double>(1, 0), 2)) << " pixels translation and " << abs(expected_r - adjusted_estimated_r) << " radians rotation" << std::endl;
        fw.close();
      }
    }
  }

  cv::Mat Traceback::evaluateMatch(cv::Mat &proposed, double pose_x, double pose_y, std::string tracer_robot, std::string traced_robot, std::string current_time)
  {
    // initial poses w.r.t. global frame
    double resolution = 0.05;
    double init_0_x = -7.0;
    double init_0_y = -1.0;
    double init_0_r = 0.0;
    double init_1_x = 7.0;
    double init_1_y = -1.0;
    double init_1_r = 0.0;
    double init_2_x = 0.5;
    double init_2_y = 3.0;
    double init_2_r = 0.785;

    // Compute ground truth pairwise matrix
    // e.g. from w.r.t frame 0 to w.r.t frame 2:
    // 0->2 = global->2 * 0->global
    //      = global->2 * inv(global->0)
    // Using the above order, initial pose values can be directly used.
    cv::Mat t_global_0(3, 3, CV_64F);
    t_global_0.at<double>(0, 0) = cos(-1 * init_0_r);
    t_global_0.at<double>(0, 1) = -sin(-1 * init_0_r);
    t_global_0.at<double>(0, 2) = -1 * init_0_x / resolution;
    t_global_0.at<double>(1, 0) = sin(-1 * init_0_r);
    t_global_0.at<double>(1, 1) = cos(-1 * init_0_r);
    t_global_0.at<double>(1, 2) = -1 * init_0_y / resolution;
    t_global_0.at<double>(2, 0) = 0.0;
    t_global_0.at<double>(2, 1) = 0.0;
    t_global_0.at<double>(2, 2) = 1;

    cv::Mat t_global_1(3, 3, CV_64F);
    t_global_1.at<double>(0, 0) = cos(-1 * init_1_r);
    t_global_1.at<double>(0, 1) = -sin(-1 * init_1_r);
    t_global_1.at<double>(0, 2) = -1 * init_1_x / resolution;
    t_global_1.at<double>(1, 0) = sin(-1 * init_1_r);
    t_global_1.at<double>(1, 1) = cos(-1 * init_1_r);
    t_global_1.at<double>(1, 2) = -1 * init_1_y / resolution;
    t_global_1.at<double>(2, 0) = 0.0;
    t_global_1.at<double>(2, 1) = 0.0;
    t_global_1.at<double>(2, 2) = 1;

    cv::Mat t_global_2(3, 3, CV_64F);
    t_global_2.at<double>(0, 0) = cos(-1 * init_2_r);
    t_global_2.at<double>(0, 1) = -sin(-1 * init_2_r);
    t_global_2.at<double>(0, 2) = -1 * init_2_x / resolution;
    t_global_2.at<double>(1, 0) = sin(-1 * init_2_r);
    t_global_2.at<double>(1, 1) = cos(-1 * init_2_r);
    t_global_2.at<double>(1, 2) = -1 * init_2_y / resolution;
    t_global_2.at<double>(2, 0) = 0.0;
    t_global_2.at<double>(2, 1) = 0.0;
    t_global_2.at<double>(2, 2) = 1;

    cv::Mat t_0_1 = t_global_1 * t_global_0.inv();
    cv::Mat t_0_2 = t_global_2 * t_global_0.inv();
    cv::Mat t_1_0 = t_0_1.inv();
    cv::Mat t_1_2 = t_global_2 * t_global_1.inv();
    cv::Mat t_2_0 = t_0_2.inv();
    cv::Mat t_2_1 = t_1_2.inv();

    cv::Mat ground_truth_transform, inverse_ground_truth_transform;
    if (tracer_robot == "/tb3_0" && traced_robot == "/tb3_1")
    {
      ground_truth_transform = t_0_1;
      inverse_ground_truth_transform = t_1_0;
    }
    else if (tracer_robot == "/tb3_0" && traced_robot == "/tb3_2")
    {
      ground_truth_transform = t_0_2;
      inverse_ground_truth_transform = t_2_0;
    }
    else if (tracer_robot == "/tb3_1" && traced_robot == "/tb3_0")
    {
      ground_truth_transform = t_1_0;
      inverse_ground_truth_transform = t_0_1;
    }
    else if (tracer_robot == "/tb3_1" && traced_robot == "/tb3_2")
    {
      ground_truth_transform = t_1_2;
      inverse_ground_truth_transform = t_2_1;
    }
    else if (tracer_robot == "/tb3_2" && traced_robot == "/tb3_0")
    {
      ground_truth_transform = t_2_0;
      inverse_ground_truth_transform = t_0_2;
    }
    else if (tracer_robot == "/tb3_2" && traced_robot == "/tb3_1")
    {
      ground_truth_transform = t_2_1;
      inverse_ground_truth_transform = t_1_2;
    }
    //

    {
      std::string s = "";
      for (int y = 0; y < 3; y++)
      {
        for (int x = 0; x < 3; x++)
        {
          double val = proposed.at<double>(y, x);
          if (x == 3 - 1)
          {
            s += std::to_string(val) + "\n";
          }
          else
          {
            s += std::to_string(val) + ", ";
          }
        }
      }
      ROS_INFO("proposed:\n%s", s.c_str());
    }

    {
      std::string s = "";
      for (int y = 0; y < 3; y++)
      {
        for (int x = 0; x < 3; x++)
        {
          double val = ground_truth_transform.at<double>(y, x);
          if (x == 3 - 1)
          {
            s += std::to_string(val) + "\n";
          }
          else
          {
            s += std::to_string(val) + ", ";
          }
        }
      }
      ROS_INFO("ground_truth_transform:\n%s", s.c_str());
    }

    // global origin object w.r.t. frame 0
    cv::Mat pose(3, 1, CV_64F);
    pose.at<double>(0, 0) = pose_x / resolution;
    pose.at<double>(1, 0) = pose_y / resolution;
    pose.at<double>(2, 0) = 1.0;

    cv::Mat predicted_pose = inverse_ground_truth_transform * proposed * pose;
    predicted_pose.at<double>(0, 0) *= resolution;
    predicted_pose.at<double>(1, 0) *= resolution;
    return predicted_pose;
  }

  void Traceback::collectProposingData(double pose_x, double pose_y, double predicted_pose_x, double predicted_pose_y, double score, std::string threshold, std::string tracer_robot, std::string traced_robot, std::string current_time, bool same_interval)
  {
    size_t count = ++pairwise_proposed_count_[tracer_robot][traced_robot][threshold];

    double error = sqrt(pow(predicted_pose_x - pose_x, 2) + pow(predicted_pose_y - pose_y, 2));

    if (same_interval)
    {
      std::ofstream fw(tracer_robot.substr(1) + "_" + traced_robot.substr(1) + "/" + "Error_" + std::to_string(camera_image_update_rate_) + "_rate_" + threshold + "_threshold_" + tracer_robot.substr(1) + "_current_robot_" + traced_robot.substr(1) + "_target_robot.csv", std::ofstream::app);
      if (fw.is_open())
      {
        fw << current_time << "," << count << "," << score << "," << pose_x << "," << pose_y << "," << predicted_pose_x << "," << predicted_pose_y << "," << error << std::endl;
        fw.close();
      }
    }
    else
    {
      std::ofstream fw(tracer_robot.substr(1) + "_" + traced_robot.substr(1) + "/" + "Cf_" + std::to_string(camera_image_update_rate_) + "_rate_" + threshold + "_threshold_" + tracer_robot.substr(1) + "_current_robot_" + traced_robot.substr(1) + "_target_robot.csv", std::ofstream::app);
      if (fw.is_open())
      {
        fw << current_time << "," << count << "," << score << "," << pose_x << "," << pose_y << "," << predicted_pose_x << "," << predicted_pose_y << "," << error << std::endl;
        fw.close();
      }
    }
  }

  void Traceback::writeTracebackFeedbackHistory(std::string tracer, std::string traced, std::string feedback)
  {
    std::string current_time = std::to_string(round(ros::Time::now().toSec() * 100.0) / 100.0);

    std::ofstream fw(tracer.substr(1) + "_" + traced.substr(1) + "/" + "Feedback_history_" + tracer.substr(1) + "_tracer_robot_" + traced.substr(1) + "_traced_robot.txt", std::ofstream::app);
    if (fw.is_open())
    {
      fw << current_time << " - " << feedback << std::endl;
      fw.close();
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

  void Traceback::executePushData()
  {
    ros::Rate r(data_push_rate_);
    while (node_.ok())
    {
      pushData();
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

  void Traceback::executeTransformOptimization()
  {
    ros::Rate r(transform_optimization_rate_);
    while (node_.ok())
    {
      transformOptimization();
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
    std::thread data_push_thr([this]()
                              { executePushData(); });
    std::thread estimation_thr([this]()
                               { executePoseEstimation(); });
    std::thread update_target_thr([this]()
                                  { executeUpdateTargetPoses(); });
    std::thread transform_optimization_thr([this]()
                                           { executeTransformOptimization(); });
    ros::spin();
    transform_optimization_thr.join();
    update_target_thr.join();
    estimation_thr.join();
    data_push_thr.join();
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
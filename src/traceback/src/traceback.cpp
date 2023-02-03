#include <traceback/traceback.h>

#include <thread>
#include <algorithm>
#include <regex>
#include <fstream>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/Marker.h>

namespace traceback
{
  Traceback::Traceback() : map_subscriptions_size_(0), camera_subscriptions_size_(0), tf_listener_(ros::Duration(10.0))
  {
    ros::NodeHandle private_nh("~");

    private_nh.param<std::string>("adjustment_mode", adjustment_mode_, "pointcloud");
    private_nh.param("update_target_rate", update_target_rate_, 0.2);
    private_nh.param("discovery_rate", discovery_rate_, 0.05);
    private_nh.param("estimation_rate", estimation_rate_, 0.5);
    private_nh.param("estimation_confidence", confidence_threshold_, 1.0);
    private_nh.param("essential_mat_confidence", essential_mat_confidence_threshold_, 1.0);
    private_nh.param("accept_count_needed", accept_count_needed_, 8);
    private_nh.param("reject_count_needed", reject_count_needed_, 2);
    private_nh.param("consecutive_abort_count_needed", consecutive_abort_count_needed_, 3);
    private_nh.param<std::string>("robot_map_topic", robot_map_topic_, "map");
    private_nh.param<std::string>("robot_map_updates_topic",
                                  robot_map_updates_topic_, "map_updates");
    private_nh.param<std::string>("robot_namespace", robot_namespace_, "");
    // transform tolerance is used for all tf transforms here
    private_nh.param("transform_tolerance", transform_tolerance_, 0.3);

    private_nh.param<std::string>("camera_image_topic", robot_camera_image_topic_, "camera/rgb/image_raw"); // Don't use image_raw
    private_nh.param<std::string>("camera_point_cloud_topic", robot_camera_point_cloud_topic_, "camera/depth/points");
    private_nh.param("camera_image_update_rate", camera_image_update_rate_, 0.2); // Too high update rate can result in "continue traceback looping"
  }

  void Traceback::tracebackImageAndImageUpdate(const traceback_msgs::ImageAndImage::ConstPtr &msg)
  {
    ROS_INFO("tracebackImageAndImageUpdate");

    assert(adjustment_mode_ == "triangulation" || adjustment_mode_ == "pointcloud");

    if (adjustment_mode_ == "pointcloud")
    {
      std::string tracer_robot = msg->tracer_robot;
      std::string traced_robot = msg->traced_robot;
      double src_map_origin_x = msg->src_map_origin_x;
      double src_map_origin_y = msg->src_map_origin_y;
      double dst_map_origin_x = msg->dst_map_origin_x;
      double dst_map_origin_y = msg->dst_map_origin_y;
      geometry_msgs::Pose arrived_pose = msg->arrived_pose;

      // Mark this min_index as visited so that it will not be repeatedly visited again and again.
      // Also valid for aborted goal
      // Only do for the first traceback
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
        // 1. abort with enough consecutive count
        if (++pairwise_abort_[tracer_robot][traced_robot] >= consecutive_abort_count_needed_)
        {
          writeTracebackFeedbackHistory(tracer_robot, traced_robot, "1. abort with enough consecutive count");

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
          // 1. abort with enough consecutive count END
          return;
        }
        // 2. abort without enough consecutive count
        else
        {
          writeTracebackFeedbackHistory(tracer_robot, traced_robot, "2. abort without enough consecutive count");
          continueTraceback(tracer_robot, traced_robot, src_map_origin_x, src_map_origin_y, dst_map_origin_x, dst_map_origin_y);
          // 2. abort without enough consecutive count END
          return;
        }
      }
      // 3 or 4 or 5 or 6
      else
      {
        pairwise_abort_[tracer_robot][traced_robot] = 0;

        std::string current_time = std::to_string(round(ros::Time::now().toSec() * 100.0) / 100.0);

        geometry_msgs::Quaternion goal_q = arrived_pose.orientation;
        tf2::Quaternion tf_q;
        tf_q.setW(goal_q.w);
        tf_q.setX(goal_q.x);
        tf_q.setY(goal_q.y);
        tf_q.setZ(goal_q.z);
        tf2::Matrix3x3 m(tf_q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        TransformNeeded transform_needed;
        bool is_match = camera_image_processor_.pointCloudMatching(msg->tracer_point_cloud, msg->traced_point_cloud, yaw, transform_needed, tracer_robot, traced_robot, current_time);

        // write images for debug
        {
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
          cv::imwrite(current_time + tracer_robot.substr(1) + "_tracer.png",
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
          cv::imwrite(current_time + traced_robot.substr(1) + "_traced.png",
                      cv_ptr_traced->image);
        }
        // end

        // 3 or 4
        if (is_match)
        {
          // 3. accept
          if (++pairwise_accept_reject_status_[tracer_robot][traced_robot].accept_count >= accept_count_needed_)
          {
            writeTracebackFeedbackHistory(tracer_robot, traced_robot, "3. accept");
            pairwise_accept_reject_status_[tracer_robot][traced_robot].accepted = true;

            // Sort results and only keep the middle values, then take average.
            int history_size = pairwise_triangulation_result_history_[tracer_robot][traced_robot].size();
            double tx_arr[history_size];
            double ty_arr[history_size];
            double r_arr[history_size];
            for (size_t i = 0; i < history_size; ++i)
            {
              TransformAdjustmentResult result = pairwise_triangulation_result_history_[tracer_robot][traced_robot][i];
              {
                std::ofstream fw("Accepted_transform_" + current_time + "_" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot.txt", std::ofstream::app);
                if (fw.is_open())
                {
                  fw << "Index " << i << " transform_needed (tx, ty, r) = (" << result.transform_needed.tx << ", " << result.transform_needed.ty << ", " << result.transform_needed.r << ")" << std::endl;
                  fw.close();
                }
              }
            }
            for (size_t i = 0; i < history_size; ++i)
            {
              TransformAdjustmentResult result = pairwise_triangulation_result_history_[tracer_robot][traced_robot][i];
              cv::Mat adjusted_transform = result.adjusted_transform;
              double tx = adjusted_transform.at<double>(0, 2);
              double ty = adjusted_transform.at<double>(1, 2);
              double r = atan2(adjusted_transform.at<double>(1, 0), adjusted_transform.at<double>(0, 0));

              {
                std::ofstream fw("Accepted_transform_" + current_time + "_" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot.txt", std::ofstream::app);
                if (fw.is_open())
                {
                  fw << "Index " << i << " adjusted transform (tx, ty, r) = (" << tx << ", " << ty << ", " << r << ")" << std::endl;
                  fw.close();
                }
              }

              tx_arr[i] = tx;
              ty_arr[i] = ty;
              r_arr[i] = r;
            }
            std::sort(tx_arr, tx_arr + history_size);
            std::sort(ty_arr, ty_arr + history_size);
            std::sort(r_arr, r_arr + history_size);

            // e.g. 9: keeps 7, 12: keeps 8.
            int number_to_discard_per_side = history_size / 6;
            int number_to_keep = history_size - number_to_discard_per_side * 2;
            int begin_index = (history_size - number_to_keep) / 2;
            // inc
            int end_index = begin_index + number_to_keep - 1;

            double average_tx = 0;
            double average_ty = 0;
            double average_r = 0;
            for (int i = begin_index; i <= end_index; ++i)
            {
              average_tx += tx_arr[i];
              average_ty += ty_arr[i];
              average_r += r_arr[i];
            }
            assert(number_to_keep != 0);
            average_tx /= number_to_keep;
            average_ty /= number_to_keep;
            average_r /= number_to_keep;

            // Compute transformation from average (tx, ty, r)
            cv::Mat average_adjusted_transform(3, 3, CV_64F);
            average_adjusted_transform.at<double>(0, 0) = cos(average_r);
            average_adjusted_transform.at<double>(0, 1) = -sin(average_r);
            average_adjusted_transform.at<double>(0, 2) = average_tx;
            average_adjusted_transform.at<double>(1, 0) = sin(average_r);
            average_adjusted_transform.at<double>(1, 1) = cos(average_r);
            average_adjusted_transform.at<double>(1, 2) = average_ty;
            average_adjusted_transform.at<double>(2, 0) = 0;
            average_adjusted_transform.at<double>(2, 1) = 0;
            average_adjusted_transform.at<double>(2, 2) = 1;

            //
            transform_estimator_.updateBestTransforms(average_adjusted_transform, tracer_robot, traced_robot, best_transforms_, has_best_transforms_);
            //

            {
              std::ofstream fw("Accepted_transform_" + current_time + "_" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot.txt", std::ofstream::app);
              if (fw.is_open())
              {
                fw << "Unadjusted transform:" << std::endl;
                if (history_size != 0)
                {
                  cv::Mat world_transform = pairwise_triangulation_result_history_[tracer_robot][traced_robot][0].world_transform;
                  fw << world_transform.at<double>(0, 0) << "\t" << world_transform.at<double>(0, 1) << "\t" << world_transform.at<double>(0, 2) << std::endl;
                  fw << world_transform.at<double>(1, 0) << "\t" << world_transform.at<double>(1, 1) << "\t" << world_transform.at<double>(1, 2) << std::endl;
                  fw << world_transform.at<double>(2, 0) << "\t" << world_transform.at<double>(2, 1) << "\t" << world_transform.at<double>(2, 2) << std::endl;
                }
                else
                {
                  fw << "None because history size is 0, which should not happen!" << std::endl;
                }
                fw << "Average (tx, ty, r) = (" << average_tx << ", " << average_ty << ", " << average_r << ")" << std::endl;
                fw << "Adjusted transform using average:" << std::endl;
                fw << average_adjusted_transform.at<double>(0, 0) << "\t" << average_adjusted_transform.at<double>(0, 1) << "\t" << average_adjusted_transform.at<double>(0, 2) << std::endl;
                fw << average_adjusted_transform.at<double>(1, 0) << "\t" << average_adjusted_transform.at<double>(1, 1) << "\t" << average_adjusted_transform.at<double>(1, 2) << std::endl;
                fw << average_adjusted_transform.at<double>(2, 0) << "\t" << average_adjusted_transform.at<double>(2, 1) << "\t" << average_adjusted_transform.at<double>(2, 2) << std::endl;
                fw.close();
              }
            }

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
              for (auto it = best_transforms_[tracer_robot].begin(); it != best_transforms_[tracer_robot].end(); ++it)
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
                  std::ofstream fw("Best_transforms_" + current_time + "_" + tracer_robot.substr(1) + "_tracer_robot.txt", std::ofstream::app);
                  if (fw.is_open())
                  {
                    fw << "Best transform from " << tracer_robot << " to " << dst_robot << " :" << std::endl;
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

            robots_to_in_traceback_[tracer_robot] = false;
            // 3. accept END
            return;
          }
          // 4. match but not yet aceept
          else
          {
            writeTracebackFeedbackHistory(tracer_robot, traced_robot, "4. match but not yet aceept");

            {
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

              std::vector<cv::Mat> mat_transforms = robots_src_to_current_transforms_vectors_[tracer_robot][tracer_robot_index];

              // Convert OpenCV transform to world transform, considering the origins
              cv::Mat mat_transform = mat_transforms[traced_robot_index];
              cv::Mat world_transform;
              imageTransformToMapTransform(mat_transform, world_transform, resolutions_[tracer_robot_index], resolutions_[traced_robot_index], src_map_origin_x, src_map_origin_y, dst_map_origin_x, dst_map_origin_y);
              //

              cv::Mat adjusted_transform;
              double length_of_translation = sqrt(transform_needed.tx * transform_needed.tx + transform_needed.ty * transform_needed.ty);
              findAdjustedTransformation(world_transform, adjusted_transform, length_of_translation, transform_needed.tx / length_of_translation, transform_needed.ty / length_of_translation, transform_needed.r, arrived_pose.position.x, arrived_pose.position.y, resolutions_[tracer_robot_index]);
              TransformAdjustmentResult result;
              result.current_time = current_time;
              result.transform_needed = transform_needed;
              result.world_transform = world_transform;
              result.adjusted_transform = adjusted_transform;
              pairwise_triangulation_result_history_[tracer_robot][traced_robot].push_back(result);
            }

            continueTraceback(tracer_robot, traced_robot, src_map_origin_x, src_map_origin_y, dst_map_origin_x, dst_map_origin_y);
            // 4. match but not yet aceept END
            return;
          }
        }
        // Does not match
        // 5 or 6
        else
        {
          // 5. reject
          if (++pairwise_accept_reject_status_[tracer_robot][traced_robot].reject_count >= reject_count_needed_ && pairwise_accept_reject_status_[tracer_robot][traced_robot].reject_count >= 2 * pairwise_accept_reject_status_[tracer_robot][traced_robot].accept_count)
          {
            writeTracebackFeedbackHistory(tracer_robot, traced_robot, "5. reject");
            pairwise_accept_reject_status_[tracer_robot][traced_robot].accept_count = 0;
            pairwise_accept_reject_status_[tracer_robot][traced_robot].reject_count = 0;
            robots_to_in_traceback_[tracer_robot] = false;
            // 5. reject END
            return;
          }
          // 6. does not match but not yet reject
          else
          {
            writeTracebackFeedbackHistory(tracer_robot, traced_robot, "6. does not match but not yet reject");
            continueTraceback(tracer_robot, traced_robot, src_map_origin_x, src_map_origin_y, dst_map_origin_x, dst_map_origin_y);
            // 6. does not match but not yet reject END
            return;
          }
        }
      }
    }
    /*




    */
    else if (adjustment_mode_ == "triangulation")
    {
      std::string tracer_robot = msg->tracer_robot;
      std::string traced_robot = msg->traced_robot;
      double src_map_origin_x = msg->src_map_origin_x;
      double src_map_origin_y = msg->src_map_origin_y;
      double dst_map_origin_x = msg->dst_map_origin_x;
      double dst_map_origin_y = msg->dst_map_origin_y;
      geometry_msgs::Pose arrived_pose = msg->arrived_pose;

      // 1 or 2 or 3 or 4 or 5
      if (!msg->second_traceback)
      {
        ROS_INFO("First traceback done for tracer %s", tracer_robot.c_str());
        ROS_INFO("Abort = %s", msg->aborted ? "true" : "false");

        {
          ROS_DEBUG("Arrived position (x, y) = (%f, %f)", arrived_pose.position.x, arrived_pose.position.y);
          ROS_DEBUG("Arrived orientation (x, y, z, w) = (%f, %f, %f, %f)", arrived_pose.orientation.x, arrived_pose.orientation.y, arrived_pose.orientation.z, arrived_pose.orientation.w);
        }

        // Mark this min_index as visited so that it will not be repeatedly visited again and again.
        // Also valid for aborted goal
        // Only do for the first traceback
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
          // 1. first traceback, abort with enough consecutive count
          if (++pairwise_abort_[tracer_robot][traced_robot] >= consecutive_abort_count_needed_)
          {
            writeTracebackFeedbackHistory(tracer_robot, traced_robot, "1. first traceback, abort with enough consecutive count");

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
            // 1. first traceback, abort with enough consecutive count END
            return;
          }
          // 2. first traceback, abort without enough consecutive count
          else
          {
            writeTracebackFeedbackHistory(tracer_robot, traced_robot, "2. first traceback, abort without enough consecutive count");
            continueTraceback(tracer_robot, traced_robot, src_map_origin_x, src_map_origin_y, dst_map_origin_x, dst_map_origin_y);
            // 2. first traceback, abort without enough consecutive count END
            return;
          }
        }
        // 3 or 4 or 5
        else
        {
          pairwise_abort_[tracer_robot][traced_robot] = 0;

          // Get cv images and analyze (first traceback, reflect in image filenames)
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
          cv::imwrite(current_time + "_first_traceback_" + tracer_robot.substr(1) + "_tracer.png",
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
          cv::imwrite(current_time + "_first_traceback_" + traced_robot.substr(1) + "_traced.png",
                      cv_ptr_traced->image);

          geometry_msgs::Quaternion goal_q = arrived_pose.orientation;
          tf2::Quaternion tf_q;
          tf_q.setW(goal_q.w);
          tf_q.setX(goal_q.x);
          tf_q.setY(goal_q.y);
          tf_q.setZ(goal_q.z);
          tf2::Matrix3x3 m(tf_q);
          double roll, pitch, yaw;
          m.getRPY(roll, pitch, yaw);

          TransformNeeded transform_needed;
          bool is_unwanted_translation_angle = false;
          bool is_match = camera_image_processor_.findFurtherTransformNeeded(cv_ptr_tracer->image, cv_ptr_traced->image, FeatureType::SURF,
                                                                             essential_mat_confidence_threshold_, yaw, transform_needed, is_unwanted_translation_angle, tracer_robot, traced_robot, current_time);
          // Get cv images and analyze END

          if (is_match)
          {
            // 10. first traceback, match but unwanted translation angle
            if (is_unwanted_translation_angle)
            {
              writeTracebackFeedbackHistory(tracer_robot, traced_robot, "10. first traceback, match but unwanted translation angle");
              continueTraceback(tracer_robot, traced_robot, src_map_origin_x, src_map_origin_y, dst_map_origin_x, dst_map_origin_y);
              // 10. first traceback, match but unwanted translation angle END
              return;
            }
            // 3. first traceback, match
            else
            {
              writeTracebackFeedbackHistory(tracer_robot, traced_robot, "3. first traceback, match");
              // Set new triangulation first result
              {
                FirstTracebackResult result;
                result.first_x = arrived_pose.position.x;
                result.first_y = arrived_pose.position.y;
                result.first_tracer_to_traced_tx = transform_needed.tx;
                result.first_tracer_to_traced_ty = transform_needed.ty;
                result.first_tracer_to_traced_r = transform_needed.r;
                pairwise_first_traceback_result_[tracer_robot][traced_robot] = result;
              }

              // Second traceback: after each match, navigate to a nearby location in order to
              // estimate the absolute scale

              move_base_msgs::MoveBaseGoal goal;
              geometry_msgs::PoseStamped new_pose_stamped;
              new_pose_stamped.pose = arrived_pose;

              // Rotate by 30 degree
              // geometry_msgs::Quaternion goal_q = arrived_pose.orientation;
              // geometry_msgs::Quaternion transform_q;
              // tf2::Quaternion tf2_goal_q;
              // tf2_goal_q.setW(goal_q.w);
              // tf2_goal_q.setX(goal_q.x);
              // tf2_goal_q.setY(goal_q.y);
              // tf2_goal_q.setZ(goal_q.z);
              // tf2::Quaternion tf2_transform_q;
              // tf2_transform_q.setW(0.9659258 );
              // tf2_transform_q.setX(0.0);
              // tf2_transform_q.setY(0.0);
              // tf2_transform_q.setZ(0.258819);
              // tf2::Quaternion tf2_new_q = tf2_transform_q * tf2_goal_q;
              // geometry_msgs::Quaternion new_q;
              // new_q.w = tf2_new_q.w();
              // new_q.x = tf2_new_q.x();
              // new_q.y = tf2_new_q.y();
              // new_q.z = tf2_new_q.z();
              // new_pose_stamped.pose.orientation = new_q;
              //

              double distance = 0.25;
              new_pose_stamped.pose.position.x -= cos(yaw) * distance;
              new_pose_stamped.pose.position.y -= sin(yaw) * distance;
              new_pose_stamped.header.frame_id = tracer_robot.substr(1) + tracer_robot + "/map";
              new_pose_stamped.header.stamp = ros::Time::now();
              goal.target_pose = new_pose_stamped;

              traceback_msgs::GoalAndImage goal_and_image;
              goal_and_image.goal = goal;
              goal_and_image.image = msg->traced_image;
              goal_and_image.tracer_robot = tracer_robot;
              goal_and_image.traced_robot = traced_robot;
              goal_and_image.src_map_origin_x = src_map_origin_x;
              goal_and_image.src_map_origin_y = src_map_origin_y;
              goal_and_image.dst_map_origin_x = dst_map_origin_x;
              goal_and_image.dst_map_origin_y = dst_map_origin_y;
              goal_and_image.stamp = msg->stamp;
              goal_and_image.second_traceback = true;
              ROS_DEBUG("Goal and image to be sent");

              /** Visualize goal in src robot frame */
              visualizeGoal(new_pose_stamped, tracer_robot);
              /** Visualize goal in src robot frame END */
              robots_to_goal_and_image_publisher_[tracer_robot].publish(goal_and_image);

              // 3. first traceback, match END
              return;
            }
          }
          // Reject if first traceback does not match
          // 4 or 5
          else
          {
            // Update AcceptRejectStatus
            // Reject when reject count is at least reject_count_needed_ and reject count is at least two times accept count
            // 4. first traceback, reject
            if (++pairwise_accept_reject_status_[tracer_robot][traced_robot].reject_count >= reject_count_needed_ && pairwise_accept_reject_status_[tracer_robot][traced_robot].reject_count >= 2 * pairwise_accept_reject_status_[tracer_robot][traced_robot].accept_count)
            {
              writeTracebackFeedbackHistory(tracer_robot, traced_robot, "4. first traceback, reject");
              pairwise_accept_reject_status_[tracer_robot][traced_robot].accept_count = 0;
              pairwise_accept_reject_status_[tracer_robot][traced_robot].reject_count = 0;
              robots_to_in_traceback_[tracer_robot] = false;
              // 4. first traceback, reject END
              return;
            }
            else
            {
              writeTracebackFeedbackHistory(tracer_robot, traced_robot, "5. first traceback, does not match but not yet reject");
              // Assume does not have to pop_front the list first
              // 5. first traceback, does not match but not yet reject
              continueTraceback(tracer_robot, traced_robot, src_map_origin_x, src_map_origin_y, dst_map_origin_x, dst_map_origin_y);
              // 5. first traceback, does not match but not yet reject END
              return;
            }
          }
        }
      }
      // 6 or 7 or 8 or 9
      else
      {
        ROS_INFO("Second traceback done for tracer %s", tracer_robot.c_str());
        ROS_INFO("Abort = %s", msg->aborted ? "true" : "false");

        {
          ROS_DEBUG("Arrived position (x, y) = (%f, %f)", arrived_pose.position.x, arrived_pose.position.y);
          ROS_DEBUG("Arrived orientation (x, y, z, w) = (%f, %f, %f, %f)", arrived_pose.orientation.x, arrived_pose.orientation.y, arrived_pose.orientation.z, arrived_pose.orientation.w);
        }

        // 6. second traceback, abort
        if (msg->aborted)
        {
          writeTracebackFeedbackHistory(tracer_robot, traced_robot, "6. second traceback, abort");
          continueTraceback(tracer_robot, traced_robot, src_map_origin_x, src_map_origin_y, dst_map_origin_x, dst_map_origin_y);
          // 6. second traceback, abort END
          return;
        }
        // 7 or 8 or 9

        // Get cv images and analyze (second traceback, reflect in image filenames)
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
        cv::imwrite(current_time + "_second_traceback_" + tracer_robot.substr(1) + "_tracer.png",
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
        cv::imwrite(current_time + "_second_traceback_" + traced_robot.substr(1) + "_traced.png",
                    cv_ptr_traced->image);

        geometry_msgs::Quaternion goal_q = arrived_pose.orientation;
        tf2::Quaternion tf_q;
        tf_q.setW(goal_q.w);
        tf_q.setX(goal_q.x);
        tf_q.setY(goal_q.y);
        tf_q.setZ(goal_q.z);
        tf2::Matrix3x3 m(tf_q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

        TransformNeeded transform_needed;
        bool is_unwanted_translation_angle = false;
        bool is_match = camera_image_processor_.findFurtherTransformNeeded(cv_ptr_tracer->image, cv_ptr_traced->image, FeatureType::SURF,
                                                                           essential_mat_confidence_threshold_, yaw, transform_needed, is_unwanted_translation_angle, tracer_robot, traced_robot, current_time);
        // Get cv images and analyze END

        // 7 or 8
        if (is_match)
        {
          // If accepted, no further traceback is needed for this ordered pair,
          // but remember to enable other tracebacks for the same tracer
          // 7. second traceback, accept
          if (++pairwise_accept_reject_status_[tracer_robot][traced_robot].accept_count >= accept_count_needed_)
          {
            writeTracebackFeedbackHistory(tracer_robot, traced_robot, "7. second traceback, accept");
            pairwise_accept_reject_status_[tracer_robot][traced_robot].accepted = true;

            // Sort results and only keep the middle values, then take average.
            int history_size = pairwise_triangulation_result_history_[tracer_robot][traced_robot].size();
            double tx_arr[history_size];
            double ty_arr[history_size];
            double r_arr[history_size];
            for (size_t i = 0; i < history_size; ++i)
            {
              TransformAdjustmentResult result = pairwise_triangulation_result_history_[tracer_robot][traced_robot][i];
              cv::Mat adjusted_transform = result.adjusted_transform;
              double tx = adjusted_transform.at<double>(0, 2);
              double ty = adjusted_transform.at<double>(1, 2);
              double r = atan2(adjusted_transform.at<double>(1, 0), adjusted_transform.at<double>(0, 0));

              {
                std::ofstream fw("Accepted_transform_" + current_time + "_" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot.txt", std::ofstream::app);
                if (fw.is_open())
                {
                  fw << "Index " << i << " adjusted transform (tx, ty, r) = (" << tx << ", " << ty << ", " << r << ")" << std::endl;
                  fw.close();
                }
              }

              tx_arr[i] = tx;
              ty_arr[i] = ty;
              r_arr[i] = r;
            }
            std::sort(tx_arr, tx_arr + history_size);
            std::sort(ty_arr, ty_arr + history_size);
            std::sort(r_arr, r_arr + history_size);

            // e.g. 9: keeps 5, 10: keeps 6.
            int number_to_keep = history_size / 2 + 1;
            int begin_index = (history_size - number_to_keep) / 2;
            // inc
            int end_index = begin_index + number_to_keep - 1;

            double average_tx = 0;
            double average_ty = 0;
            double average_r = 0;
            for (int i = begin_index; i <= end_index; ++i)
            {
              average_tx += tx_arr[i];
              average_ty += ty_arr[i];
              average_r += r_arr[i];
            }
            assert(number_to_keep != 0);
            average_tx /= number_to_keep;
            average_ty /= number_to_keep;
            average_r /= number_to_keep;

            // Compute transformation from average (tx, ty, r)
            cv::Mat average_adjusted_transform(3, 3, CV_64F);
            average_adjusted_transform.at<double>(0, 0) = cos(average_r);
            average_adjusted_transform.at<double>(0, 1) = -sin(average_r);
            average_adjusted_transform.at<double>(0, 2) = average_tx;
            average_adjusted_transform.at<double>(1, 0) = sin(average_r);
            average_adjusted_transform.at<double>(1, 1) = cos(average_r);
            average_adjusted_transform.at<double>(1, 2) = average_ty;
            average_adjusted_transform.at<double>(2, 0) = 0;
            average_adjusted_transform.at<double>(2, 1) = 0;
            average_adjusted_transform.at<double>(2, 2) = 1;

            //
            transform_estimator_.updateBestTransforms(average_adjusted_transform, tracer_robot, traced_robot, best_transforms_, has_best_transforms_);
            //

            {
              std::ofstream fw("Accepted_transform_" + current_time + "_" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot.txt", std::ofstream::app);
              if (fw.is_open())
              {
                fw << "Unadjusted transform:" << std::endl;
                if (history_size != 0)
                {
                  cv::Mat world_transform = pairwise_triangulation_result_history_[tracer_robot][traced_robot][0].world_transform;
                  fw << world_transform.at<double>(0, 0) << "\t" << world_transform.at<double>(0, 1) << "\t" << world_transform.at<double>(0, 2) << std::endl;
                  fw << world_transform.at<double>(1, 0) << "\t" << world_transform.at<double>(1, 1) << "\t" << world_transform.at<double>(1, 2) << std::endl;
                  fw << world_transform.at<double>(2, 0) << "\t" << world_transform.at<double>(2, 1) << "\t" << world_transform.at<double>(2, 2) << std::endl;
                }
                else
                {
                  fw << "None because history size is 0, which should not happen!" << std::endl;
                }
                fw << "Average (tx, ty, r) = (" << average_tx << ", " << average_ty << ", " << average_r << ")" << std::endl;
                fw << "Adjusted transform using average:" << std::endl;
                fw << average_adjusted_transform.at<double>(0, 0) << "\t" << average_adjusted_transform.at<double>(0, 1) << "\t" << average_adjusted_transform.at<double>(0, 2) << std::endl;
                fw << average_adjusted_transform.at<double>(1, 0) << "\t" << average_adjusted_transform.at<double>(1, 1) << "\t" << average_adjusted_transform.at<double>(1, 2) << std::endl;
                fw << average_adjusted_transform.at<double>(2, 0) << "\t" << average_adjusted_transform.at<double>(2, 1) << "\t" << average_adjusted_transform.at<double>(2, 2) << std::endl;
                fw.close();
              }
            }

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
              for (auto it = best_transforms_[tracer_robot].begin(); it != best_transforms_[tracer_robot].end(); ++it)
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
                  std::ofstream fw("Best_transforms_" + current_time + "_" + tracer_robot.substr(1) + "_tracer_robot.txt", std::ofstream::app);
                  if (fw.is_open())
                  {
                    fw << "Best transform from " << tracer_robot << " to " << dst_robot << " :" << std::endl;
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
            // TEST HARDCODE sending traceback transforms END

            robots_to_in_traceback_[tracer_robot] = false;
            // 7. second traceback, accept END
            return;
          }
          // Add this triangulation result to triangulation result history
          // 8. second traceback, match but not yet aceept
          else
          {
            writeTracebackFeedbackHistory(tracer_robot, traced_robot, "8. second traceback, match but not yet aceept");

            FirstTracebackResult result = pairwise_first_traceback_result_[tracer_robot][traced_robot];
            double first_x = result.first_x;
            double first_y = result.first_y;
            double first_tracer_to_traced_tx = result.first_tracer_to_traced_tx;
            double first_tracer_to_traced_ty = result.first_tracer_to_traced_ty;
            double second_x = arrived_pose.position.x;
            double second_y = arrived_pose.position.y;
            double second_tracer_to_traced_tx = transform_needed.tx;
            double second_tracer_to_traced_ty = transform_needed.ty;

            double length_of_translation = findLengthOfTranslationByTriangulation(first_x, first_y, first_tracer_to_traced_tx, first_tracer_to_traced_ty, second_x, second_y, second_tracer_to_traced_tx, second_tracer_to_traced_ty);
            double first_tracer_to_traced_r = result.first_tracer_to_traced_r;
            {
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

              std::vector<cv::Mat> mat_transforms = robots_src_to_current_transforms_vectors_[tracer_robot][tracer_robot_index];

              // Convert OpenCV transform to world transform, considering the origins
              cv::Mat mat_transform = mat_transforms[traced_robot_index];
              cv::Mat world_transform;
              imageTransformToMapTransform(mat_transform, world_transform, resolutions_[tracer_robot_index], resolutions_[traced_robot_index], src_map_origin_x, src_map_origin_y, dst_map_origin_x, dst_map_origin_y);
              //

              cv::Mat adjusted_transform;
              findAdjustedTransformation(world_transform, adjusted_transform, length_of_translation, first_tracer_to_traced_tx, first_tracer_to_traced_ty, first_tracer_to_traced_r, first_x, first_y, resolutions_[tracer_robot_index]);
              TransformAdjustmentResult result;
              result.world_transform = world_transform;
              result.adjusted_transform = adjusted_transform;
              pairwise_triangulation_result_history_[tracer_robot][traced_robot].push_back(result);
            }

            continueTraceback(tracer_robot, traced_robot, src_map_origin_x, src_map_origin_y, dst_map_origin_x, dst_map_origin_y);
            // 8. second traceback, match but not yet aceept END
            return;
          }
        }
        // 9. second traceback, does not match
        else
        {
          writeTracebackFeedbackHistory(tracer_robot, traced_robot, "9. second traceback, does not match");
          continueTraceback(tracer_robot, traced_robot, src_map_origin_x, src_map_origin_y, dst_map_origin_x, dst_map_origin_y);
          // 9. second traceback, does not match END
          return;
        }
      }
    }
  }

  void Traceback::continueTraceback(std::string tracer_robot, std::string traced_robot, double src_map_origin_x, double src_map_origin_y, double dst_map_origin_x, double dst_map_origin_y)
  {
    std::string robot_name_src = tracer_robot;
    std::string robot_name_dst = traced_robot;
    ROS_INFO("Continue traceback process for robot %s", robot_name_src.c_str());

    std::list<PoseImagePair>::iterator temp = robots_to_current_it_[robot_name_src];

    bool whole_list_visited = false;
    bool pass_end = false;
    ROS_DEBUG("Stamp is %ld", temp->stamp);
    // ++temp;
    // if (temp == camera_image_processor_.robots_to_all_pose_image_pairs_[robot_name_dst].end())
    // {
    //   temp = camera_image_processor_.robots_to_all_pose_image_pairs_[robot_name_dst].begin();
    //   pass_end = true;
    // }
    while (camera_image_processor_.robots_to_all_visited_pose_image_pair_indexes_[robot_name_dst].count(temp->stamp))
    {
      ++temp;
      if (temp == camera_image_processor_.robots_to_all_pose_image_pairs_[robot_name_dst].end())
      {
        temp = camera_image_processor_.robots_to_all_pose_image_pairs_[robot_name_dst].begin();

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

    std::vector<std::vector<cv::Mat>> transforms_vectors;
    std::vector<cv::Point2d> map_origins;
    std::vector<cv::Point2f> centers;
    std::vector<std::vector<double>> confidences;
    // Ensure consistency of transforms_vectors_, centers_ and confidences_
    {
      boost::shared_lock<boost::shared_mutex> lock(transform_estimator_.updates_mutex_);
      transforms_vectors = transform_estimator_.getTransformsVectors();
      // transform_estimator_.printTransformsVectors(transforms_vectors);

      map_origins = map_origins_;

      centers = transform_estimator_.getCenters();
      for (auto &p : centers)
      {
        // ROS_INFO("center = (%f, %f)", p.x, p.y);
      }

      confidences = transform_estimator_.getConfidences();
      // transform_estimator_.printConfidences(confidences);
    }

    for (size_t i = 0; i < confidences.size(); ++i)
    {
      std::string robot_name_src = transforms_indexes_[i];
      // Find it, filtering accepted tracebacks
      // auto it = max_element(confidences[i].begin(), confidences[i].end());
      std::vector<double>::iterator it = confidences[i].begin();
      bool found = false;
      double max_confidence = -1.0;
      size_t dst = 0;
      for (auto itt = confidences[i].begin(); itt != confidences[i].end(); ++itt)
      {
        if (pairwise_accept_reject_status_[robot_name_src][transforms_indexes_[dst]].accepted)
        {
          ++dst;
          ROS_INFO("Already accepted.");
          continue;
        }
        if (pairwise_paused_[robot_name_src][transforms_indexes_[dst]])
        {
          ++dst;
          ROS_INFO("Being paused.");
          continue;
        }
        if (*itt > max_confidence)
        {
          it = itt;
          max_confidence = *itt;
          found = true;
        }
        ++dst;
      }

      if (!found)
      {
        continue;
      }

      // Find it END
      size_t max_position = it - confidences[i].begin();
      std::string robot_name_dst = transforms_indexes_[max_position];

      // ROS_INFO("confidences[%zu] (max_position, max_confidence) = (%zu, %f)", i, max_position, max_confidence);

      // No transform that passes confidence_threshold_ exists
      if (abs(max_confidence - 0.0) < transform_estimator_.ZERO_ERROR)
      {
        continue;
      }

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
      pose_src.at<double>(0, 0) = (pose.position.x - src_map_origin_x) / resolutions_[i];
      pose_src.at<double>(1, 0) = (pose.position.y - src_map_origin_y) / resolutions_[i];
      pose_src.at<double>(2, 0) = 1.0;

      cv::Mat pose_dst = transforms_vectors[i][max_position] * pose_src;
      pose_dst.at<double>(0, 0) *= resolutions_[max_position];
      pose_dst.at<double>(1, 0) *= resolutions_[max_position];
      pose_dst.at<double>(0, 0) += src_map_origin_x;
      pose_dst.at<double>(1, 0) += src_map_origin_y;
      // Also adjust the difference between the origins
      pose_dst.at<double>(0, 0) += dst_map_origin_x;
      pose_dst.at<double>(1, 0) += dst_map_origin_y;
      pose_dst.at<double>(0, 0) -= src_map_origin_x;
      pose_dst.at<double>(1, 0) -= src_map_origin_y;

      // ROS_INFO("transformed pose (x, y) = (%f, %f)", pose_dst.at<double>(0, 0), pose_dst.at<double>(1, 0));

      if (robots_to_in_traceback_[robot_name_src])
      {
        continue; // continue to next robot since the current robot is currently in traceback process
      }
      else
      {
        robots_to_in_traceback_[robot_name_src] = true;
      }

      ROS_INFO("Start traceback process for robot %s", robot_name_src.c_str());
      ROS_INFO("confidences[%zu] (max_position, max_confidence) = (%zu, %f)", i, max_position, max_confidence);
      ROS_INFO("{%s} pose %zu (x, y) = (%f, %f)", robot_name_src.c_str(), i, pose.position.x, pose.position.y);

      ROS_INFO("transforms[%s][%s] (width, height) = (%d, %d)", robot_name_src.c_str(), robot_name_dst.c_str(), transforms_vectors[i][max_position].cols, transforms_vectors[i][max_position].rows);

      int width = transforms_vectors[i][max_position].cols;
      int height = transforms_vectors[i][max_position].rows;
      std::string s = "";
      for (int y = 0; y < height; y++)
      {
        for (int x = 0; x < width; x++)
        {
          double val = transforms_vectors[i][max_position].at<double>(y, x);
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

      // This is only updated here (start/restart traceback)
      robots_src_to_current_transforms_vectors_[robot_name_src] = transforms_vectors;

      // Clear triangulation history
      pairwise_triangulation_result_history_[robot_name_src][robot_name_dst].clear();

      /** just for finding min_it */
      double min_distance = DBL_MAX;
      std::list<PoseImagePair>::iterator min_it = camera_image_processor_.robots_to_all_pose_image_pairs_[robot_name_dst].begin();

      for (auto it = camera_image_processor_.robots_to_all_pose_image_pairs_[robot_name_dst].begin(); it != camera_image_processor_.robots_to_all_pose_image_pairs_[robot_name_dst].end(); ++it)
      {
        if (camera_image_processor_.robots_to_all_visited_pose_image_pair_indexes_[robot_name_dst].count(it->stamp))
        {
          continue;
        }

        double dst_x = it->pose.position.x;
        double dst_y = it->pose.position.y;
        double src_x = pose_dst.at<double>(0, 0);
        double src_y = pose_dst.at<double>(1, 0);
        double distance = sqrt(pow(dst_x - src_x, 2) + pow(dst_y - src_y, 2));
        if (distance < min_distance)
        {
          min_distance = distance;
          min_it = it;
        }
      }

      robots_to_current_it_[robot_name_src] = min_it;
      /** just for finding min_it END */

      startOrContinueTraceback(robot_name_src, robot_name_dst, src_map_origin_x, src_map_origin_y, dst_map_origin_x, dst_map_origin_y);
    }
  }

  void Traceback::startOrContinueTraceback(std::string robot_name_src, std::string robot_name_dst, double src_map_origin_x, double src_map_origin_y, double dst_map_origin_x, double dst_map_origin_y)
  {
    /** Get parameters other than robot names */
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

    std::vector<std::vector<cv::Mat>> transforms_vectors = robots_src_to_current_transforms_vectors_[robot_name_src];
    /** Get parameters other than robot names END */

    std::list<PoseImagePair>::iterator current_it = robots_to_current_it_[robot_name_src];

    double goal_x = current_it->pose.position.x;
    double goal_y = current_it->pose.position.y;

    // Transform goal from dst frame to src (robot i) frame
    // Same as above, it is required to manually rotate about the bottom-left corner, which
    // is (-20m, -20m) or (-400px, -400px) when the resolution is 0.05.
    cv::Mat goal_dst(3, 1, CV_64F);
    goal_dst.at<double>(0, 0) = (goal_x - dst_map_origin_x) / resolutions_[max_position];
    goal_dst.at<double>(1, 0) = (goal_y - dst_map_origin_y) / resolutions_[max_position];
    goal_dst.at<double>(2, 0) = 1.0;

    cv::Mat goal_src = transforms_vectors[max_position][i] * goal_dst;
    goal_src.at<double>(0, 0) *= resolutions_[i];
    goal_src.at<double>(1, 0) *= resolutions_[i];
    goal_src.at<double>(0, 0) += dst_map_origin_x;
    goal_src.at<double>(1, 0) += dst_map_origin_y;
    // Also adjust the difference between the origins
    goal_src.at<double>(0, 0) += src_map_origin_x;
    goal_src.at<double>(1, 0) += src_map_origin_y;
    goal_src.at<double>(0, 0) -= dst_map_origin_x;
    goal_src.at<double>(1, 0) -= dst_map_origin_y;

    ROS_INFO("transformed goal_src (x, y) = (%f, %f)", goal_src.at<double>(0, 0), goal_src.at<double>(1, 0));

    geometry_msgs::Point target_position;
    target_position.x = goal_src.at<double>(0, 0);
    target_position.y = goal_src.at<double>(1, 0);
    target_position.z = 0.0f;

    // Transform rotation
    // Note that due to scaling, the "rotation matrix" values can exceed 1, and therefore need to normalize it.
    geometry_msgs::Quaternion goal_q = current_it->pose.orientation;
    geometry_msgs::Quaternion transform_q;
    cv::Mat transform = transforms_vectors[max_position][i];
    matToQuaternion(transform, transform_q);
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
    goal_and_image.image = current_it->image;
    goal_and_image.point_cloud = current_it->point_cloud;
    goal_and_image.tracer_robot = robot_name_src;
    goal_and_image.traced_robot = robot_name_dst;
    goal_and_image.src_map_origin_x = src_map_origin_x;
    goal_and_image.src_map_origin_y = src_map_origin_y;
    goal_and_image.dst_map_origin_x = dst_map_origin_x;
    goal_and_image.dst_map_origin_y = dst_map_origin_y;
    goal_and_image.second_traceback = false;
    goal_and_image.stamp = current_it->stamp;
    ROS_DEBUG("Goal and image to be sent");

    /** Visualize goal in src robot frame */
    visualizeGoal(goal.target_pose, robot_name_src);
    /** Visualize goal in src robot frame END */
    robots_to_goal_and_image_publisher_[robot_name_src].publish(goal_and_image);
  }

  void Traceback::visualizeGoal(geometry_msgs::PoseStamped pose_stamped, std::string robot_name)
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
    m.ns = "traceback_goal";
    m.scale.x = 0.2;
    m.scale.y = 0.2;
    m.scale.z = 0.2;
    // lives forever
    m.lifetime = ros::Duration(0);
    m.frame_locked = true;

    m.action = visualization_msgs::Marker::ADD;
    m.type = visualization_msgs::Marker::ARROW;
    size_t id = 0;
    m.id = int(id);
    m.color = red;
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
    ROS_DEBUG("Receive updated camera image started.");
    for (auto current : camera_image_processor_.robots_to_current_image_)
    {
      std::string robot_name = current.first;
      geometry_msgs::Pose pose = camera_image_processor_.robots_to_current_pose_[current.first];
      sensor_msgs::PointCloud2 point_cloud = camera_image_processor_.robots_to_current_point_cloud_[current.first];
      PoseImagePair pose_image_pair;
      pose_image_pair.pose = pose;
      pose_image_pair.image = current.second;
      pose_image_pair.point_cloud = point_cloud;
      pose_image_pair.stamp = ros::Time::now().toNSec();

      auto all = camera_image_processor_.robots_to_all_pose_image_pairs_.find(current.first);
      if (all != camera_image_processor_.robots_to_all_pose_image_pairs_.end())
      {
        all->second.emplace_back(pose_image_pair);
        PoseImagePair latestPair = *std::max_element(camera_image_processor_.robots_to_all_pose_image_pairs_[current.first].begin(), camera_image_processor_.robots_to_all_pose_image_pairs_[current.first].end());
        // ROS_INFO("latestPair.stamp: %ld", latestPair.stamp);
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
    // TODO transforms_indexes_ can have concurrency problem
    transforms_indexes_.clear();
    resolutions_.clear();
    map_origins_.clear();
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
        map_origins_.emplace_back(cv::Point2d(subscription.readonly_map->info.origin.position.x, subscription.readonly_map->info.origin.position.y));
        ++i;
      }
    }

    transform_estimator_.feed(grids.begin(), grids.end());
    transform_estimator_.estimateTransforms(FeatureType::AKAZE,
                                            confidence_threshold_);
  }

  void Traceback::CameraImageUpdate(const sensor_msgs::ImageConstPtr &msg)
  {
    std::string frame_id = msg->header.frame_id;
    std::string robot_name = "/" + frame_id.substr(0, frame_id.find("/"));
    camera_image_processor_.robots_to_temp_image_[robot_name] = *msg;
  }

  void Traceback::CameraPointCloudUpdate(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    std::string frame_id = msg->header.frame_id;
    std::string robot_name = "/" + frame_id.substr(0, frame_id.find("/"));
    camera_image_processor_.robots_to_current_point_cloud_[robot_name] = *msg;
    camera_image_processor_.robots_to_current_pose_[robot_name] = getRobotPose(robot_name);
    camera_image_processor_.robots_to_current_image_[robot_name] = camera_image_processor_.robots_to_temp_image_[robot_name];
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
        std::string camera_rgb_topic;
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
        camera_point_cloud_topic = ros::names::append(robot_name, robot_camera_point_cloud_topic_);

        subscription.robot_namespace = robot_name;

        ROS_INFO("Subscribing to CAMERA rgb topic: %s.", camera_rgb_topic.c_str());
        ROS_INFO("Subscribing to CAMERA point cloud topic: %s.", camera_point_cloud_topic.c_str());

        // Insert empty std::vector to the map to prevent future error when accessing the map by robot name.
        auto it = camera_image_processor_.robots_to_all_pose_image_pairs_.find(subscription.robot_namespace);
        if (it == camera_image_processor_.robots_to_all_pose_image_pairs_.end())
        {
          camera_image_processor_.robots_to_all_pose_image_pairs_.insert({subscription.robot_namespace, {}});
        }

        subscription.camera_rgb_sub.subscribe(node_,
                                              camera_rgb_topic, 10);

        subscription.camera_point_cloud_sub.subscribe(node_,
                                                      camera_point_cloud_topic, 10);

        subscription.camera_rgb_sub.registerCallback(boost::bind(&Traceback::CameraImageUpdate, this, _1));
        subscription.camera_point_cloud_sub.registerCallback(boost::bind(&Traceback::CameraPointCloudUpdate, this, _1));

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
        robots_to_current_it_.emplace(robot_name, nullptr);

        traceback_transforms_publisher_ = node_.advertise<traceback_msgs::TracebackTransforms>(traceback_transforms_topic_, 10);

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
        }
      }
    }
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

  double Traceback::findLengthOfTranslationByTriangulation(double first_x, double first_y, double first_tracer_to_traced_tx, double first_tracer_to_traced_ty, double second_x, double second_y, double second_tracer_to_traced_tx, double second_tracer_to_traced_ty)
  {
    double first_transformation_distance = sqrt(first_tracer_to_traced_tx * first_tracer_to_traced_tx + first_tracer_to_traced_ty * first_tracer_to_traced_ty);
    double second_transformation_distance = sqrt(second_tracer_to_traced_tx * second_tracer_to_traced_tx + second_tracer_to_traced_ty * second_tracer_to_traced_ty);
    double angle_between_two_camera_transformations = acos((first_tracer_to_traced_tx * second_tracer_to_traced_tx + first_tracer_to_traced_ty * second_tracer_to_traced_ty) / (first_transformation_distance * second_transformation_distance));
    double second_to_first_x = first_x - second_x;
    double second_to_first_y = first_y - second_y;
    double movement_distance = sqrt(second_to_first_x * second_to_first_x + second_to_first_y * second_to_first_y);
    double angle_between_movement_and_second_camera_transformation = acos((second_to_first_x * second_tracer_to_traced_tx + second_to_first_y * second_tracer_to_traced_ty) / (movement_distance * second_transformation_distance));

    // sine formula
    double length = movement_distance / sin(angle_between_two_camera_transformations) * sin(angle_between_movement_and_second_camera_transformation);
    std::string current_time = std::to_string(round(ros::Time::now().toSec() * 100.0) / 100.0);

    // {
    //   std::ofstream fw("Find_length_" + current_time + "_.txt", std::ofstream::out);
    //   if (fw.is_open())
    //   {
    //     fw << "first_x: " << first_x << std::endl;
    //     fw << "first_y: " << first_y << std::endl;
    //     fw << "first_tracer_to_traced_tx: " << first_tracer_to_traced_tx << std::endl;
    //     fw << "first_tracer_to_traced_ty: " << first_tracer_to_traced_ty << std::endl;
    //     fw << "second_x: " << second_x << std::endl;
    //     fw << "second_y: " << second_y << std::endl;
    //     fw << "second_tracer_to_traced_tx: " << second_tracer_to_traced_tx << std::endl;
    //     fw << "second_tracer_to_traced_ty: " << second_tracer_to_traced_ty << std::endl;
    //     fw << std::endl;
    //     fw << "first_transformation_distance: " << first_transformation_distance << std::endl;
    //     fw << "second_transformation_distance: " << second_transformation_distance << std::endl;
    //     fw << "angle_between_two_camera_transformations: " << angle_between_two_camera_transformations << std::endl;
    //     fw << "second_to_first_x: " << second_to_first_x << std::endl;
    //     fw << "second_to_first_y: " << second_to_first_y << std::endl;
    //     fw << "movement_distance: " << movement_distance << std::endl;
    //     fw << "angle_between_movement_and_second_camera_transformation: " << angle_between_movement_and_second_camera_transformation << std::endl;
    //     fw << std::endl;
    //     fw << "length: " << length << std::endl;
    //     fw.close();
    //   }
    // }

    return length;
  }

  void Traceback::findAdjustedTransformation(cv::Mat &original, cv::Mat &adjusted, double scale, double first_tracer_to_traced_tx, double first_tracer_to_traced_ty, double first_tracer_to_traced_r, double first_x, double first_y, float src_resolution)
  {
    cv::Mat t1(3, 3, CV_64F);
    t1.at<double>(0, 0) = 1.0;
    t1.at<double>(0, 1) = 0.0;
    t1.at<double>(0, 2) = -1 * first_x / src_resolution;
    t1.at<double>(1, 0) = 0.0;
    t1.at<double>(1, 1) = 1.0;
    t1.at<double>(1, 2) = -1 * first_y / src_resolution;
    t1.at<double>(2, 0) = 0.0;
    t1.at<double>(2, 1) = 0.0;
    t1.at<double>(2, 2) = 1.0;

    cv::Mat adjustment(3, 3, CV_64F);
    adjustment.at<double>(0, 0) = cos(first_tracer_to_traced_r);
    adjustment.at<double>(0, 1) = -sin(first_tracer_to_traced_r);
    adjustment.at<double>(0, 2) = first_tracer_to_traced_tx * scale / src_resolution;
    adjustment.at<double>(1, 0) = sin(first_tracer_to_traced_r);
    adjustment.at<double>(1, 1) = cos(first_tracer_to_traced_r);
    adjustment.at<double>(1, 2) = first_tracer_to_traced_ty * scale / src_resolution;
    adjustment.at<double>(2, 0) = 0.0;
    adjustment.at<double>(2, 1) = 0.0;
    adjustment.at<double>(2, 2) = 1.0;

    cv::Mat t2(3, 3, CV_64F);
    t2.at<double>(0, 0) = 1.0;
    t2.at<double>(0, 1) = 0.0;
    t2.at<double>(0, 2) = first_x / src_resolution;
    t2.at<double>(1, 0) = 0.0;
    t2.at<double>(1, 1) = 1.0;
    t2.at<double>(1, 2) = first_y / src_resolution;
    t2.at<double>(2, 0) = 0.0;
    t2.at<double>(2, 1) = 0.0;
    t2.at<double>(2, 2) = 1.0;

    adjusted = t2 * adjustment * t1 * original;
  }

  void Traceback::writeTracebackFeedbackHistory(std::string tracer, std::string traced, std::string feedback)
  {
    std::string current_time = std::to_string(round(ros::Time::now().toSec() * 100.0) / 100.0);

    std::ofstream fw("Feedback_history_" + tracer.substr(1) + "_tracer_robot_" + traced.substr(1) + "_traced_robot.txt", std::ofstream::app);
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
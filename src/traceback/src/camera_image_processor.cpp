#include <traceback/camera_image_processor.h>

#include <ros/assert.h>
#include <ros/console.h>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/calib3d/calib3d.hpp"

#include <fstream>

namespace traceback
{
    bool CameraImageProcessor::computeFeaturesAndDepths(cv::detail::ImageFeatures &out_features, std::vector<double> &out_depths, const cv::Mat &image, FeatureType feature_type, const cv::Mat &depth_image)
    {
        auto finder = internal::chooseFeatureFinder(feature_type);

        cv::detail::computeImageFeatures(finder, image, out_features);

        std::vector<cv::KeyPoint> keypoints;
        std::vector<cv::Point2f> image_points;
        for (auto &keypoint : out_features.keypoints)
        {
            keypoints.push_back(keypoint);
        }
        cv::KeyPoint::convert(keypoints, image_points);

        out_depths.clear();
        for (auto &point : image_points)
        {
            out_depths.emplace_back(1.0 * depth_image.at<float>((int)point.y, (int)point.x));
        }

        if (out_features.keypoints.size() == 0 || out_depths.size() == 0)
        {
            return false;
        }

        return true;
    }

    MatchAndSolveResult CameraImageProcessor::matchAndSolveWithFeaturesAndDepths(cv::detail::ImageFeatures &tracer_robot_features, cv::detail::ImageFeatures &traced_robot_features, std::vector<double> tracer_robot_depths, std::vector<double> traced_robot_depths, double confidence, double yaw, TransformNeeded &transform_needed, std::string tracer_robot, std::string traced_robot, std::string current_time)
    {
        std::vector<cv::detail::ImageFeatures> image_features = {tracer_robot_features, traced_robot_features};
        std::vector<cv::detail::MatchesInfo> pairwise_matches;
        std::vector<int> good_indices;
        cv::Ptr<cv::detail::FeaturesMatcher> matcher =
            cv::makePtr<cv::detail::AffineBestOf2NearestMatcher>();

        try
        {
            ROS_DEBUG("pairwise matching features");
            (*matcher)(image_features, pairwise_matches);
            matcher = {};
        }
        catch (std::exception e)
        {
            ROS_INFO("Not enough features, catched!");
            {
                MatchAndSolveResult result;
                result.match = false;
                result.solved = false;
                return result;
            }
        }

        double match_score = -1.0; // does not even have match
        for (auto &match_info : pairwise_matches)
        {
            if (match_info.H.empty() ||
                match_info.src_img_idx >= match_info.dst_img_idx)
            {
                continue;
            }
            match_score = match_info.confidence;
        }

        good_indices = cv::detail::leaveBiggestComponent(
            image_features, pairwise_matches, static_cast<float>(confidence));

        // Write match score to file
        {
            std::ofstream fw(tracer_robot.substr(1) + "_" + traced_robot.substr(1) + "/" + "Match_features_score_" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot" + "_match_score.txt", std::ofstream::app);
            if (fw.is_open())
            {
                if (good_indices.size() == 1)
                {
                    fw << "Does not match at time " << current_time << " with confidence " << match_score << std::endl;
                }
                else
                {
                    fw << "Match at time " << current_time << " with confidence " << match_score << std::endl;
                }
                fw.close();
            }
        }
        // END

        // no match found
        if (good_indices.size() == 1)
        {
            {
                MatchAndSolveResult result;
                result.match = false;
                result.solved = false;
                return result;
            }
        }

        std::vector<cv::Point2f> image_points1, image_points2;
        std::vector<double> matched_depths1, matched_depths2;

        if (image_features.size() == 2)
        {
            // std::vector<cv::KeyPoint> keypoints1 = image_features[0].keypoints;
            // cv::KeyPoint::convert(keypoints1, points1);
            // std::vector<cv::KeyPoint> keypoints2 = image_features[1].keypoints;
            // cv::KeyPoint::convert(keypoints2, points2);
            std::vector<cv::KeyPoint> keypoints1, keypoints2;
            for (auto &match_info : pairwise_matches)
            {
                if (match_info.H.empty() ||
                    match_info.src_img_idx >= match_info.dst_img_idx)
                {
                    continue;
                }

                for (size_t i = 0; i < match_info.matches.size(); ++i)
                {
                    auto match = match_info.matches[i];
                    if (match_info.inliers_mask[i] == 1)
                    {
                        // match.imgIdx is always 0
                        keypoints1.emplace_back(image_features[0].keypoints[match.queryIdx]);
                        keypoints2.emplace_back(image_features[1].keypoints[match.trainIdx]);

                        matched_depths1.emplace_back(tracer_robot_depths[match.queryIdx]);
                        matched_depths2.emplace_back(traced_robot_depths[match.trainIdx]);
                    }
                }
            }
            cv::KeyPoint::convert(keypoints1, image_points1);
            cv::KeyPoint::convert(keypoints2, image_points2);
        }
        else
        {
            ROS_ERROR("image_features must be of size 2, matching exactly 2 images!");
        }

        ROS_INFO("Number of inlier matches is %zu", image_points1.size());
        // Filter case where match number is too small
        // DLT algorithm needs at least 4 points for pose estimation from 3D-2D point correspondences. (expected: 'count >= 6')
        if (image_points1.size() < 4)
        {
            {
                MatchAndSolveResult result;
                result.match = true;
                result.solved = false;
                return result;
            }
        }

        double fx = 554.254691191187;
        double fy = 554.254691191187;
        double cx = 320.5;
        double cy = 240.5;

        std::vector<cv::Point3d> points1;                // nan depth is filtered
        std::vector<cv::Point2d> filtered_image_points1; // nan depth is filtered
        std::vector<cv::Point2d> filtered_image_points2; // nan depth is filtered
        for (size_t i = 0; i < image_points1.size(); ++i)
        {
            cv::Point2d image_point = 1.0 * image_points1[i];
            double depth = matched_depths1[i];
            // ROS_DEBUG("depth: %f", depth);
            if (isnan(depth))
            {
                continue;
            }
            double x = depth * (image_point.x - cx) / fx;
            double y = depth * (image_point.y - cy) / fy;
            double z = depth;
            points1.emplace_back(cv::Point3d(x, y, z));
            filtered_image_points1.push_back(image_point);
            filtered_image_points2.push_back(1.0 * image_points2[i]);
        }

        // Below is the same as matchAndSolve
        // For the match, write the number of filtered matches to file
        {
            std::ofstream fw(tracer_robot.substr(1) + "_" + traced_robot.substr(1) + "/" + "Filtered_features_number_of_matches" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot.txt", std::ofstream::app);
            if (fw.is_open())
            {
                fw << "Number of filtered matches at " << current_time << " is " << points1.size() << std::endl;
                fw.close();
            }
        }
        //

        // Filter case where filtered (depth != nan) match number is too small
        if (filtered_image_points1.size() < 4)
        {
            {
                MatchAndSolveResult result;
                result.match = true;
                result.solved = false;
                return result;
            }
        }

        // HARDCODE currently
        // if (filtered_image_points1.size() < 20)
        // {
        //     {
        //         MatchAndSolveResult result;
        //         result.match = true;
        //         result.solved = false;
        //         return result;
        //     }
        // }

        // solvepnp
        double k[9] = {554.254691191187, 0.0, 320.5, 0.0, 554.254691191187, 240.5, 0.0, 0.0, 1.0};
        cv::Mat camera_K = cv::Mat(3, 3, CV_64F, k);
        // For "plumb_bob", the 5 parameters are: (k1, k2, t1, t2, k3).
        double d[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
        cv::Mat camera_D = cv::Mat(5, 1, CV_64F, d);

        cv::Mat rvec1 = cv::Mat(3, 1, CV_64F);
        cv::Mat tvec1 = cv::Mat(3, 1, CV_64F);
        cv::Mat rvec2 = cv::Mat(3, 1, CV_64F);
        cv::Mat tvec2 = cv::Mat(3, 1, CV_64F);
        // Use init guess
        rvec1.at<double>(0, 0) = 0.0;
        rvec1.at<double>(1, 0) = 0.0;
        rvec1.at<double>(2, 0) = 0.0;
        tvec1.at<double>(0, 0) = 0.0;
        tvec1.at<double>(1, 0) = 0.0;
        tvec1.at<double>(2, 0) = 0.0;
        rvec2.at<double>(0, 0) = 0.0;
        rvec2.at<double>(1, 0) = 0.0;
        rvec2.at<double>(2, 0) = 0.0;
        tvec2.at<double>(0, 0) = 0.0;
        tvec2.at<double>(1, 0) = 0.0;
        tvec2.at<double>(2, 0) = 0.0;
        // Use init guess END
        cv::solvePnPRansac(points1, filtered_image_points1, camera_K, camera_D, rvec1, tvec1, true, 100, 0.5f);
        cv::solvePnPRansac(points1, filtered_image_points2, camera_K, camera_D, rvec2, tvec2, true, 100, 0.5f);

        cv::Mat rmat1;
        cv::Rodrigues(rvec1, rmat1);
        cv::Mat rmat2;
        cv::Rodrigues(rvec2, rmat2);

        // compute C2MC1
        // R 1 to 2 and t 1 to 2
        cv::Mat transform_R = rmat2 * rmat1.t();
        cv::Mat transform_t = rmat2 * (-rmat1.t() * tvec1) + tvec2;

        //
        cv::Vec3d rot = rotationMatrixToEulerAngles(transform_R);
        {
            std::ofstream fw(tracer_robot.substr(1) + "_" + traced_robot.substr(1) + "/" + current_time + "_" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot" + "_features_transform_R.txt", std::ofstream::out);
            if (fw.is_open())
            {
                fw << "XYZ rotation is: " << rotationMatrixToEulerAngles(rmat1) << std::endl;
                fw << "Rotation matrix C1MO (O to C1):" << std::endl;
                fw << rmat1.at<double>(0, 0) << "\t" << rmat1.at<double>(0, 1) << "\t" << rmat1.at<double>(0, 2) << std::endl;
                fw << rmat1.at<double>(1, 0) << "\t" << rmat1.at<double>(1, 1) << "\t" << rmat1.at<double>(1, 2) << std::endl;
                fw << rmat1.at<double>(2, 0) << "\t" << rmat1.at<double>(2, 1) << "\t" << rmat1.at<double>(2, 2) << std::endl;

                fw << "XYZ rotation is: " << rotationMatrixToEulerAngles(rmat2) << std::endl;
                fw << "Rotation matrix C2MO (O to C2):" << std::endl;
                fw << rmat2.at<double>(0, 0) << "\t" << rmat2.at<double>(0, 1) << "\t" << rmat2.at<double>(0, 2) << std::endl;
                fw << rmat2.at<double>(1, 0) << "\t" << rmat2.at<double>(1, 1) << "\t" << rmat2.at<double>(1, 2) << std::endl;
                fw << rmat2.at<double>(2, 0) << "\t" << rmat2.at<double>(2, 1) << "\t" << rmat2.at<double>(2, 2) << std::endl;

                fw << "XYZ rotation is: " << rot << std::endl;
                fw << "Rotation matrix R:" << std::endl;
                fw << transform_R.at<double>(0, 0) << "\t" << transform_R.at<double>(0, 1) << "\t" << transform_R.at<double>(0, 2) << std::endl;
                fw << transform_R.at<double>(1, 0) << "\t" << transform_R.at<double>(1, 1) << "\t" << transform_R.at<double>(1, 2) << std::endl;
                fw << transform_R.at<double>(2, 0) << "\t" << transform_R.at<double>(2, 1) << "\t" << transform_R.at<double>(2, 2) << std::endl;
                // fw << std::endl;
                fw.close();
            }
        }

        {
            std::ofstream fw(tracer_robot.substr(1) + "_" + traced_robot.substr(1) + "/" + current_time + "_" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot" + "_features_transform_t.txt", std::ofstream::out);
            if (fw.is_open())
            {
                fw << "Translation matrix C1MO (O to C1):" << std::endl;
                fw << tvec1.at<double>(0, 0) << std::endl;
                fw << tvec1.at<double>(1, 0) << std::endl;
                fw << tvec1.at<double>(2, 0) << std::endl;

                fw << "Translation matrix C2MO (O to C2):" << std::endl;
                fw << tvec2.at<double>(0, 0) << std::endl;
                fw << tvec2.at<double>(1, 0) << std::endl;
                fw << tvec2.at<double>(2, 0) << std::endl;

                fw << "Translation matrix t:" << std::endl;
                fw << transform_t.at<double>(0, 0) << std::endl;
                fw << transform_t.at<double>(1, 0) << std::endl;
                fw << transform_t.at<double>(2, 0) << std::endl;
                fw.close();
            }
        }

        /*
        Bear in mind that tracer is the src image and traced is the dst image.

        transform_t is of the form (x, y, z)
        t.y should be considered 0.
        If yaw=0, the more positive the t.z is, the more the goal (traced) is behind from the tracer, the more negative x translation is needed
        to translate from tracer to traced.
        If yaw=0, the more positive the t.x is, the more the goal (traced) is to the left of the tracer, the more positive y translation is needed
        to translate from tracer to traced.

        When computing transform_t, must consider the current orientation.
        The goal is already in the tracer's coordinate frame, and the essential matrix
        is actually to rotate, then translate from tracer to traced, in the tracer's coordinate frame.

        transform_t probably needs to be scaled (element-wise multiplication) before it can be used.

        For transform_R, only consider the y-axis rotation since this y-axis rotation is the z-axis rotation in the robot world.
        The more positive this rotation, the more positive z-axis rotation is needed in the robot world.

        transform_R can be directly used.
        */
        transform_needed.tx = (-1 * transform_t.at<double>(2, 0) * cos(yaw)) + (-1 * transform_t.at<double>(0, 0) * sin(yaw));
        transform_needed.ty = transform_t.at<double>(0, 0) * cos(yaw) + (-1 * transform_t.at<double>(2, 0) * sin(yaw));
        transform_needed.r = rot[1];

        // Regard as does not match when solvepnp returns result that does not make sense
        if (abs(transform_needed.tx) > 1.5 || abs(transform_needed.ty) > 1.5)
        {
            {
                MatchAndSolveResult result;
                result.match = false;
                result.solved = false;
                return result;
            }
        }
        //

        ROS_INFO("Debug");

        {
            MatchAndSolveResult result;
            result.match = true;
            result.solved = true;
            return result;
        }
    }

    MatchAndSolveResult CameraImageProcessor::matchAndSolve(const cv::Mat &tracer_robot_color_image, const cv::Mat &traced_robot_color_image, const cv::Mat &tracer_robot_depth_image, const cv::Mat &traced_robot_depth_image, FeatureType feature_type, double confidence, double yaw, TransformNeeded &transform_needed, std::string tracer_robot, std::string traced_robot, std::string current_time)
    {
        const std::vector<cv::Mat> &images = {tracer_robot_color_image, traced_robot_color_image};
        std::vector<cv::detail::ImageFeatures> image_features;
        std::vector<cv::detail::MatchesInfo> pairwise_matches;
        std::vector<cv::detail::CameraParams> transforms;
        std::vector<int> good_indices;
        // TODO investigate value translation effect on features
        auto finder = internal::chooseFeatureFinder(feature_type);
        cv::Ptr<cv::detail::FeaturesMatcher> matcher =
            cv::makePtr<cv::detail::AffineBestOf2NearestMatcher>();

        if (tracer_robot_color_image.empty() || traced_robot_color_image.empty())
        {
            ROS_ERROR("Either traced robot image or tracer robot image is empty, which should not be the case!");
            {
                MatchAndSolveResult result;
                result.match = false;
                result.solved = false;
                return result;
            }
        }

        image_features.reserve(images.size());
        for (const cv::Mat &image : images)
        {
            image_features.emplace_back();
            if (!image.empty())
            {
#if CV_VERSION_MAJOR >= 4
                cv::detail::computeImageFeatures(finder, image, image_features.back());
#else
                (*finder)(image, image_features.back());
#endif
            }
        }
        finder = {};

        /* find corespondent features */
        try
        {
            ROS_DEBUG("pairwise matching features");
            (*matcher)(image_features, pairwise_matches);
            matcher = {};
        }
        catch (std::exception e)
        {
            ROS_INFO("Not enough features, catched!");
            {
                MatchAndSolveResult result;
                result.match = false;
                result.solved = false;
                return result;
            }
        }

        // #ifndef NDEBUG
        internal::writeDebugMatchingInfo(images, image_features, pairwise_matches, traced_robot, tracer_robot, current_time);
        // #endif

        double match_score = -1.0; // does not even have match
        for (auto &match_info : pairwise_matches)
        {
            if (match_info.H.empty() ||
                match_info.src_img_idx >= match_info.dst_img_idx)
            {
                continue;
            }
            match_score = match_info.confidence;
        }

        good_indices = cv::detail::leaveBiggestComponent(
            image_features, pairwise_matches, static_cast<float>(confidence));

        // Write match score to file
        {
            std::ofstream fw(tracer_robot.substr(1) + "_" + traced_robot.substr(1) + "/" + "Match_traceback_score_" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot" + "_match_score.txt", std::ofstream::app);
            if (fw.is_open())
            {
                if (good_indices.size() == 1)
                {
                    fw << "Does not match at time " << current_time << " with confidence " << match_score << std::endl;
                }
                else
                {
                    fw << "Match at time " << current_time << " with confidence " << match_score << std::endl;
                }
                fw.close();
            }
        }
        // END

        // no match found
        if (good_indices.size() == 1)
        {
            {
                MatchAndSolveResult result;
                result.match = false;
                result.solved = false;
                return result;
            }
        }

        std::vector<cv::Point2f> image_points1, image_points2;

        if (image_features.size() == 2)
        {
            // std::vector<cv::KeyPoint> keypoints1 = image_features[0].keypoints;
            // cv::KeyPoint::convert(keypoints1, points1);
            // std::vector<cv::KeyPoint> keypoints2 = image_features[1].keypoints;
            // cv::KeyPoint::convert(keypoints2, points2);
            std::vector<cv::KeyPoint> keypoints1, keypoints2;
            for (auto &match_info : pairwise_matches)
            {
                if (match_info.H.empty() ||
                    match_info.src_img_idx >= match_info.dst_img_idx)
                {
                    continue;
                }

                for (size_t i = 0; i < match_info.matches.size(); ++i)
                {
                    auto match = match_info.matches[i];
                    if (match_info.inliers_mask[i] == 1)
                    {
                        // match.imgIdx is always 0
                        keypoints1.emplace_back(image_features[0].keypoints[match.queryIdx]);
                        keypoints2.emplace_back(image_features[1].keypoints[match.trainIdx]);
                    }
                }
            }
            cv::KeyPoint::convert(keypoints1, image_points1);
            cv::KeyPoint::convert(keypoints2, image_points2);
        }
        else
        {
            ROS_ERROR("image_features must be of size 2, matching exactly 2 images!");
        }

        ROS_INFO("Number of inlier matches is %zu", image_points1.size());
        // Filter case where match number is too small
        // DLT algorithm needs at least 4 points for pose estimation from 3D-2D point correspondences. (expected: 'count >= 6')
        if (image_points1.size() < 4)
        {
            {
                MatchAndSolveResult result;
                result.match = true;
                result.solved = false;
                return result;
            }
        }

        double fx = 554.254691191187;
        double fy = 554.254691191187;
        double cx = 320.5;
        double cy = 240.5;

        std::vector<cv::Point3d> points1;                // nan depth is filtered
        std::vector<cv::Point2d> filtered_image_points1; // nan depth is filtered
        std::vector<cv::Point2d> filtered_image_points2; // nan depth is filtered
        for (size_t i = 0; i < image_points1.size(); ++i)
        {
            cv::Point2d image_point = 1.0 * image_points1[i];
            double depth = 1.0 * tracer_robot_depth_image.at<float>((int)image_point.y, (int)image_point.x);
            // ROS_DEBUG("depth: %f", depth);
            if (isnan(depth))
            {
                continue;
            }
            double x = depth * (image_point.x - cx) / fx;
            double y = depth * (image_point.y - cy) / fy;
            double z = depth;
            points1.emplace_back(cv::Point3d(x, y, z));
            filtered_image_points1.push_back(image_point);
            filtered_image_points2.push_back(1.0 * image_points2[i]);
        }

        // For the match, write the number of filtered matches to file
        {
            std::ofstream fw(tracer_robot.substr(1) + "_" + traced_robot.substr(1) + "/" + "Filtered_traceback_number_of_matches" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot.txt", std::ofstream::app);
            if (fw.is_open())
            {
                fw << "Number of filtered matches at " << current_time << " is " << points1.size() << std::endl;
                fw.close();
            }
        }
        //

        // Filter case where filtered (depth != nan) match number is too small
        if (filtered_image_points1.size() < 4)
        {
            {
                MatchAndSolveResult result;
                result.match = true;
                result.solved = false;
                return result;
            }
        }

        // HARDCODE currently
        // if (filtered_image_points1.size() < 20)
        // {
        //     {
        //         MatchAndSolveResult result;
        //         result.match = true;
        //         result.solved = false;
        //         return result;
        //     }
        // }

        // solvepnp
        double k[9] = {554.254691191187, 0.0, 320.5, 0.0, 554.254691191187, 240.5, 0.0, 0.0, 1.0};
        cv::Mat camera_K = cv::Mat(3, 3, CV_64F, k);
        // For "plumb_bob", the 5 parameters are: (k1, k2, t1, t2, k3).
        double d[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
        cv::Mat camera_D = cv::Mat(5, 1, CV_64F, d);

        cv::Mat rvec1 = cv::Mat(3, 1, CV_64F);
        cv::Mat tvec1 = cv::Mat(3, 1, CV_64F);
        cv::Mat rvec2 = cv::Mat(3, 1, CV_64F);
        cv::Mat tvec2 = cv::Mat(3, 1, CV_64F);
        // Use init guess
        rvec1.at<double>(0, 0) = 0.0;
        rvec1.at<double>(1, 0) = 0.0;
        rvec1.at<double>(2, 0) = 0.0;
        tvec1.at<double>(0, 0) = 0.0;
        tvec1.at<double>(1, 0) = 0.0;
        tvec1.at<double>(2, 0) = 0.0;
        rvec2.at<double>(0, 0) = 0.0;
        rvec2.at<double>(1, 0) = 0.0;
        rvec2.at<double>(2, 0) = 0.0;
        tvec2.at<double>(0, 0) = 0.0;
        tvec2.at<double>(1, 0) = 0.0;
        tvec2.at<double>(2, 0) = 0.0;
        // Use init guess END
        cv::solvePnPRansac(points1, filtered_image_points1, camera_K, camera_D, rvec1, tvec1, true, 100, 0.5f);
        cv::solvePnPRansac(points1, filtered_image_points2, camera_K, camera_D, rvec2, tvec2, true, 100, 0.5f);

        cv::Mat rmat1;
        cv::Rodrigues(rvec1, rmat1);
        cv::Mat rmat2;
        cv::Rodrigues(rvec2, rmat2);

        // compute C2MC1
        // R 1 to 2 and t 1 to 2
        cv::Mat transform_R = rmat2 * rmat1.t();
        cv::Mat transform_t = rmat2 * (-rmat1.t() * tvec1) + tvec2;

        //
        cv::Vec3d rot = rotationMatrixToEulerAngles(transform_R);
        {
            std::ofstream fw(tracer_robot.substr(1) + "_" + traced_robot.substr(1) + "/" + current_time + "_" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot" + "_traceback_transform_R.txt", std::ofstream::out);
            if (fw.is_open())
            {
                fw << "XYZ rotation is: " << rotationMatrixToEulerAngles(rmat1) << std::endl;
                fw << "Rotation matrix C1MO (O to C1):" << std::endl;
                fw << rmat1.at<double>(0, 0) << "\t" << rmat1.at<double>(0, 1) << "\t" << rmat1.at<double>(0, 2) << std::endl;
                fw << rmat1.at<double>(1, 0) << "\t" << rmat1.at<double>(1, 1) << "\t" << rmat1.at<double>(1, 2) << std::endl;
                fw << rmat1.at<double>(2, 0) << "\t" << rmat1.at<double>(2, 1) << "\t" << rmat1.at<double>(2, 2) << std::endl;

                fw << "XYZ rotation is: " << rotationMatrixToEulerAngles(rmat2) << std::endl;
                fw << "Rotation matrix C2MO (O to C2):" << std::endl;
                fw << rmat2.at<double>(0, 0) << "\t" << rmat2.at<double>(0, 1) << "\t" << rmat2.at<double>(0, 2) << std::endl;
                fw << rmat2.at<double>(1, 0) << "\t" << rmat2.at<double>(1, 1) << "\t" << rmat2.at<double>(1, 2) << std::endl;
                fw << rmat2.at<double>(2, 0) << "\t" << rmat2.at<double>(2, 1) << "\t" << rmat2.at<double>(2, 2) << std::endl;

                fw << "XYZ rotation is: " << rot << std::endl;
                fw << "Rotation matrix R:" << std::endl;
                fw << transform_R.at<double>(0, 0) << "\t" << transform_R.at<double>(0, 1) << "\t" << transform_R.at<double>(0, 2) << std::endl;
                fw << transform_R.at<double>(1, 0) << "\t" << transform_R.at<double>(1, 1) << "\t" << transform_R.at<double>(1, 2) << std::endl;
                fw << transform_R.at<double>(2, 0) << "\t" << transform_R.at<double>(2, 1) << "\t" << transform_R.at<double>(2, 2) << std::endl;
                // fw << std::endl;
                fw.close();
            }
        }

        {
            std::ofstream fw(tracer_robot.substr(1) + "_" + traced_robot.substr(1) + "/" + current_time + "_" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot" + "_traceback_transform_t.txt", std::ofstream::out);
            if (fw.is_open())
            {
                fw << "Translation matrix C1MO (O to C1):" << std::endl;
                fw << tvec1.at<double>(0, 0) << std::endl;
                fw << tvec1.at<double>(1, 0) << std::endl;
                fw << tvec1.at<double>(2, 0) << std::endl;

                fw << "Translation matrix C2MO (O to C2):" << std::endl;
                fw << tvec2.at<double>(0, 0) << std::endl;
                fw << tvec2.at<double>(1, 0) << std::endl;
                fw << tvec2.at<double>(2, 0) << std::endl;

                fw << "Translation matrix t:" << std::endl;
                fw << transform_t.at<double>(0, 0) << std::endl;
                fw << transform_t.at<double>(1, 0) << std::endl;
                fw << transform_t.at<double>(2, 0) << std::endl;
                fw.close();
            }
        }

        /*
        Bear in mind that tracer is the src image and traced is the dst image.

        transform_t is of the form (x, y, z)
        t.y should be considered 0.
        If yaw=0, the more positive the t.z is, the more the goal (traced) is behind from the tracer, the more negative x translation is needed
        to translate from tracer to traced.
        If yaw=0, the more positive the t.x is, the more the goal (traced) is to the left of the tracer, the more positive y translation is needed
        to translate from tracer to traced.

        When computing transform_t, must consider the current orientation.
        The goal is already in the tracer's coordinate frame, and the essential matrix
        is actually to rotate, then translate from tracer to traced, in the tracer's coordinate frame.

        transform_t probably needs to be scaled (element-wise multiplication) before it can be used.

        For transform_R, only consider the y-axis rotation since this y-axis rotation is the z-axis rotation in the robot world.
        The more positive this rotation, the more positive z-axis rotation is needed in the robot world.

        transform_R can be directly used.
        */
        transform_needed.tx = (-1 * transform_t.at<double>(2, 0) * cos(yaw)) + (-1 * transform_t.at<double>(0, 0) * sin(yaw));
        transform_needed.ty = transform_t.at<double>(0, 0) * cos(yaw) + (-1 * transform_t.at<double>(2, 0) * sin(yaw));
        transform_needed.r = rot[1];

        // Regard as does not match when solvepnp returns result that does not make sense
        if (abs(transform_needed.tx) > 1.5 || abs(transform_needed.ty) > 1.5)
        {
            {
                MatchAndSolveResult result;
                result.match = false;
                result.solved = false;
                return result;
            }
        }
        //

        ROS_INFO("Debug");

        {
            MatchAndSolveResult result;
            result.match = true;
            result.solved = true;
            return result;
        }
    }

    std::vector<double> CameraImageProcessor::LMOptimize(std::vector<double> x_values,
                                                         std::vector<double> y_values,
                                                         std::vector<double> tx_values,
                                                         std::vector<double> ty_values,
                                                         std::vector<double> r_values,
                                                         double init_tx,
                                                         double init_ty,
                                                         double init_r)
    {
        // 'm' is the number of data points.
        int m = x_values.size();

        // Move the data into an Eigen Matrix.
        // The first column has the input values, x. The second column is the f(x) values.
        Eigen::MatrixXd measuredValues(m, 5);
        for (int i = 0; i < m; i++)
        {
            measuredValues(i, 0) = x_values[i];
            measuredValues(i, 1) = y_values[i];
            measuredValues(i, 2) = tx_values[i];
            measuredValues(i, 3) = ty_values[i];
            measuredValues(i, 4) = r_values[i];
        }

        // 'n' is the number of parameters in the function.
        // f(x) = a(x^2) + b(x) + c has 3 parameters: a, b, c
        int n = 3;

        // 'x' is vector of length 'n' containing the initial values for the parameters.
        // The parameters 'x' are also referred to as the 'inputs' in the context of LM optimization.
        // The LM optimization inputs should not be confused with the x input values.
        Eigen::VectorXd x(n);
        x(0) = init_tx; // initial value for 'a'
        x(1) = init_ty; // initial value for 'b'
        x(2) = init_r;  // initial value for 'c'

        //
        // Run the LM optimization
        // Create a LevenbergMarquardt object and pass it the functor.
        //

        LMFunctor functor;
        functor.measuredValues = measuredValues;
        functor.m = m;
        functor.n = n;

        Eigen::LevenbergMarquardt<LMFunctor, double> lm(functor);
        int status = lm.minimize(x);
        ROS_INFO("LM optimization status: %d", status);

        //
        // Results
        // The 'x' vector also contains the results of the optimization.
        //
        ROS_INFO("Optimization results (a, b, c) = (%f, %f, %f)", x(0), x(1), x(2));

        return {x(0), x(1), x(2)};
    }

    // Calculates rotation matrix to euler angles
    // The result is the same as MATLAB except the order
    // of the euler angles ( x and z are swapped ).
    cv::Vec3d CameraImageProcessor::rotationMatrixToEulerAngles(cv::Mat &R)
    {
        double sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

        bool singular = sy < 1e-6; // If

        double x, y, z;
        if (!singular)
        {
            x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
            y = atan2(-R.at<double>(2, 0), sy);
            z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
        }
        else
        {
            x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
            y = atan2(-R.at<double>(2, 0), sy);
            z = 0;
        }
        return cv::Vec3d(x, y, z);
    }
}

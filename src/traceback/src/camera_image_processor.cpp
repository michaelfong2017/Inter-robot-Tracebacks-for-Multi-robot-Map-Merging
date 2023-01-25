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
    bool CameraImageProcessor::findEssentialMatrix(const cv::Mat &traced_robot_image, const cv::Mat &tracer_robot_image, FeatureType feature_type,
                                                   double confidence, std::string traced_robot, std::string tracer_robot, std::string current_time)
    {
        const std::vector<cv::Mat> &images = {traced_robot_image, tracer_robot_image};
        std::vector<cv::detail::ImageFeatures> image_features;
        std::vector<cv::detail::MatchesInfo> pairwise_matches;
        std::vector<cv::detail::CameraParams> transforms;
        std::vector<int> good_indices;
        // TODO investigate value translation effect on features
        auto finder = internal::chooseFeatureFinder(feature_type);
        cv::Ptr<cv::detail::FeaturesMatcher> matcher =
            cv::makePtr<cv::detail::AffineBestOf2NearestMatcher>();

        if (traced_robot_image.empty() || tracer_robot_image.empty())
        {
            ROS_ERROR("Either traced robot image or tracer robot image is empty, which should not be the case!");
            return true;
        }

        ROS_DEBUG("findEssentialMatrix computing features");
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
        ROS_DEBUG("pairwise matching features");
        (*matcher)(image_features, pairwise_matches);
        matcher = {};

        /* use only matches that has enough confidence. leave out matches that are not
         * connected (small components) */
        /* e.g. pairwise_matches becomes [(0, 0), (0, 1), (1, 0), (1, 1)]
        good_indices becomes [1, 2]
        Therefore, the 0 and 1 in pairwise_matches actually correspond to images 1 and 2 rather than images 0 and 1,
        so do the transforms 0 and 1 later, which actually correspond to images 1 and 2 too. */
        good_indices = cv::detail::leaveBiggestComponent(
            image_features, pairwise_matches, static_cast<float>(confidence));

        // #ifndef NDEBUG
        internal::writeDebugMatchingInfo(images, image_features, pairwise_matches, traced_robot, tracer_robot, current_time);
        // #endif

        // no match found
        if (good_indices.size() == 1)
        {
            return true;
        }

        for (auto &match_info : pairwise_matches)
        {
            ROS_INFO("match_info %d, %d", match_info.src_img_idx, match_info.dst_img_idx);
        }
        for (auto &indice : good_indices)
        {
            ROS_INFO("indice %d", indice);
        }
        // ROS_INFO("transforms size %zu", transforms.size());
        ROS_INFO("Debug");

        std::vector<cv::Point2f> points1, points2;

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
                for (auto &match : match_info.matches)
                {
                    // match.imgIdx is always 0
                    keypoints1.emplace_back(image_features[0].keypoints[match.queryIdx]);
                    keypoints2.emplace_back(image_features[1].keypoints[match.trainIdx]);
                }
            }
            cv::KeyPoint::convert(keypoints1, points1);
            cv::KeyPoint::convert(keypoints2, points2);
        }
        else
        {
            ROS_ERROR("image_features must be of size 2, matching exactly 2 images!");
        }

        double k[9] = {530.4669406576809, 0.0, 320.5, 0.0, 530.4669406576809, 240.5, 0.0, 0.0, 1.0};
        cv::Mat camera_K = cv::Mat(3, 3, CV_64F, k);

        cv::Mat essential_mat = cv::findEssentialMat(points1, points2, camera_K, cv::RANSAC);

        {
            std::ofstream fw(current_time + "_" + traced_robot.substr(1) + "_traced_robot_" + tracer_robot.substr(1) + "_tracer_robot" + "_essential_mat.txt", std::ofstream::out);
            if (fw.is_open())
            {
                fw << "Essential matrix E:" << std::endl;
                fw << essential_mat.at<double>(0, 0) << "\t" << essential_mat.at<double>(0, 1) << "\t" << essential_mat.at<double>(0, 2) << std::endl;
                fw << essential_mat.at<double>(1, 0) << "\t" << essential_mat.at<double>(1, 1) << "\t" << essential_mat.at<double>(1, 2) << std::endl;
                fw << essential_mat.at<double>(2, 0) << "\t" << essential_mat.at<double>(2, 1) << "\t" << essential_mat.at<double>(2, 2) << std::endl;
                fw.close();
            }
        }

        /*
        Bear in mind that traced is the src image and tracer is the dst image.

        transform_t is of the form (x, y, z)
        t.y should be considered 0.
        The more positive the t.z is, the more the tracer is behind from the goal, the more positive x translation is needed
        in the final transformation.
        The more positive the t.x is, the more the tracer is to the left of the goal, the more negative y translation is needed
        in the final transformation.

        transform_t probably needs to be scaled (element-wise multiplication) before it can be used.

        For transform_R, only consider the y-axis rotation since this y-axis rotation is the z-axis rotation in the robot world.
        The more positive this rotation, the more negative z-axis rotation is needed in the robot world.

        transform_R can be directly used.
        */
        cv::Mat transform_R, transform_t;
        int number_of_inliers = cv::recoverPose(essential_mat, points1, points2, camera_K, transform_R, transform_t);

        {
            std::ofstream fw(current_time + "_" + traced_robot.substr(1) + "_traced_robot_" + tracer_robot.substr(1) + "_tracer_robot" + "_transform_R.txt", std::ofstream::out);
            if (fw.is_open())
            {
                fw << "Number of inliers: " << number_of_inliers << std::endl;
                fw << "Rotation matrix R:" << std::endl;
                fw << transform_R.at<double>(0, 0) << "\t" << transform_R.at<double>(0, 1) << "\t" << transform_R.at<double>(0, 2) << std::endl;
                fw << transform_R.at<double>(1, 0) << "\t" << transform_R.at<double>(1, 1) << "\t" << transform_R.at<double>(1, 2) << std::endl;
                fw << transform_R.at<double>(2, 0) << "\t" << transform_R.at<double>(2, 1) << "\t" << transform_R.at<double>(2, 2) << std::endl;
                // fw << std::endl;
                fw.close();
            }
        }

        {
            std::ofstream fw(current_time + "_" + traced_robot.substr(1) + "_traced_robot_" + tracer_robot.substr(1) + "_tracer_robot" + "_transform_t.txt", std::ofstream::out);
            if (fw.is_open())
            {
                fw << "Number of inliers: " << number_of_inliers << std::endl;
                fw << "Translation matrix t:" << std::endl;
                fw << transform_t.at<double>(0, 0) << std::endl;
                fw << transform_t.at<double>(1, 0) << std::endl;
                fw << transform_t.at<double>(2, 0) << std::endl;
                fw.close();
            }
        }

        ROS_INFO("Debug");

        return true;
    }

    // Calculates rotation matrix to euler angles
    // The result is the same as MATLAB except the order
    // of the euler angles ( x and z are swapped ).
    cv::Vec3f CameraImageProcessor::rotationMatrixToEulerAngles(cv::Mat &R)
    {
        float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

        bool singular = sy < 1e-6; // If

        float x, y, z;
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
        return cv::Vec3f(x, y, z);
    }
}

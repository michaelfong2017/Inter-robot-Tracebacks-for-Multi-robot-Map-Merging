#include <traceback/camera_image_processor.h>

#include <ros/assert.h>
#include <ros/console.h>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace traceback
{
    bool CameraImageProcessor::findEssentialMatrix(const cv::Mat &traced_robot_image, const cv::Mat &tracer_robot_image, FeatureType feature_type,
                                                   double confidence)
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

        // #ifndef NDEBUG
        internal::writeDebugMatchingInfo(images, image_features, pairwise_matches);
        // #endif

        /* use only matches that has enough confidence. leave out matches that are not
         * connected (small components) */
        /* e.g. pairwise_matches becomes [(0, 0), (0, 1), (1, 0), (1, 1)]
        good_indices becomes [1, 2]
        Therefore, the 0 and 1 in pairwise_matches actually correspond to images 1 and 2 rather than images 0 and 1,
        so do the transforms 0 and 1 later, which actually correspond to images 1 and 2 too. */
        good_indices = cv::detail::leaveBiggestComponent(
            image_features, pairwise_matches, static_cast<float>(confidence));

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
        ROS_INFO("transforms size %zu", transforms.size());
        ROS_INFO("Debug");

        return true;
    }
}

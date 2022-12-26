#include <traceback/transform_estimator.h>
#include <traceback/estimation_internal.h>

#include <ros/assert.h>
#include <ros/console.h>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>

namespace traceback
{
    bool TransformEstimator::estimateTransforms(FeatureType feature_type,
                                                double confidence)
    {
        std::vector<cv::detail::ImageFeatures> image_features;
        std::vector<cv::detail::MatchesInfo> pairwise_matches;
        std::vector<cv::detail::CameraParams> transforms;
        std::vector<int> good_indices;
        // TODO investigate value translation effect on features
        auto finder = internal::chooseFeatureFinder(feature_type);
        cv::Ptr<cv::detail::FeaturesMatcher> matcher =
            cv::makePtr<cv::detail::AffineBestOf2NearestMatcher>();
        cv::Ptr<cv::detail::Estimator> estimator =
            cv::makePtr<cv::detail::AffineBasedEstimator>();
        cv::Ptr<cv::detail::BundleAdjusterBase> adjuster =
            cv::makePtr<cv::detail::BundleAdjusterAffinePartial>();

        if (images_.empty())
        {
            return true;
        }

        /* find features in images */
        ROS_DEBUG("computing features");
        image_features.reserve(images_.size());
        for (const cv::Mat &image : images_)
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
        internal::writeDebugMatchingInfo(images_, image_features, pairwise_matches);
        // #endif

        /* use only matches that has enough confidence. leave out matches that are not
         * connected (small components) */ 
        good_indices = cv::detail::leaveBiggestComponent(
            image_features, pairwise_matches, static_cast<float>(confidence));

        // no match found. try set first non-empty grid as reference frame. we try to
        // avoid setting empty grid as reference frame, in case some maps never
        // arrive. If all is empty just set null transforms.
        if (good_indices.size() == 1)
        {
            transforms_.clear();
            transforms_.resize(images_.size());
            for (size_t i = 0; i < images_.size(); ++i)
            {
                if (!images_[i].empty())
                {
                    // set identity
                    transforms_[i] = cv::Mat::eye(3, 3, CV_64F);
                    break;
                }
            }
            return true;
        }

        /* estimate transform */
        ROS_DEBUG("calculating transforms in global reference frame");
        // note: currently used estimator never fails
        if (!(*estimator)(image_features, pairwise_matches, transforms))
        {
            return false;
        }

        /* levmarq optimization */
        // openCV just accepts float transforms
        for (auto &transform : transforms)
        {
            transform.R.convertTo(transform.R, CV_32F);
        }
        ROS_DEBUG("optimizing global transforms");
        adjuster->setConfThresh(confidence);
        if (!(*adjuster)(image_features, pairwise_matches, transforms))
        {
            ROS_WARN("Bundle adjusting failed. Could not estimate transforms.");
            return false;
        }

        transforms_.clear();
        transforms_.resize(images_.size());
        size_t i = 0;
        for (auto &j : good_indices)
        {
            // we want to work with transforms as doubles
            transforms[i].R.convertTo(transforms_[static_cast<size_t>(j)], CV_64F);
            ++i;
        }

        for (size_t k = 0; k < transforms_.size(); ++k)
        {
            int width = transforms_[k].cols;
            int height = transforms_[k].rows;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    double val = transforms_[k].at<double>(y, x);
                    // do whatever you want with your value
                    ROS_INFO("transform[%zu]<%d, %d> = %f", k, y, x, val);
                }
            }
        }

        return true;
    }

}
#include <traceback/transform_estimator.h>
#include <traceback/estimation_internal.h>

#include <ros/assert.h>
#include <ros/console.h>
#include <opencv2/stitching/detail/matchers.hpp>

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

        /* estimate transform */
        ROS_DEBUG("calculating transforms in global reference frame");
        // note: currently used estimator never fails
        if (!(*estimator)(image_features, pairwise_matches, transforms))
        {
            return false;
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

        // transforms_.clear();
        // transforms_.resize(images_.size());
        // size_t i = 0;
        // for (auto &j : good_indices)
        // {
        //     // we want to work with transforms as doubles
        //     transforms[i].R.convertTo(transforms_[static_cast<size_t>(j)], CV_64F);
        //     ++i;
        // }

        toPairwiseTransforms(transforms, good_indices, images_.size(), transforms_vectors_);

        printTransformsVectors(transforms_vectors_);

        // for (size_t k = 0; k < transforms_.size(); ++k)
        // {
        //     int width = transforms_[k].cols;
        //     int height = transforms_[k].rows;
        //     for (int y = 0; y < height; y++)
        //     {
        //         for (int x = 0; x < width; x++)
        //         {
        //             double val = transforms_[k].at<double>(y, x);
        //             // do whatever you want with your value
        //             ROS_INFO("transform[%zu]<%d, %d> = %f", k, y, x, val);
        //         }
        //     }
        // }

        return true;
    }

    // Input .R should already be CV_32F.
    void TransformEstimator::toPairwiseTransforms(std::vector<cv::detail::CameraParams> transforms, std::vector<int> good_indices, size_t images_size, std::vector<std::vector<cv::Mat>> &transforms_vectors)
    {
        // e.g. good_indices is [1, 2] and if transforms[1] is the identity transform, image 2 will then be the reference image with identity transform.
        size_t identity_index = good_indices[findIdentityTransform(transforms)];

        transforms_vectors.clear();
        transforms_vectors.resize(images_size);

        for (auto &trans : transforms_vectors)
        {
            trans.resize(images_size);
        }

        // e.g. 20, 21, 22
        for (size_t j = 0; j < transforms.size(); ++j)
        {
            transforms[j].R.convertTo(transforms_vectors[identity_index][good_indices[j]], CV_64F);
        }

        // e.g. 02, 12
        for (size_t i = 0; i < transforms.size(); ++i)
        {
            if (good_indices[i] == identity_index)
                continue;

            cv::Mat temp;
            // Translation only, no rotation case
            if (abs(transforms[i].R.at<float>(0, 0) - 0.0f) < ZERO_ERROR && abs(transforms[i].R.at<float>(0, 1) - 0.0f) < ZERO_ERROR && abs(transforms[i].R.at<float>(1, 0) - 0.0f) < ZERO_ERROR && abs(transforms[i].R.at<float>(1, 1) - 0.0f) < ZERO_ERROR) {
                temp = transforms[i].R;
                float tx = transforms[i].R.at<float>(0, 2);
                temp.at<float>(0, 2) = -1 * tx;
                float ty = transforms[i].R.at<float>(1, 2);
                temp.at<float>(1, 2) = -1 * ty;
            }
            else {
                temp = transforms[i].R.inv();
            }

            temp.convertTo(transforms_vectors[good_indices[i]][identity_index], CV_64F);
        }

        // all remaining, e.g. 00, 01, 10, 11
        for (size_t i = 0; i < images_size; ++i)
        {
            for (size_t j = 0; j < images_size; ++j)
            {
                if (!transforms_vectors[i][j].empty())
                    continue;

                if (i == j)
                {
                    // set identity
                    transforms_vectors[i][j] = cv::Mat::eye(3, 3, CV_64F);
                }
                else if (std::find(good_indices.begin(), good_indices.end(), static_cast<int>(i)) != good_indices.end()
                && std::find(good_indices.begin(), good_indices.end(), static_cast<int>(j)) != good_indices.end()
                )
                {
                    cv::Mat temp = transforms_vectors[identity_index][j] * transforms_vectors[i][identity_index];
                    temp.convertTo(transforms_vectors[i][j], CV_64F);
                }
            }
        }
    }

    size_t TransformEstimator::findIdentityTransform(std::vector<cv::detail::CameraParams> transforms)
    {
        for (size_t k = 0; k < transforms.size(); ++k)
        {
            if (abs(transforms[k].R.at<float>(0, 0) - 1.0f) < ZERO_ERROR && abs(transforms[k].R.at<float>(0, 1) - 0.0f) < ZERO_ERROR && abs(transforms[k].R.at<float>(0, 2) - 0.0f) < ZERO_ERROR && abs(transforms[k].R.at<float>(1, 0) - 0.0f) < ZERO_ERROR && abs(transforms[k].R.at<float>(1, 1) - 1.0f) < ZERO_ERROR && abs(transforms[k].R.at<float>(1, 2) - 0.0f) < ZERO_ERROR && abs(transforms[k].R.at<float>(2, 0) - 0.0f) < ZERO_ERROR && abs(transforms[k].R.at<float>(2, 1) - 0.0f) < ZERO_ERROR && abs(transforms[k].R.at<float>(2, 2) - 1.0f) < ZERO_ERROR)
            {
                return k;
            }
        }
        ROS_ERROR("Identity transform does not exist, which should not happen.");
        return -1;
    }

    void TransformEstimator::printTransformsVectors(const std::vector<std::vector<cv::Mat>> transforms_vectors)
    {
        for (size_t i = 0; i < transforms_vectors.size(); ++i)
        {
            ROS_INFO("transform[%zu].size = %zu", i, transforms_vectors[i].size());

            for (size_t j = 0; j < transforms_vectors[i].size(); ++j)
            {
                ROS_INFO("transform[%zu][%zu] (width, height) = (%d, %d)", i, j, transforms_vectors[i][j].cols, transforms_vectors[i][j].rows);

                int width = transforms_vectors[i][j].cols;
                int height = transforms_vectors[i][j].rows;
                std::string s = "";
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        double val = transforms_vectors[i][j].at<double>(y, x);
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
            }
        }
    }
}
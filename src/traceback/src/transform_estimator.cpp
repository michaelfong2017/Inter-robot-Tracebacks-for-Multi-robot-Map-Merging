/*******************************************************************************
 * BSD 3-Clause License
 * 
 * Copyright (c) 2023, Fong Chun Him
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

#include <traceback/transform_estimator.h>
#include <traceback/estimation_internal.h>

#include <ros/assert.h>
#include <ros/console.h>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <numeric>

namespace traceback
{
    cv::detail::ImageFeatures TransformEstimator::computeFeatures(const cv::Mat &image, FeatureType feature_type)
    {
        auto finder = internal::chooseFeatureFinder(feature_type);
        cv::Ptr<cv::detail::FeaturesMatcher> matcher =
            cv::makePtr<cv::detail::AffineBestOf2NearestMatcher>();

        cv::detail::ImageFeatures image_features;
        cv::detail::computeImageFeatures(finder, image, image_features);
        return image_features;
    }

    double TransformEstimator::matchTwoFeatures(cv::detail::ImageFeatures &features1, cv::detail::ImageFeatures &features2, double confidence)
    {
        if (features1.getKeypoints().size() < 2 || features2.getKeypoints().size() < 2)
        {
            return -1.0;
        }

        std::vector<cv::detail::ImageFeatures> image_features = {features1, features2};
        std::vector<cv::detail::MatchesInfo> pairwise_matches;
        std::vector<cv::detail::CameraParams> transforms;
        std::vector<int> good_indices;
        cv::Ptr<cv::detail::FeaturesMatcher> matcher =
            cv::makePtr<cv::detail::AffineBestOf2NearestMatcher>();
        cv::Ptr<cv::detail::Estimator> estimator =
            cv::makePtr<cv::detail::AffineBasedEstimator>();
        cv::Ptr<cv::detail::BundleAdjusterBase> adjuster =
            cv::makePtr<cv::detail::BundleAdjusterAffinePartial>();

        /* find corespondent features */
        (*matcher)(image_features, pairwise_matches);
        matcher = {};

        // #ifndef NDEBUG
        // internal::writeDebugMatchingInfo(images_, image_features, pairwise_matches, "_", "_");
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
            return -1.0;
        }

        /* estimate transform */
        // ROS_DEBUG("calculating transforms in global reference frame");
        // note: currently used estimator never fails
        if (!(*estimator)(image_features, pairwise_matches, transforms))
        {
            return -1.0;
        }

        // for (auto &match_info : pairwise_matches)
        // {
        //     ROS_INFO("match_info %d, %d", match_info.src_img_idx, match_info.dst_img_idx);
        // }
        // for (auto &indice : good_indices)
        // {
        //     ROS_INFO("indice %d", indice);
        // }
        // ROS_INFO("transforms size %zu", transforms.size());

        for (auto &match_info : pairwise_matches)
        {
            if (match_info.H.empty() ||
                match_info.src_img_idx == match_info.dst_img_idx)
            {
                continue;
            }

            return match_info.confidence;
        }

        return -1.0;
    }

    void TransformEstimator::setTransformsVectors(std::vector<std::vector<cv::Mat>> transforms_vectors)
    {
        transforms_vectors_ = transforms_vectors;
    }

    void TransformEstimator::clearTransformsVectors()
    {
        transforms_vectors_.clear();
    }

    void TransformEstimator::setConfidences(std::vector<std::vector<double>> confidences)
    {
        confidences_ = confidences;
    }

    void TransformEstimator::clearConfidences()
    {
        confidences_.clear();
    }

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

        /** Debug contours */
        // size_t i = 0;
        // for (const cv::Mat &image : images_)
        // {
        //     findWeightedCenterOfConvexHulls(image, i);
        //     ++i;
        // }

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
        // internal::writeDebugMatchingInfo(images_, image_features, pairwise_matches, "_", "_");
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

            std::string s = "";
            for (int y = 0; y < 3; y++)
            {
                for (int x = 0; x < 3; x++)
                {
                    double val = transform.R.at<float>(y, x);
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
        }
        ROS_DEBUG("optimizing global transforms");
        adjuster->setConfThresh(confidence);
        if (!(*adjuster)(image_features, pairwise_matches, transforms))
        {
            ROS_WARN("Bundle adjusting failed. Could not estimate transforms.");
            return false;
        }

        // TEST change transforms
        // size_t identity_index = good_indices[findIdentityTransform(transforms)];
        // for (size_t i = 0; i < transforms.size(); ++i)
        // {
        //     if (good_indices[i] == identity_index)
        //         continue;

        //     if (i == 0)
        //     {
        //         transforms[i].R.at<float>(0, 2) += 12.0f * identity_index;
        //     }
        //     else
        //     {
        //         transforms[i].R.at<float>(1, 2) += 12.0f * i;
        //     }
        // }
        // TEST

        // Ensure consistency of transforms_vectors_, centers_ and confidences_
        {
            boost::lock_guard<boost::shared_mutex> lock(updates_mutex_);
            toPairwiseTransforms(transforms, good_indices, images_.size(), transforms_vectors_);
            // printTransformsVectors(transforms_vectors_);

            centers_.clear();
            centers_.resize(images_.size());
            size_t i = 0;
            for (const cv::Mat &image : images_)
            {
                centers_[i] = findWeightedCenterOfConvexHulls(image, i);
                ++i;
            }

            findPairwiseConfidences(pairwise_matches, good_indices, images_.size(), confidences_);
        }

        return true;
    }

    cv::Point2f TransformEstimator::findWeightedCenterOfConvexHulls(cv::Mat image, size_t image_index)
    {
        cv::blur(image, image, cv::Size(3, 3));
        cv::Mat canny_output;
        cv::Canny(image, canny_output, 100.0, 200.0);
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(canny_output, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        std::vector<std::vector<cv::Point>> hulls(contours.size());
        for (size_t i = 0; i < contours.size(); i++)
        {
            convexHull(contours[i], hulls[i]);
        }

        cv::Point2f center_contours, center_hulls;

        findWeightedCenter(contours, center_contours);
        findWeightedCenter(hulls, center_hulls);

        cv::Mat drawing = cv::Mat::zeros(canny_output.size(), CV_8UC3);
        for (size_t i = 0; i < contours.size(); i++)
        {
            cv::Scalar color = cv::Scalar(0, 100, 100, 100);
            drawContours(drawing, contours, (int)i, color, 2, cv::LINE_8, hierarchy, 0);
            drawContours(drawing, hulls, (int)i, cv::Scalar(100, 100, 0, 100), 2, cv::LINE_8, hierarchy, 0);
        }
        circle(drawing, center_contours, 4, cv::Scalar(0, 200, 200, 200), -1);
        circle(drawing, center_hulls, 4, cv::Scalar(200, 200, 0, 200), -1);

        cv::imwrite(std::to_string(image_index) + "_contour.png",
                    drawing);

        return center_hulls;
    }

    void TransformEstimator::findWeightedCenter(std::vector<std::vector<cv::Point>> contoursOrHulls, cv::Point2f &center)
    {
        std::vector<cv::Moments> mu(contoursOrHulls.size());
        for (size_t i = 0; i < contoursOrHulls.size(); i++)
        {
            mu[i] = moments(contoursOrHulls[i]);
        }
        std::vector<cv::Point2f> mc(contoursOrHulls.size());
        for (size_t i = 0; i < contoursOrHulls.size(); i++)
        {
            // add 1e-5 to avoid division by zero
            mc[i] = cv::Point2f(static_cast<float>(mu[i].m10 / (mu[i].m00 + 1e-5)),
                                static_cast<float>(mu[i].m01 / (mu[i].m00 + 1e-5)));
        }

        std::vector<double> perimeters(contoursOrHulls.size());
        for (size_t i = 0; i < contoursOrHulls.size(); i++)
        {
            perimeters[i] = cv::arcLength(contoursOrHulls[i], true);
        }

        double sum_x = 0;
        double sum_y = 0;
        for (size_t i = 0; i < contoursOrHulls.size(); i++)
        {
            sum_x += mc[i].x * perimeters[i];
            sum_y += mc[i].y * perimeters[i];
        }

        double total_length = std::accumulate(
            perimeters.begin(), perimeters.end(), // Run from begin to end
            0.0,                                  // Initialize with a zero point
            std::plus<double>()                   // Use addition for each point (default)
        );

        center = cv::Point2f(sum_x / total_length, sum_y / total_length);
    }

    void TransformEstimator::findPairwiseConfidences(std::vector<cv::detail::MatchesInfo> pairwise_matches, std::vector<int> good_indices, size_t images_size, std::vector<std::vector<double>> &confidences)
    {
        confidences.clear();
        confidences.resize(images_size);

        for (auto &confi : confidences)
        {
            confi.resize(images_size);
        }

        for (auto &match_info : pairwise_matches)
        {
            // Filter out (src_img_idx, dst_img_idx) == (-1, -1)
            if (match_info.H.empty() ||
                match_info.src_img_idx == match_info.dst_img_idx)
            {
                continue;
            }

            confidences[good_indices[match_info.src_img_idx]][good_indices[match_info.dst_img_idx]] = match_info.confidence;
        }

        // Set confidences of all self-transforms to -1
        for (size_t i = 0; i < images_size; ++i)
        {
            confidences[i][i] = -1.0;
        }

        // Confidences not explicitly set should be zeroes
    }

    // Input .R should already be CV_32F.
    void TransformEstimator::toPairwiseTransforms(std::vector<cv::detail::CameraParams> transforms, std::vector<int> good_indices, size_t images_size, std::vector<std::vector<cv::Mat>> &transforms_vectors)
    {
        transforms_vectors.clear();
        transforms_vectors.resize(images_size);

        for (auto &trans : transforms_vectors)
        {
            trans.resize(images_size);
        }

        // e.g. good_indices is [1, 2] and if transforms[1] is the identity transform, image 2 will then be the reference image with identity transform.
        size_t identity_index = good_indices[findIdentityTransform(transforms)];

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
            invertAffineTransform(transforms[i].R.rowRange(0, 2), temp);
            float data[3] = {0.0f, 0.0f, 1.0f};
            cv::Mat row(1, 3, CV_32F, data);
            temp.push_back(row);

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
                else if (std::find(good_indices.begin(), good_indices.end(), static_cast<int>(i)) != good_indices.end() && std::find(good_indices.begin(), good_indices.end(), static_cast<int>(j)) != good_indices.end())
                {
                    cv::Mat temp = transforms_vectors[identity_index][j] * transforms_vectors[i][identity_index];
                    temp.convertTo(transforms_vectors[i][j], CV_64F);
                }
            }
        }
    }

    // Not just multiply transformation matrix because we are dealing with differences between initial poses.
    void TransformEstimator::updateBestTransforms(cv::Mat tracer_to_traced, std::string tracer, std::string traced, std::unordered_map<std::string, std::unordered_map<std::string, cv::Mat>> &best_transforms,
                                                  std::unordered_set<std::string> &has_best_transforms)
    {
        if (has_best_transforms.count(tracer) && has_best_transforms.count(traced))
        {
            best_transforms[tracer][traced] = tracer_to_traced.clone();
            cv::Mat temp;
            invertAffineTransform(tracer_to_traced.rowRange(0, 2), temp);
            double data[3] = {0.0, 0.0, 1.0};
            cv::Mat row(1, 3, CV_64F, data);
            temp.push_back(row);
            best_transforms[traced][tracer] = temp;
            return;
        }
        // 1
        if (best_transforms[tracer][traced].empty())
        {
            best_transforms[tracer][traced] = tracer_to_traced.clone();
        }
        // 2
        if (best_transforms[traced][tracer].empty())
        {
            cv::Mat temp;
            invertAffineTransform(tracer_to_traced.rowRange(0, 2), temp);
            double data[3] = {0.0, 0.0, 1.0};
            cv::Mat row(1, 3, CV_64F, data);
            temp.push_back(row);
            best_transforms[traced][tracer] = temp;
        }
        // 3
        if (!has_best_transforms.count(tracer) && !has_best_transforms.count(traced))
        {
            best_transforms[tracer][tracer] = cv::Mat::eye(3, 3, CV_64F);
            best_transforms[traced][traced] = cv::Mat::eye(3, 3, CV_64F);
            has_best_transforms.insert(tracer);
            has_best_transforms.insert(traced);
            return;
        }
        // Update traced->1, traced->2, 1->traced, 2->traced, etc, if tracer->1, tracer->2, etc exists
        // e.g. traced->1 = tracer->1 * traced->tracer
        // e.g. 1->traced = tracer->traced * 1->tracer
        else if (has_best_transforms.count(tracer) && !has_best_transforms.count(traced))
        {
            best_transforms[traced][traced] = cv::Mat::eye(3, 3, CV_64F);

            for (auto it = has_best_transforms.begin(); it != has_best_transforms.end(); ++it)
            {
                std::string k = *it;
                if (!best_transforms[tracer][k].empty())
                {
                    best_transforms[traced][k] = best_transforms[tracer][k] * best_transforms[traced][tracer];
                }
                if (!best_transforms[k][tracer].empty())
                {
                    best_transforms[k][traced] = best_transforms[tracer][traced] * best_transforms[k][tracer];
                }
            }

            has_best_transforms.insert(traced);
        }
        // Update tracer->1, tracer->2, 1->tracer, 2->tracer, etc, if traced->1, traced->2, etc exists
        // e.g. tracer->1 = traced->1 * tracer->traced
        // e.g. 1->tracer = traced->tracer * 1->traced
        else if (has_best_transforms.count(traced) && !has_best_transforms.count(tracer))
        {
            best_transforms[tracer][tracer] = cv::Mat::eye(3, 3, CV_64F);

            for (auto it = has_best_transforms.begin(); it != has_best_transforms.end(); ++it)
            {
                std::string k = *it;
                if (!best_transforms[traced][k].empty())
                {
                    best_transforms[tracer][k] = best_transforms[traced][k] * best_transforms[tracer][traced];
                }
                if (!best_transforms[k][traced].empty())
                {
                    best_transforms[k][tracer] = best_transforms[traced][tracer] * best_transforms[k][traced];
                }
            }

            has_best_transforms.insert(tracer);
        }
    }

    void TransformEstimator::printConfidences(const std::vector<std::vector<double>> confidences)
    {
        for (size_t i = 0; i < confidences.size(); ++i)
        {
            ROS_INFO("confidences[%zu].size = %zu", i, confidences[i].size());

            for (size_t j = 0; j < confidences[i].size(); ++j)
            {
                ROS_INFO("confidences[%zu][%zu] = %f", i, j, confidences[i][j]);
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
            ROS_INFO("transforms[%zu].size = %zu", i, transforms_vectors[i].size());

            for (size_t j = 0; j < transforms_vectors[i].size(); ++j)
            {
                ROS_INFO("transforms[%zu][%zu] (width, height) = (%d, %d)", i, j, transforms_vectors[i][j].cols, transforms_vectors[i][j].rows);

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

    std::vector<std::vector<cv::Mat>> TransformEstimator::getTransformsVectors()
    {
        return transforms_vectors_;
    }

    std::vector<cv::Point2f> TransformEstimator::getCenters()
    {
        return centers_;
    }

    std::vector<std::vector<double>> TransformEstimator::getConfidences()
    {
        return confidences_;
    }
}
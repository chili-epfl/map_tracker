
#include "Map_Tracker.hpp"

#include <vector>
#include <stdexcept>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace map_tracker
{

static const std::vector<cv::Scalar> colorPalette = {
    cv::Scalar(76, 114, 176),
    cv::Scalar(85, 168, 104),
    cv::Scalar(196, 78, 82),
    cv::Scalar(129, 114, 178),
    cv::Scalar(204, 185, 116),
    cv::Scalar(100, 181, 205)
};

Timer::Timer()
    : mTicks{0}
    , mIndex(0)
    , mFps(-1.0)
{}

void Timer::tic()
{
    std::clock_t current = std::clock();
    std::clock_t prev = mTicks[mIndex];
    mTicks[mIndex] = current;
    ++mIndex;
    if(mIndex >= N)
        mIndex = 0;

    if(prev != 0)
        mFps = CLOCKS_PER_SEC * N / static_cast<double>(current - prev);
}

void CalibrationInfo::clear()
{
    objectPoints.clear();
    imagePoints.clear();
}

void drawPointsAndIds(cv::Mat& inputImage, const std::vector<DetectionGH>& matches)
{
    //draw Id
    for(unsigned int i = 0; i < matches.size(); ++i)
    {
        char pointIdStr[100];
        sprintf(pointIdStr, "%d", matches[i].id);
        circle(inputImage, matches[i].position, 4, cvScalar(0, 250, 250), -1, 8, 0);
        putText(inputImage, pointIdStr, matches[i].position, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250,250,250), 1, CV_AA);
    }
}

void drawAxes(cv::Mat& image, const cv::Mat& orientation)
{
    static const cv::Scalar axes_colors[] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255)};
    cv::Size::value_type width = image.size().width;
    //cv::Point2d center(width * 0.1, width * 0.1);
    cv::Point2d center(width * 0.5, image.size().height * 0.5);
    double length = width * 0.05;


    float sizecercle = 10;
    circle(image, center, sizecercle, cv::Scalar(255,255,255),2);

    for(int i = 0; i < 3; ++i)
    {
        //const cv::Point2d direction(orientation.at<float>(i, 1), orientation.at<float>(i, 0));
        const cv::Point2d direction(orientation.at<float>(0, i), orientation.at<float>(1, i));
        const cv::Point2d arrow = center - length * direction;
        const cv::Scalar& color = axes_colors[i];

        cv::line(image, center, arrow, color,2);

        circle(image, arrow, sizecercle + 8 * orientation.at<float>(2, i), color,2);
    }
}

Map_Tracker::Map_Tracker(const std::string& configPath)
{
    std::string configFile; configFile = configPath + "Config.xml";
    cv::FileStorage fs(configFile, cv::FileStorage::READ);

    if(!fs.isOpened())
    {
        std::cerr << "Could not open configFile " << configFile << std::endl;
        throw std::runtime_error("Configuration file not found!");
    }

    std::string calibrationFile;
    fs["calibrationFile"]>> calibrationFile;
    calibrationFile = configPath + calibrationFile;


    std::vector<std::string> landmarkFiles;
    cv::FileNode nLm = fs["landmarkFiles"];
    cv::FileNodeIterator it = nLm.begin(), it_end = nLm.end(); // Go through the node
    for (; it != it_end; ++it)
    {
        std::string landmarkFile = (std::string)*it;
        landmarkFile = configPath + landmarkFile;
        landmarkFiles.push_back(landmarkFile);
    }

    init(calibrationFile, landmarkFiles);

}



void Map_Tracker::init(const std::string& calibrationFile, const std::vector<std::string>& landmarkFiles)
{

    cv::FileStorage calibrationStorage(calibrationFile, cv::FileStorage::READ);
    if(!calibrationStorage.isOpened())
    {
        std::cerr << "Could not open " << calibrationFile << std::endl;
        throw std::runtime_error("Calibration file not found!");
    }

    std::vector<cv::FileStorage> landmarkStorages;
    for(auto& landmarkFile : landmarkFiles)
    {
        cv::FileStorage fs(landmarkFile, cv::FileStorage::READ);
        if(!fs.isOpened())
            throw std::runtime_error("Marker file not found");
        landmarkStorages.push_back(fs);
    }

    init(calibrationStorage, landmarkStorages);
}

Map_Tracker::Map_Tracker(cv::FileStorage& calibrationStorage, std::vector<cv::FileStorage>& landmarkStorages)
{
    init(calibrationStorage, landmarkStorages);
}

void Map_Tracker::init(cv::FileStorage& calibrationStorage,
                         std::vector<cv::FileStorage>& landmarkStorages)
{
    mDetectionInfo.init(landmarkStorages.size());
    mFeatureExtractor = cv::BRISK::create();

    readCalibrationFromFileStorage(calibrationStorage, mCalibration);


    // Load landmarks
    for(auto& landmarkStorage : landmarkStorages)
        mLandmarks.push_back(Landmark::fromFileStorage(landmarkStorage));
}

void Map_Tracker::resizeCalibration(const cv::Size& imgSize)
{
    // loadCalibration(mCalibrationFile, imgSize, &mCalibration);
    rescaleCalibration(mCalibration, imgSize);
}




void Map_Tracker::writeCalibration(cv::FileStorage& output)
{
    writeCalibrationToFileStorage(mCalibration,output);

}



void Map_Tracker::updateLandmarks(const cv::Mat& input,
                           const cv::Mat* deviceOrientation)
{
    if(input.size() != mCalibration.imageSize)
        resizeCalibration(input.size());


    // Landmark detection and tracking
    static int counter = 100;

    if(!mDetectionInfo.prevImageLandm.empty())
    {
        ++counter;

        //check if all the landmarks are tracked
        bool allTracked = true;
        auto lmcDetectionsIt = mDetectionInfo.landmarkDetections.cbegin();
        for(; lmcDetectionsIt != mDetectionInfo.landmarkDetections.cend(); ++lmcDetectionsIt)
        {
            const cv::Mat& h = lmcDetectionsIt->getHomography();
            if(h.empty())
                allTracked = false;
        }


        // Extract features only once every 20 frames and only if need to do any detection (ie all markers are not tracked)
        std::vector<cv::KeyPoint> detectedKeypoints;
        cv::Mat detectedDescriptors;
        if(!allTracked && counter >= 20)
        {
            mFeatureExtractor->detectAndCompute(input, cv::noArray(),
                                                detectedKeypoints, detectedDescriptors);
            counter = 0;
        }

        auto landmarksIt = mLandmarks.cbegin();
        auto lmDetectionsIt = mDetectionInfo.landmarkDetections.begin();
        for(; landmarksIt != mLandmarks.cend(); ++landmarksIt, ++lmDetectionsIt)
            landmarksIt->find(input, mDetectionInfo.prevImageLandm, mCalibration, detectedKeypoints, detectedDescriptors, *lmDetectionsIt);
    }

    input.copyTo(mDetectionInfo.prevImageLandm);

    mTimer.tic();
}

bool Map_Tracker::updateCalibration()
{
    //for each tracked landmark add the matches to the calibration tool
    //ie for each image and each landmark, the set of 3D points and their projections
    auto landmarksIt = mLandmarks.cbegin();
    auto lmDetectionsIt = mDetectionInfo.landmarkDetections.begin();

    for(; landmarksIt != mLandmarks.cend(); ++landmarksIt, ++lmDetectionsIt)
    {
        const cv::Mat& h = lmDetectionsIt->getHomography();
        if(!h.empty() && lmDetectionsIt->getCorrespondences().size()>100)
        {
            std::vector<cv::Point3f> lmObjectPoints;
            std::vector<cv::Point2f> lmImagePoints;

            float scale = landmarksIt->getRealSize().width/landmarksIt->getImage().size().width;
            auto correspIt = lmDetectionsIt->getCorrespondences().cbegin();
            for(; correspIt != lmDetectionsIt->getCorrespondences().cend(); ++correspIt)
            {
                lmImagePoints.push_back(correspIt->second);
                cv::Point3f lmPoint = cv::Point3f(scale*landmarksIt->getKeypointPos()[correspIt->first].x,scale*landmarksIt->getKeypointPos()[correspIt->first].y,0.);
                lmObjectPoints.push_back(lmPoint);
            }


            mCalibrationInfo.objectPoints.push_back(lmObjectPoints);
            mCalibrationInfo.imagePoints.push_back(lmImagePoints);
        }
    }

    if(mCalibrationInfo.objectPoints.size() >= mCalibrationInfo.nbFramesForCalibration)
    {
        std::vector<cv::Mat> rotationVectors;
        std::vector<cv::Mat> translationVectors;

        cv::Mat distortionCoefficients = cv::Mat::zeros(8, 1, CV_64F); // There are 8 distortion coefficients
        cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

        int flags = 0;
        double rms = calibrateCamera(mCalibrationInfo.objectPoints, mCalibrationInfo.imagePoints, mCalibration.imageSize, mCalibration.cameraMatrix,
                      mCalibration.distCoeffs, rotationVectors, translationVectors, flags|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);

        std::cout<<"camera calibration RMS = "<<rms<<std::endl;

        mCalibrationInfo.clear();
        return true;
    } else {
        return false;
    }
}

void Map_Tracker::drawLastDetection(cv::Mat* output, cv::Mat* deviceOrientation) const
{
    // mDetectionInfo.image.copyTo(*output);

    //plot FPS
    char fpsStr[100];
    sprintf(fpsStr, "%0.1f fps ", getTimer().getFps());
    putText(*output, fpsStr,
                cv::Point2i(10,output->size().height-10),
                cv::FONT_HERSHEY_COMPLEX_SMALL,
                0.8, cvScalar(250,250,250), 1, CV_AA);



    //check compatibility of landmark pose and device orientation and plot device orientation
     //if(deviceOrientation)
     if(0)
     {
        //transform orientation matrix into opencv matrix style
        cv::Mat orientationCvStandard(3, 3, CV_32F);
        //for us, orientation amtrix should be one which we multiply world coordinates with to get camera coord
        //the one we get is the inverse of that, and x and y axes are switched, and axis all inverted
        for(int i=0;i<3;i++)orientationCvStandard.at<float>(0, i) = -deviceOrientation->at<float>(i, 1);
        for(int i=0;i<3;i++)orientationCvStandard.at<float>(1, i) = -deviceOrientation->at<float>(i, 0);
        for(int i=0;i<3;i++)orientationCvStandard.at<float>(2, i) = -deviceOrientation->at<float>(i, 2);

         drawAxes(*output, orientationCvStandard);

         //orientation is a 3x3 matrix with for each column the coordinates in the camera frame of the x,y and z axes
         //let's see how much first landmark normal agrees with z axis
         cv::Affine3d poseLm0 = mDetectionInfo.landmarkDetections[0].getPose();

         //this pose is applied to a world coordinate to transform it into camera coordinate
         //so if apply vector [0 0 1] to its rotation, then should get z vector expressed in camera frame
         cv::Vec3d z_w(0.,0.,1.);
         cv::Vec3d z_cl(poseLm0.rotation()(0,2),poseLm0.rotation()(1,2),poseLm0.rotation()(2,2));
         cv::Vec3d z_co(orientationCvStandard.at<float>(0, 2),orientationCvStandard.at<float>(1, 2),orientationCvStandard.at<float>(2, 2));

         //compare with orientation from IMU
         float z_agree = z_cl.dot(z_co);

         //plot it
        char zaStr[100];
        sprintf(zaStr, "%0.1f za ", z_agree);
        putText(*output, zaStr,
                    cv::Point2i(output->size().width - 40,output->size().height-10),
                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                    0.8, cvScalar(250,250,250), 1, CV_AA);

        //debug
        static const cv::Scalar axes_colors[] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255)};
        cv::Size::value_type width = output->size().width;
        //cv::Point2d center(width * 0.1, width * 0.1);
        cv::Point2d center(width * 0.25, output->size().height * 0.5);
        double length = width * 0.05;

        float sizecercle = 5;
        circle(*output, center, sizecercle, cv::Scalar(255,255,255),2);


        for(int i = 0; i < 3; ++i)
        {
            const cv::Point2d direction(poseLm0.rotation()(0, i), poseLm0.rotation()(1, i));
            const cv::Point2d arrow = center - length * direction;
            const cv::Scalar& color = axes_colors[i];

            cv::line(*output, center, arrow, color, 2);

            circle(*output, arrow, sizecercle + 3*poseLm0.rotation()(2, i), color,2);
        }

     }

    // Draw landmark detections
    std::vector<cv::Point2f> corners(4);

    auto lmDetectionsIt = mDetectionInfo.landmarkDetections.cbegin();
    auto landmarksIt = mLandmarks.cbegin();
    auto colorIt = colorPalette.cbegin();
    int cpt = -1;
    int cpt_plot = 0;
    for(; landmarksIt != mLandmarks.cend(); ++landmarksIt, ++lmDetectionsIt, ++colorIt)
    {
        cpt ++;
        const Landmark& landmark = *landmarksIt;
        const cv::Mat& h = lmDetectionsIt->getHomography();

        // Reset the color iterator if needed
        if(colorIt == colorPalette.cend())
            colorIt = colorPalette.cbegin();

        if(h.empty())
            continue;

        cv::perspectiveTransform(landmark.getCorners(), corners, h);
        cv::line(*output, corners[0], corners[1], *colorIt, 2);
        cv::line(*output, corners[1], corners[2], *colorIt, 2);
        cv::line(*output, corners[2], corners[3], *colorIt, 2);
        cv::line(*output, corners[3], corners[0], *colorIt, 2);

        for(auto c : lmDetectionsIt->getCorrespondences())
        {
            cv::Point2f p = c.second;
            cv::circle(*output, p, 2, cv::Scalar(0, 255, 255));
        }

        //draw pose
        //draw object frame (axis XYZ)
        std::vector<cv::Point3f> framePoints;
        framePoints.push_back(cv::Point3f(0,0,0));
        framePoints.push_back(cv::Point3f(0.03,0,0));
        framePoints.push_back(cv::Point3f(0,0.03,0));
        framePoints.push_back(cv::Point3f(0,0,0.03));

        //cv::Affine3d pose = lmDetectionsIt->getPose().inv();
        cv::Affine3d pose = lmDetectionsIt->getPose();
        std::vector<cv::Point2f> vprojVertices;
        cv::projectPoints(framePoints, pose.rvec(), pose.translation(), mCalibration.cameraMatrix, mCalibration.distCoeffs, vprojVertices);
        cv::line(*output, vprojVertices[0], vprojVertices[1], cv::Scalar(255,0,0), 2);
        cv::line(*output, vprojVertices[0], vprojVertices[2], cv::Scalar(0,255,0), 2);
        cv::line(*output, vprojVertices[0], vprojVertices[3], cv::Scalar(0,255,255), 2);

        //print confidence
        char confStr[100];
        sprintf(confStr, "c[%d]: %0.1f", cpt,lmDetectionsIt->getConfidence());
        putText(*output, confStr,
                    cv::Point2i(output->size().width - 100,20+20*cpt_plot),
                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                    0.8, cvScalar(250,250,250), 1, CV_AA);
        cpt_plot++;

    }
}

}

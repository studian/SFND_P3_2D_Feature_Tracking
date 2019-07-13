/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc

    // ...modified start: MP.1 Data Buffer Optimization
    // dataBufferSize: no. of images which are held in memory (ring buffer) at the same time
    int dataBufferSize = 3; // default, original code: int dataBufferSize = 2;
    // ...modified end: MP.1 Data Buffer Optimization

    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    //.. add start: MP.7, MP.8, and MP.9
    vector<string> detector_type_names = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
    vector<string> descriptor_type_names = {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};

    ofstream detector_file;
    detector_file.open ("../MP_7_Counts_Keypoints.csv");

    ofstream det_des_matches;
    det_des_matches.open ("../MP_8_Counts_Matched_Keypoints.csv");

    ofstream det_des_time;
    det_des_time.open ("../MP_9_Log_Time.csv");    

    for(auto detector_type_name:detector_type_names) // start loop detector_types
    {
        bool write_detector = false;

        for(auto descriptor_type_name:descriptor_type_names) // start loop descriptor_types
        {
            if(detector_type_name.compare("AKAZE")!=0 && descriptor_type_name.compare("AKAZE")==0)
                continue;

            if(detector_type_name.compare("AKAZE")==0 && descriptor_type_name.compare("AKAZE")==0)
                continue;    

            dataBuffer.clear();
            
            cout << "===================================================================" << endl;
            cout << "Detector Type: " << detector_type_name << "   Descriptor Type: " << descriptor_type_name << endl;
            cout << "===================================================================" << endl;

            //.. add start: MP.7 Performance Evaluation 1
            // Write to detector keypoints number file
            if(!write_detector)
            {
                detector_file << detector_type_name;
            }                
            //.. add end: MP.7 Performance Evaluation 1

            //.. add start: MP.8 Performance Evaluation 2
            det_des_matches << detector_type_name << "_" << descriptor_type_name;
            //.. add end: MP.8 Performance Evaluation 2

            //.. add start: MP.9 Performance Evaluation 3
            det_des_time << detector_type_name << "_" << descriptor_type_name;
            //.. add end: MP.9 Performance Evaluation 3

            //.. add end: MP.7, MP.8, and MP.9

            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
            {
                //.. add start: MP.9 Performance Evaluation 3
                double t = (double)cv::getTickCount();
                //.. add end: MP.9 Performance Evaluation 3

                /* LOAD IMAGE INTO BUFFER */

                // assemble filenames for current index
                ostringstream imgNumber;
                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

                // load image from file and convert to grayscale
                cv::Mat img, imgGray;
                img = cv::imread(imgFullFilename);
                cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

                //// STUDENT ASSIGNMENT
                //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

                // push image into data frame buffer
                DataFrame frame;
                frame.cameraImg = imgGray;

                // ...add start: MP.1 Data Buffer Optimization
                if (  dataBuffer.size()+1 > dataBufferSize) 
                {
                    dataBuffer.erase(dataBuffer.begin());
                    cout << "REPLACE IMAGE IN BUFFER done" << endl;
                }
                // ...add end: MP.1 Data Buffer Optimization

                dataBuffer.push_back(frame);

                //// EOF STUDENT ASSIGNMENT
                cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

                /* DETECT IMAGE KEYPOINTS */

                // extract 2D keypoints from current image
                vector<cv::KeyPoint> keypoints; // create empty feature list for current image

                // ...modified start: MP.7, MP.8, MP.9
                string detectorType = detector_type_name; //"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"
                // ...modified end: MP.7, MP.8, MP.9

                //// STUDENT ASSIGNMENT
                //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
                //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

                if (detectorType.compare("SHITOMASI") == 0)
                {
                    detKeypointsShiTomasi(keypoints, imgGray, false);
                }
                // ...add start: MP.2 Keypoint Detection
                // detectorType = HARRIS
                else if (detectorType.compare("HARRIS") == 0) 
                {
                    detKeypointsHarris(keypoints, imgGray, false);
                }
                // Modern detector types, detectorType = FAST, BRISK, ORB, AKAZE, SIFT
                else if (detectorType.compare("FAST")  == 0 ||
                        detectorType.compare("BRISK") == 0 ||
                        detectorType.compare("ORB")   == 0 ||
                        detectorType.compare("AKAZE") == 0 ||
                        detectorType.compare("SIFT")  == 0)
                {
                    detKeypointsModern(keypoints, imgGray, detectorType, false);
                }
                else
                {
                    throw invalid_argument(detectorType + " is not a valid detectorType. Try SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT.");
                }
                // ...add end: MP.2 Keypoint Detection
                //// EOF STUDENT ASSIGNMENT

                //// STUDENT ASSIGNMENT
                //// TASK MP.3 -> only keep keypoints on the preceding vehicle

                // only keep keypoints on the preceding vehicle
                bool bFocusOnVehicle = true;
                cv::Rect vehicleRect(535, 180, 180, 150);

                // ...add start: MP.3 Keypoint Removal
                vector<cv::KeyPoint>::iterator keypoint;
                vector<cv::KeyPoint> keypoints_roi;
                // ...add end: MP.3 Keypoint Removal

                if (bFocusOnVehicle)
                {
                    // ...add start: MP.3 Keypoint Removal
                    for(keypoint = keypoints.begin(); keypoint != keypoints.end(); ++keypoint)
                    {
                        if (vehicleRect.contains(keypoint->pt))
                        {  
                            cv::KeyPoint newKeyPoint;
                            newKeyPoint.pt = cv::Point2f(keypoint->pt);
                            newKeyPoint.size = 1;
                            keypoints_roi.push_back(newKeyPoint);
                        }
                    }

                    keypoints =  keypoints_roi;
                    cout << "IN ROI n= " << keypoints.size()<<" keypoints"<<endl;
                    // ...add end: MP.3 Keypoint Removal
                }

                //.. add start: MP.7 Performance Evaluation 1
                if(!write_detector)
                {
                    detector_file  << ", " << keypoints.size();
                }                
                //.. add end: MP.7 Performance Evaluation 1

                //// EOF STUDENT ASSIGNMENT

                // optional : limit number of keypoints (helpful for debugging and learning)
                bool bLimitKpts = false;
                if (bLimitKpts)
                {
                    int maxKeypoints = 50;

                    if (detectorType.compare("SHITOMASI") == 0)
                    { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                    }
                    cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                    cout << " NOTE: Keypoints have been limited!" << endl;
                }

                // push keypoints and descriptor for current frame to end of data buffer
                (dataBuffer.end() - 1)->keypoints = keypoints;
                cout << "#2 : DETECT KEYPOINTS done" << endl;

                /* EXTRACT KEYPOINT DESCRIPTORS */

                //// STUDENT ASSIGNMENT
                //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
                //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

                cv::Mat descriptors;

                // ...modified start: MP.7, MP.8, MP.9
                string descriptorType = descriptor_type_name; // BRIEF, ORB, FREAK, AKAZE, SIFT
                // ...modified end: MP.7, MP.8, MP.9

                descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
                //// EOF STUDENT ASSIGNMENT

                // push descriptors for current frame to end of data buffer
                (dataBuffer.end() - 1)->descriptors = descriptors;

                cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

                if (dataBuffer.size() > 1) // wait until at least two images have been processed
                {
                    /* MATCH KEYPOINT DESCRIPTORS */

                    vector<cv::DMatch> matches;
                    string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
                    //string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG                    
                    
                    //.. add start: MP.4 Keypoint Descriptors
                    string descriptorType;
                    if (descriptorType.compare("SIFT") == 0) 
                    {
                        descriptorType == "DES_HOG";
                    }
                    else
                    {
                        descriptorType == "DES_BINARY";
                    }                    
                    //.. add end: MP.4 Keypoint Descriptors              

                    //.. modified start: MP.6 Descriptor Distance Ratio
                    string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN
                    //.. modified end: MP.6 Descriptor Distance Ratio

                    //// STUDENT ASSIGNMENT
                    //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
                    //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

                    matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                    (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                    matches, descriptorType, matcherType, selectorType);

                    //// EOF STUDENT ASSIGNMENT

                    // store matches in current data frame
                    (dataBuffer.end() - 1)->kptMatches = matches;

                    //.. add start: MP.8 Performance Evaluation 2
                    det_des_matches << ", " << matches.size();
                    //.. add end: MP.8 Performance Evaluation 2

                    //.. add start: MP.9 Performance Evaluation 3
                    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
                    det_des_time << ", " << 1000*t;
                    //.. add end: MP.9 Performance Evaluation 3

                    cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                    // visualize matches between current and previous image
                    bVis = false; //true
                    if (bVis)
                    {
                        cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                        cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                        (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                        matches, matchImg,
                                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                                        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                        string windowName = "Matching keypoints between two camera images";
                        cv::namedWindow(windowName, 7);
                        cv::imshow(windowName, matchImg);
                        cout << "Press key to continue to next image" << endl;
                        cv::waitKey(0); // wait for key to be pressed
                    }
                    bVis = false;
                }

            } // eof loop over all images

            //.. add start: MP.7, MP.8, and MP.9
            if(!write_detector)
            {
                detector_file << endl;   
            }
            
            write_detector = true;

            det_des_matches << endl;
            det_des_time << endl;
        }// eof loop over descriptor_types
    }// eof loop over detector_types

    detector_file.close();
    det_des_matches.close();
    det_des_time.close();
    //.. add end: MP.7, MP.8, and MP.9

    return 0;
}

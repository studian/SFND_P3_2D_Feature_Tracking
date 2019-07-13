# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson. 
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures. 

See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning. 

## [Rubric](https://review.udacity.com/#!/rubrics/2549/view) Points
---
### 1. Data Buffer

#### MP.1 Data Buffer Optimization
* Implement a vector for dataBuffer objects whose size does not exceed a limit (e.g. 2 elements). 
* This can be achieved by pushing in new elements on one end and removing elements on the other end.
* Solution: Lines 40 ~ 43 at `MidTermProject_Camera_Student.cpp`
```c++
// ...add start: MP.1 Data Buffer Optimization
if (  dataBuffer.size()+1 > dataBufferSize) 
{
    dataBuffer.erase(dataBuffer.begin());
    cout << "REPLACE IMAGE IN BUFFER done" << endl;
}
// ...add end: MP.1 Data Buffer Optimization
```
* Solution code: Lines 120 ~ 130 at `MidTermProject_Camera_Student.cpp`
```c++
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
 
```
### 2. Keypoints

#### MP.2 Keypoint Detection
* Implement detectors HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT and make them selectable by setting a string accordingly.
* Solution code: Lines 154 ~ 173 at `MidTermProject_Camera_Student.cpp`
```c++
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
```
* Solution code: Lines 162 ~ 285 at `matching2D_Student.cpp`
```c++
// ...add start: MP.2 Keypoint Detection
// detectorType = HARRIS
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    double t = (double)cv::getTickCount();

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Look for prominent corners and instantiate keypoints

    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof loop over rows

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris corner detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// detectorType = FAST, BRISK, ORB, AKAZE, SIFT
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    double t = (double)cv::getTickCount();
    cv::Ptr<cv::Feature2D> detector;

    if(detectorType.compare("FAST") == 0)
    {
        detector = cv::FastFeatureDetector::create();
        detector->detect(img, keypoints);
    }
    else if(detectorType.compare("BRISK") == 0)
    {
        detector = cv::BRISK::create();
        detector->detect(img, keypoints);
    }
    else if(detectorType.compare("ORB") == 0)
    {
        detector = cv::ORB::create();
        detector->detect(img, keypoints);
    }
    else if(detectorType.compare("AKAZE") == 0)
    {
        detector = cv::AKAZE::create();
        detector->detect(img, keypoints);   
    }
    else if(detectorType.compare("SIFT") == 0)
    {
        detector = cv::xfeatures2d::SIFT::create();
        detector->detect(img, keypoints);        
    }
    else
    {
        throw invalid_argument(detectorType + " is not a valid detectorType. Try FAST, BRISK, ORB, AKAZE, SIFT.");
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Keypoint Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
// ...add end: MP.2 Keypoint Detection
```

#### MP.3 Keypoint Removal
* Remove all keypoints outside of a pre-defined rectangle and only use the keypoints within the rectangle for further processing.
* Solution code: Lines 183 ~ 205 at `MidTermProject_Camera_Student.cpp`
```c++
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
```
### 3. Descriptors

#### MP.4 Keypoint Descriptors
* Implement descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and make them selectable by setting a string accordingly.
* Solution code: Lines 262 ~ 272 at `MidTermProject_Camera_Student.cpp`
```c++
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
```
* Solution code: In function `descKeypoints`, Lines 87 ~ 112 at `matching2D_Student.cpp`
```c++
// ...add start: MP.4 Keypoint Descriptors
else if(descriptorType.compare("BRIEF") == 0)
{
    extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
}
else if(descriptorType.compare("ORB") == 0)
{
    extractor = cv::ORB::create();
}
else if(descriptorType.compare("AKAZE") == 0)
{
    extractor = cv::AKAZE::create();
}
else if(descriptorType.compare("FREAK") == 0)
{
    extractor = cv::xfeatures2d::FREAK::create();
}
else if(descriptorType.compare("SIFT") == 0)
{
    extractor = cv::xfeatures2d::SIFT::create();
}
else
{
    throw invalid_argument( "The input method is not supported. Try BRIEF, BRISK, ORB, AKAZE, FREAK, SIFT." );
}
// ...add end: MP.4 Keypoint Descriptors
```

#### MP.5 Descriptor Matching
* Implement FLANN matching as well as k-nearest neighbor selection. 
* Both methods must be selectable using the respective strings in the main function.
* Solution code: In function `matchDescriptors`, Lines 14 ~ 45 at `matching2D_Student.cpp`
```c++
/*
if (matcherType.compare("MAT_BF") == 0)
{
    int normType = cv::NORM_HAMMING;
    matcher = cv::BFMatcher::create(normType, crossCheck);
}
*/

// ...add start: MP.5 Descriptor Matching
if (matcherType.compare("MAT_BF") == 0)
{
    int normType = cv::NORM_L2;

    if(descriptorType.compare("DES_BINARY") == 0)
    {
        normType = cv::NORM_HAMMING;        
    }        
    matcher = cv::BFMatcher::create(normType, crossCheck);
    cout << "BF matching cross-check=" << crossCheck;
}    
else if (matcherType.compare("MAT_FLANN") == 0)
{
    // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
    if (descSource.type() != CV_32F)
    { 
        descSource.convertTo(descSource, CV_32F);
        descRef.convertTo(descRef, CV_32F);
    }

    matcher = cv::FlannBasedMatcher::create();              
}
// ...add end: MP.5 Descriptor Matching
```
#### MP.6 Descriptor Distance Ratio
* Use the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints.
* Solution code: Lines 274 ~ 276 at `MidTermProject_Camera_Student.cpp`
```c++
//.. modified start: MP.6 Descriptor Distance Ratio
string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN
//.. modified end: MP.6 Descriptor Distance Ratio
```
* Solution code: In function `matchDescriptors`, Lines 53 ~ 70 at `matching2D_Student.cpp`
```
else if (selectorType.compare("SEL_KNN") == 0)
{ // k nearest neighbors (k=2)

    // ...add start: MP.6 Descriptor Distance Ratio
    vector<vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descSource, descRef, knn_matches, 2);

    double minDescDistRatio = 0.8;
    for(auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
    {
        if( ((*it)[0].distance) < ((*it)[1].distance * minDescDistRatio) )
        {
            matches.push_back((*it)[0]);
        }                
    }
    cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
    // ...add end: MP.6 Descriptor Distance Ratio
}
```

### 4. Performance
---
#### MP.7 Performance Evaluation 1
* Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. 
* Do this for all the detectors you have implemented.
* Solution Result: please check `MP_7_Counts_Keypoints.csv` file [CSV file](https://github.com/studian/SFND_P3_2D_Feature_Tracking/MP_7_Counts_Keypoints.csv).

DETECTOR  | Number of keypoints
--------  | -------------------
SHITOMASI | 111 ~ 125
HARRIS    |  14 ~  43
FAST      | 386 ~ 427
BRISK     | 254 ~ 297
ORB       |  92 ~ 130
AKAZE     | 155 ~ 179
SIFT      | 124 ~ 159

HARRIS detector has the smallest amount of keypoints.
FAST detector has the bigest amount of keypoints.

#### MP.8 Performance Evaluation 2
* Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. 
* In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.
* Solution Result: please check `MP_8_Counts_Matched_Keypoints.csv` file [CSV file](https://github.com/studian/SFND_P3_2D_Feature_Tracking/MP_8_Counts_Matched_Keypoints.csv).

#### MP.9 Performance Evaluation 3
* Log the time it takes for keypoint detection and descriptor extraction. 
* The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.
* Solution Result: please check `MP_9_Log_Time.csv` file [CSV file](https://github.com/studian/SFND_P3_2D_Feature_Tracking/MP_9_Log_Time.csv).

Considering `MP_8_Counts_Matched_Keypoints.csv` and `MP_9_Log_Time.csv` The TOP3 detector / descriptor combinations as the best choice for our purpose of detecting keypoints on vehicles are:

Rank  |  Detector/Descriptor  | The Average Number of Keypoints | Average Time
------|---------------------- | --------------------------------| --------
1st   |FAST/BRIEF             | 242 keypoints                   |  8.26 ms
2nd   |FAST/ORB               | 229 keypoints                   |  8.25 ms 
3rd   |FAST/SIFT              | 247 keypoints                   | 17.73 ms

---
* Solution code of `MP.7`, `MP.8`, and `MP.9`: Lines 48 ~ 100 at `MidTermProject_Camera_Student.cpp`
```c++
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
```
* Solution code of `MP.7`, `MP.8`, and `MP.9`: Lines 142 ~ 144 at `MidTermProject_Camera_Student.cpp`
```c++
// ...modified start: MP.7, MP.8, MP.9
string detectorType = detector_type_name; //"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"
// ...modified end: MP.7, MP.8, MP.9
```
* Solution code of `MP.7`: Lines 207 ~ 212 at `MidTermProject_Camera_Student.cpp`
```c++
//.. add start: MP.7 Performance Evaluation 1
if(!write_detector)
{
    detector_file  << ", " << keypoints.size();
}                
//.. add end: MP.7 Performance Evaluation 1
```
* Solution code of `MP.7`, `MP.8`, and `MP.9`: Lines 242 ~ 244 at `MidTermProject_Camera_Student.cpp`
```c++
// ...modified start: MP.7, MP.8, MP.9
string descriptorType = descriptor_type_name; // BRIEF, ORB, FREAK, AKAZE, SIFT
// ...modified end: MP.7, MP.8, MP.9
```
* Solution code of `MP.8` and `MP.9`: Lines 291 ~ 298 at `MidTermProject_Camera_Student.cpp`
```c++
//.. add start: MP.8 Performance Evaluation 2
det_des_matches << ", " << matches.size();
//.. add end: MP.8 Performance Evaluation 2

//.. add start: MP.9 Performance Evaluation 3
t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
det_des_time << ", " << 1000*t;
//.. add end: MP.9 Performance Evaluation 3
```
* Solution code of `MP.7`, `MP.8`, and `MP.9`: Lines 322 ~ 343 at `MidTermProject_Camera_Student.cpp`
```c++
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
```

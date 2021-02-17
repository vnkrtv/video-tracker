#include <chrono>

#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dir_nav.h>
#include <dlib/opencv/cv_image.h>

#include "db.hpp"

namespace detector {

    using namespace std::chrono;

    // Fill the vector with random colors
    void getRandomColors(vector<cv::Scalar> &colors, int numColors) {
        cv::RNG rng(0);
        for (int i = 0; i < numColors; i++) {
            colors.emplace_back(cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
        }
    }

    int videoDetection(const string &videoSrc, const string &modelPath, const set<int> &classesSet,
                       const float &confCoefficient) {
        MobileNetSSD net(cv::Size(300, 300));
        try {
            net.loadModel(modelPath);
        } catch (std::exception &e) {
            std::cerr << "Error on loading MobileNetSSD model: " << e.what() << std::endl;
            return -1;
        }
        cv::VideoCapture cap(videoSrc);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open the video file" << std::endl;
            return -1;
        }

        double dWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        double dHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        namedWindow("VideoDetect", cv::WINDOW_AUTOSIZE);

        std::clog << "Frame size : " << dWidth << " x " << dHeight << std::endl;

        cv::Mat frame, cvtFrame;

        int frameCounter = 0;
        double minTrackingQuality = 7.;
        map<int, dlib::correlation_tracker> objTrackers;  // currentObjID: tracker
        int currentObjID = 0;


        while (true) {
            auto startTime = system_clock::now();
            bool bSuccess = cap.read(frame);
            if (!bSuccess) {
                std::cerr << "Cannot read a frame from video file" << std::endl;
                break;
            }
            frameCounter++;
            dlib::cv_image<unsigned char> img(frame);

            vector<int> objIDsToDelete;
            for (auto &[objID, tracker]: objTrackers) {
                double trackingQuality = tracker.update(img);
                if (trackingQuality < minTrackingQuality) {
                    objIDsToDelete.emplace_back(objID);
                }
            }
            for (auto &objID: objIDsToDelete) {
                std::clog << "Remove objID: " << objID << " from lost of trackers" << std::endl;
                objTrackers.erase(objID);
            }

            if (!(frameCounter % 10)) {
                cvtColor(frame, cvtFrame, cv::COLOR_RGB2GRAY);
                dlib::cv_image<unsigned char> cvtImg(cvtFrame);

                auto detectedObjects = net.detectObjects(frame, classesSet, confCoefficient);
                for (auto &obj : detectedObjects) {
                    auto bbox = obj.bbox;
                    int x = bbox.x;
                    int y = bbox.y;
                    int width = bbox.width;
                    int height = bbox.height;

                    int xBar = x + static_cast<int>(0.5 * width);
                    int yBar = y + static_cast<int>(0.5 * height);

                    int matchObjID = -1;
                    for (auto &[objID, tracker]: objTrackers) {
                        auto trackedPosition = tracker.get_position();

                        int tx = static_cast<int>(trackedPosition.left());
                        int ty = static_cast<int>(trackedPosition.top());
                        int tWidth = static_cast<int>(trackedPosition.width());
                        int tHeight = static_cast<int>(trackedPosition.height());

                        int txBar = tx + static_cast<int>(0.5 * tWidth);
                        int tyBar = ty + static_cast<int>(0.5 * tHeight);

                        bool pred = (tx <= xBar <= (tx + tWidth)) &&
                                    (ty <= yBar <= (tx + tHeight)) &&
                                    (y <= tyBar <= (y + height));
                        if (pred) {
                            matchObjID = objID;
                        }
                    }
                    if (matchObjID == -1) {
                        std::clog << "Creating new tracker: objID(" << currentObjID << ")" << std::endl;
                        dlib::correlation_tracker tracker;
                        tracker.start_track(cvtImg, dlib::rectangle(x, y, x + width, y + height));
                        objTrackers[currentObjID++] = tracker;
                    }

                }
            }

            for (auto &[objID, tracker]: objTrackers) {
                auto trackedPosition = tracker.get_position();

                int tx = static_cast<int>(trackedPosition.left());
                int ty = static_cast<int>(trackedPosition.top());
                int tWidth = static_cast<int>(trackedPosition.width());
                int tHeight = static_cast<int>(trackedPosition.height());

                string label = "ObjectID(" + std::to_string(objID) + ")";
                auto bbox = cv::Rect2i(tx, ty, tx + tWidth, ty + tHeight);
                cv::rectangle(frame, bbox, (0, 0, 255), 1);
                cv::putText(frame, label, cv::Point2i(bbox.x, bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255));
            }

            auto endTime = system_clock::now();

            cv::imshow("VideoDetect", frame);

            if (cv::waitKey(30) == 27) {
                std::cout << "Esc key is pressed by user. Bye!" << std::endl;
                break;
            }

        }
        cv::destroyAllWindows();
        return 0;
    }

} // namespace detector

//        int maxDisappeared = 50;
//        CentroidTracker centroidTracker(maxDisappeared);
//        TrackerType trType = TR_MIL;

//            auto boxesVec = vector<cv::Rect2i>();
//            for (auto & obj : detectedObjects) {
//                boxesVec.emplace_back(obj.bbox);
//                cv::rectangle(frame, obj.bbox, (0, 0, 255), 1);
//                cv::putText(frame, obj.getLabel(), cv::Point2i(obj.bbox.x, obj.bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
//                            (255, 255, 255));
//            }
//            auto objects = centroidTracker.update(boxesVec);
//            for (auto & [objectId, centroid] : objects) {
//                string text = "ID " + std::to_string(objectId);
//                cv::putText(frame,
//                            text,
//                            cv::Point2i(centroid.x - 10, centroid.y - 10),
//                            cv::FONT_HERSHEY_SIMPLEX,
//                            0.5,
//                            (255, 255, 255));
//                cv::circle(frame,
//                           cv::Point2i(centroid.x, centroid.y),
//                           4,
//                           (255, 255, 255),
//                           -1);
//            }




//        vector<cv::Rect> bboxesVec;
//        cap >> frame;
//        bool showCrosshair = true;
//        bool fromCenter = false;
//
//        std::cout << "\n==========================================================\n";
//        std::cout << "OpenCV says press c to cancel objects selection process" << std::endl;
//        std::cout << "It doesn't work. Press Escape to exit selection process" << std::endl;
//        std::cout << "\n==========================================================\n";
//
//        cv::selectROIs("MultiTracker", frame, bboxesVec, showCrosshair, fromCenter);
//
//        if (bboxesVec.empty()) {
//            return 0;
//        }
//        vector<cv::Scalar> colors;
//        getRandomColors(colors, bboxesVec.size());
//
//        string trackerType = "CSRT";
//        Ptr<cv::MultiTracker> multiTracker = cv::MultiTracker::create();
//
//        for(int i=0; i < bboxes.size(); i++)
//            multiTracker->add(createTrackerByName(trackerType), frame, Rect2d(bboxes[i]));
//
//        TrackerType trType = TR_MIL;
//        auto trackers = vector<ObjectTracker>();
//        vector<dlib::correlation_tracker> trackers;



//            cvtColor(frame, cvtFrame, cv::COLOR_RGB2GRAY);
//            dlib::cv_image<unsigned char> img(cvtFrame);
//            dlib::correlation_tracker tracker;
//            tracker.start_track(img, dlib::centered_rect(dlib::point(120,100), 80, 150));
//
//            dlib::image_window win;
//            tracker.update(img);
//
//            win.set_image(img);
//            win.clear_overlay();
//            win.add_overlay(tracker.get_position());
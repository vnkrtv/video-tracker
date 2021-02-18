#include <chrono>

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
        MobileNetSSD net;
        try {
            net.loadModel(modelPath);
            std::clog << "Loaded MobileNetSSD model" << std::endl;
        } catch (std::exception &e) {
            std::cerr << "Error on loading MobileNetSSD model: " << e.what() << std::endl;
            return -1;
        }

        cv::VideoCapture cap(videoSrc);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open the video file" << std::endl;
            return -1;
        }
        std::clog << "Opened video source: " << videoSrc << std::endl;

        namedWindow("Video tracker", cv::WINDOW_AUTOSIZE);
        double dWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        double dHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        std::clog << "Frame size : " << dWidth << " x " << dHeight << std::endl;

        cv::Mat frame;
        int frameCounter = 0;
        double minTrackingQuality = 7.;
        MultiTracker multiTracker(minTrackingQuality);

        do {
            auto startTime = system_clock::now();
            bool bSuccess = cap.read(frame);
            if (!bSuccess) {
                std::cerr << "Cannot read a frame from video file" << std::endl;
                break;
            }
            dlib::cv_image<dlib::bgr_pixel> img(cvIplImage(frame));

            multiTracker.update(img);
            if (!(frameCounter % 10)) {
                auto detectedObjects = net.detectObjects(frame, classesSet, confCoefficient);
                multiTracker.addTrackers(img, detectedObjects);
            }

            for (auto &[objID, tracker]: multiTracker.getTrackers()) {
                auto bbox = multiTracker.getObjectBbox(tracker);
                cv::rectangle(frame, bbox, (0, 0, 255), 2);
                cv::putText(frame, "ID: " + std::to_string(objID), cv::Point2i(bbox.x, bbox.y - 18), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255));
                cv::putText(frame, multiTracker.getLabel(objID), cv::Point2i(bbox.x, bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255));
            }

            auto endTime = system_clock::now();

            frameCounter++;
            cv::imshow("Video tracker", frame);
        } while (cv::waitKey(30) != 27);
        std::clog << "Esc key is pressed by user. Bye!" << std::endl;
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
//        auto getTrackers = vector<MultiTracker>();
//        vector<dlib::correlation_tracker> getTrackers;



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
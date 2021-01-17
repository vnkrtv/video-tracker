#include "tracker.hpp"

#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dir_nav.h>
#include <dlib/opencv/cv_image.h>

namespace detector {

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
        auto cap = cv::VideoCapture(videoSrc);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open the video file" << std::endl;
            return -1;
        }

        double dWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        double dHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        namedWindow("VideoDetect", cv::WINDOW_AUTOSIZE);

        std::cout << "Frame size : " << dWidth << " x " << dHeight << std::endl;

        cv::Mat frame, cvtFrame;

        while (true) {
            bool bSuccess = cap.read(frame);
            if (!bSuccess) {
                std::cout << "Cannot read a frame from video file" << std::endl;
                break;
            }

            auto detectedObjects = net.detectObjects(frame, classesSet, confCoefficient);
            for (auto & obj : detectedObjects) {
                cv::rectangle(frame, obj.bbox, (0, 0, 255), 1);
                cv::putText(frame, obj.getLabel(), cv::Point2i(obj.bbox.x, obj.bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255));
            }

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
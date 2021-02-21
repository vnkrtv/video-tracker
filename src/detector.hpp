#include <chrono>

#include "db.hpp"

namespace detector {

    using namespace std::chrono;

    auto fontFace = cv::FONT_HERSHEY_SIMPLEX;
    auto color = cv::Scalar(0, 255, 255);
    auto fontScale = 0.5;

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
        SpeedDetector speedDetector;

        do {
            auto startTime = system_clock::now();
            bool bSuccess = cap.read(frame);
            if (!bSuccess) {
                std::cerr << "Cannot read a frame from video file" << std::endl;
                break;
            }
            dlib::cv_image<dlib::bgr_pixel> img(cvIplImage(frame));

            multiTracker.update(img, speedDetector);
            if (!(frameCounter % 10)) {
                auto detectedObjects = net.detectObjects(frame, classesSet, confCoefficient);
                multiTracker.addTrackers(img, detectedObjects);
            }

            auto endTime = system_clock::now();
            auto duration = duration_cast<milliseconds>(endTime - startTime).count();
            auto fps = 1000. / duration;
            cv::putText(frame, "FPS: " + std::to_string(fps), cv::Point2i(15, 15),
                        fontFace, fontScale, color);
            auto objSpeed = speedDetector.getObjectsSpeed(fps);

            for (auto &[objID, tracker]: multiTracker.getTrackers()) {
                auto bbox = multiTracker.getObjectBbox(tracker);
                auto speed = static_cast<int>(objSpeed[objID]);
                string label = std::to_string(speed) + " km/h";
                cv::rectangle(frame, bbox, color, 2);
                cv::putText(frame, label, cv::Point2i(bbox.x, bbox.y - 18),
                            fontFace, fontScale, color);
                cv::putText(frame, multiTracker.getLabel(objID), cv::Point2i(bbox.x, bbox.y - 5),
                            fontFace, fontScale, color);
            }

            frameCounter++;
            cv::imshow("Video tracker", frame);
        } while (cv::waitKey(30) != 27);
        std::clog << "Esc key is pressed by user. Bye!" << std::endl;
        cv::destroyAllWindows();
        return 0;
    }

}

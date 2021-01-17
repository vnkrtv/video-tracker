#include "model.hpp"

#include <dlib/image_processing/correlation_tracker.h>
#include <dlib/image_processing.h>

namespace detector {

    dlib::correlation_tracker()
    dlib::re

    int videoDetection(const string &videoSrc, const string &modelPath, const set<int> & classesSet, const float & confCoefficient) {
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

        cv::Mat frame;
        while (true) {
            bool bSuccess = cap.read(frame);
            if (!bSuccess) {
                std::cout << "Cannot read a frame from video file" << std::endl;
                break;
            }

            net.detect(frame, classesSet, confCoefficient);
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

#include "processor.hpp"

namespace detector {

    using namespace std::chrono;

    auto fontFace = cv::FONT_HERSHEY_SIMPLEX;
    auto color = cv::Scalar(0, 255, 255);
    auto fontScale = 0.5;
    double dlibMinTrackingQuality = 7.;


    void VideoProcessor::processFrame(cv::Mat &frame, int &frameCounter) {
        auto startTime = system_clock::now();
        bool bSuccess = _cap.read(frame);
        if (!bSuccess) {
            std::cerr << "Cannot read a frame from video file" << std::endl;
            return;
        }
        dlib::cv_image<dlib::bgr_pixel> img(cvIplImage(frame));

        _multiTracker.update(img);
        if (!(frameCounter % 10)) {
            auto detectedObjects = _net.detectObjects(frame, _classesSet, _confCoefficient);
            _multiTracker.addTrackers(img, detectedObjects);
        }

        auto endTime = system_clock::now();
        auto duration = duration_cast<milliseconds>(endTime - startTime).count();
        auto fps = 1000. / duration;
        cv::putText(frame, "FPS: " + std::to_string(fps), cv::Point2i(15, 15),
                    fontFace, fontScale, color);
        auto objSpeed = _multiTracker.getObjectsSpeed(fps);

        for (auto &[objID, tracker]: _multiTracker.getTrackers()) {
            auto bbox = MultiTracker::getObjectBbox(tracker);
            auto speed = static_cast<int>(objSpeed[objID]);
            string label = std::to_string(speed) + " km/h";
            cv::rectangle(frame, bbox, color, 2);
            cv::putText(frame, label, cv::Point2i(bbox.x, bbox.y - 18),
                        fontFace, fontScale, color);
            cv::putText(frame, _multiTracker.getLabel(objID), cv::Point2i(bbox.x, bbox.y - 5),
                        fontFace, fontScale, color);
        }

        frameCounter++;
    }

    void VideoProcessor::process() {
        cv::Mat frame;
        int frameCounter = 0;

        cv::namedWindow("Video tracker", cv::WINDOW_AUTOSIZE);
        do {
            processFrame(frame, frameCounter);
            cv::imshow("Video tracker", frame);
        } while (cv::waitKey(30) != 27);

        std::clog << "Esc key is pressed by user. Bye!" << std::endl;
        cv::destroyAllWindows();
    }

    void VideoProcessor::processToFile(const string &outFileName, const bool &displayNamedWindow) {
        int fps = 15;
        cv::VideoWriter writer(outFileName,
                               cv::VideoWriter::fourcc('D', 'I', 'V', '3'),
                               fps,
                               _frameSize,
                               true);
        cv::Mat frame;
        int frameCounter = 0;
        if (displayNamedWindow) {
            cv::namedWindow("Video tracker", cv::WINDOW_AUTOSIZE);
            do {
                processFrame(frame, frameCounter);
                writer.write(frame);
                cv::imshow("Video tracker", frame);
            } while (cv::waitKey(30) != 27);
            cv::destroyAllWindows();
        } else {
            do {
                processFrame(frame, frameCounter);
                writer.write(frame);
            } while (cv::waitKey(30) != 27);
        }
        std::clog << "Esc key is pressed by user. Bye!" << std::endl;
        writer.release();
    }

    VideoProcessor::VideoProcessor() : _multiTracker(MultiTracker(dlibMinTrackingQuality)) {}

    void VideoProcessor::loadModel(const string &modelPath, const set<int> &classesSet, const float &confCoefficient) {
        try {
            _net.loadModel(modelPath);
            std::clog << "Loaded MobileNetSSD model" << std::endl;
            _classesSet = classesSet;
            _confCoefficient = confCoefficient;
        } catch (std::exception &e) {
            std::cerr << "Error on loading MobileNetSSD model: " << e.what() << std::endl;
            exit(-1);
        }
    }

    void VideoProcessor::openVideoSrc(const string &videoSrc) {
        _cap.open(videoSrc);
        if (!_cap.isOpened()) {
            std::cerr << "Cannot open the video file" << std::endl;
            exit(-1);
        }
        std::clog << "Opened video source: " << videoSrc << std::endl;
        double _dWidth = _cap.get(cv::CAP_PROP_FRAME_WIDTH);
        double _dHeight = _cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        std::clog << "Frame size : " << _dWidth << " x " << _dHeight << std::endl;
        _frameSize = cv::Size2i(_dWidth, _dHeight);
    }

    void VideoProcessor::run(const string &outFileName, const bool &displayNamedWindow) {
        if (!displayNamedWindow && !outFileName.empty()) {
            processToFile(outFileName, !displayNamedWindow);
        } else if (!outFileName.empty()) {
            processToFile(outFileName, displayNamedWindow);
        } else {
            process();
        }
    }

} // namespace detector

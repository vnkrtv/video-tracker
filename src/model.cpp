#include "model.hpp"

#include <utility>

namespace detector {

    cv::Mat MobileNetSSD::forward(cv::Mat &frame) {
        cv::Mat rgb, resizedFrame;

        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        cv::resize(rgb, resizedFrame, _netInputSize);

        _cols = resizedFrame.cols;
        _rows = resizedFrame.rows;

        auto blob = cv::dnn::blobFromImage(resizedFrame, 1.0 / 255, _netInputSize, 127.5);
        _net.setInput(blob);
        return std::move(_net.forward());
    }

    MobileNetSSD::MobileNetSSD(cv::Size netInputSize): _netInputSize(std::move(netInputSize)) {}

    void MobileNetSSD::loadModel(const string &modelPath) {
        _net = cv::dnn::readNetFromCaffe(
                modelPath + "/MobileNetSSD_deploy.prototxt",
                modelPath + "/MobileNetSSD_deploy.caffemodel");
    }

    void MobileNetSSD::detect(
            cv::Mat &frame,
            const set<int> &classesSet,
            const float &confCoefficient) {
        auto out = forward(frame);
        for (int i = 0; i < out.size[2]; i++) {
            auto classVec = out.at<cv::Vec<float, 7>>(0, 0, i);
            auto classId = static_cast<int>(classVec[1]);
            auto confidence = classVec[2];
            if (confidence > confCoefficient && classesSet.find(classId) != classesSet.end()) {
                auto label = std::to_string(classId) + ": " + std::to_string(confidence);

                double heightFactor = frame.rows / 300.0;
                double widthFactor = frame.cols / 300.0;

                int xLeftBottom = static_cast<int>( static_cast<int>(classVec[3] * float(_cols)) * widthFactor );
                int yLeftBottom = static_cast<int>( static_cast<int>(classVec[4] * float(_rows)) * heightFactor );
                int xRightTop = static_cast<int>( static_cast<int>(classVec[5] * float(_cols)) * widthFactor );
                int yRightTop = static_cast<int>( static_cast<int>(classVec[6] * float(_rows)) * heightFactor );

                cv::rectangle(frame, cv::Point2i(xLeftBottom, yLeftBottom), cv::Point2i(xRightTop, yRightTop),
                              (0, 0, 255), 1);
                cv::putText(frame, label, cv::Point2i(xLeftBottom, yRightTop + 15), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255));
            }
        }
    }

}; // namespace detector

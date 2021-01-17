#include "model.hpp"

#include <utility>

namespace detector {

    unordered_map<int, string> class2name{
            {CL_BACKGROUND,   "Background"},
            {CL_AEROPLANE,    "Aeroplane"},
            {CL_BICYCLE,      "Bicycle"},
            {CL_BIRD,         "Bird"},
            {CL_BOAT,         "Boat"},
            {CL_BOTTLE,       "Bottle"},
            {CL_BUS,          "Bus"},
            {CL_CAR,          "Car"},
            {CL_CAT,          "Cat"},
            {CL_CHAIR,        "Chair"},
            {CL_COW,          "Cow"},
            {CL_DINING_TABLE, "Dining table"},
            {CL_DOG,          "Dog"},
            {CL_HORSE,        "Horse"},
            {CL_MOTORBIKE,    "Motorbike"},
            {CL_PERSON,       "Person"},
            {CL_POTTED_PLANT, "Potted plant"},
            {CL_SHEEP,        "Sheep"},
            {CL_SOFA,         "Sofa"},
            {CL_TRAIN,        "Train"},
            {CL_TV_MONITOR,   "TV Monitor"}
    };

    DetectionResult::DetectionResult(int _classId, int _confPercent, cv::Rect2i _bbox) :
            classId(_classId), confPercent(_confPercent), bbox(std::move(_bbox)) {}

    string DetectionResult::getLabel() const {
        return class2name[classId] + ": " + std::to_string(confPercent) + " %";
    }

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

    cv::Rect2i MobileNetSSD::getDetectedObjBox(const cv::Mat &frame, const cv::Vec<float, 7> &classVec) const {
        double heightFactor = frame.rows / 300.0;
        double widthFactor = frame.cols / 300.0;

        int xLeftBottom = static_cast<int>( static_cast<int>(classVec[3] * float(_cols)) * widthFactor );
        int yLeftBottom = static_cast<int>( static_cast<int>(classVec[4] * float(_rows)) * heightFactor );
        int xRightTop = static_cast<int>( static_cast<int>(classVec[5] * float(_cols)) * widthFactor );
        int yRightTop = static_cast<int>( static_cast<int>(classVec[6] * float(_rows)) * heightFactor );

        return cv::Rect2i(cv::Point2i(xLeftBottom, yLeftBottom), cv::Point2i(xRightTop, yRightTop));
    }

    MobileNetSSD::MobileNetSSD(cv::Size netInputSize) : _netInputSize(std::move(netInputSize)) {}

    void MobileNetSSD::loadModel(const string &modelPath) {
        _net = cv::dnn::readNetFromCaffe(
                modelPath + "/MobileNetSSD_deploy.prototxt",
                modelPath + "/MobileNetSSD_deploy.caffemodel");
    }

    vector<DetectionResult> MobileNetSSD::detectObjects(
            cv::Mat &frame,
            const set<int> &classesSet,
            const float &confCoefficient) {
        vector<DetectionResult> detectedObjects;
        auto out = forward(frame);
        for (int i = 0; i < out.size[2]; i++) {
            auto classVec = out.at<cv::Vec<float, 7>>(0, 0, i);
            auto classId = static_cast<int>(classVec[1]);
            auto confidence = classVec[2];
            if (confidence > confCoefficient && classesSet.find(classId) != classesSet.end()) {
                int confPercent = int(100 * confidence);
                auto bbox = getDetectedObjBox(frame, classVec);
                detectedObjects.emplace_back(DetectionResult(classId, confPercent, bbox));
            }
        }
        return detectedObjects;
    }

}; // namespace detector

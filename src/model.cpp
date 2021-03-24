#include "model.hpp"

#include <utility>

namespace detector {

    unordered_map<ObjectClass, string> class2name{
            {ObjectClass::BACKGROUND,   "Background"},
            {ObjectClass::AEROPLANE,    "Aeroplane"},
            {ObjectClass::BICYCLE,      "Bicycle"},
            {ObjectClass::BIRD,         "Bird"},
            {ObjectClass::BOAT,         "Boat"},
            {ObjectClass::BOTTLE,       "Bottle"},
            {ObjectClass::BUS,          "Bus"},
            {ObjectClass::CAR,          "Car"},
            {ObjectClass::CAT,          "Cat"},
            {ObjectClass::CHAIR,        "Chair"},
            {ObjectClass::COW,          "Cow"},
            {ObjectClass::DINING_TABLE, "Dining table"},
            {ObjectClass::DOG,          "Dog"},
            {ObjectClass::HORSE,        "Horse"},
            {ObjectClass::MOTORBIKE,    "Motorbike"},
            {ObjectClass::PERSON,       "Person"},
            {ObjectClass::POTTED_PLANT, "Potted plant"},
            {ObjectClass::SHEEP,        "Sheep"},
            {ObjectClass::SOFA,         "Sofa"},
            {ObjectClass::TRAIN,        "Train"},
            {ObjectClass::TV_MONITOR,   "TV Monitor"}
    };

    DetectionResult::DetectionResult(int _classId, int _confPercent, cv::Rect2i _bbox) :
            classId(_classId), confPercent(_confPercent), bbox(std::move(_bbox)) {}

    string DetectionResult::getLabel() const {
        return class2name[static_cast<ObjectClass>(classId)] + ": " + std::to_string(confPercent) + "%";
    }

    cv::Mat MobileNetSSD::forward(cv::Mat &frame) {
        _cols = frame.cols;
        _rows = frame.rows;

        auto blob = cv::dnn::blobFromImage(frame, 1.0 / 255, cv::Size_<int>(_cols, _rows), 127.5);
        _net.setInput(blob);
        return std::move(_net.forward());
    }

    cv::Rect2i MobileNetSSD::getDetectedObjBox(const cv::Mat &frame, const cv::Vec<float, 7> &classVec) const {
        int xLeftBottom = static_cast<int>( static_cast<int>(classVec[3] * float(_cols)) );
        int yLeftBottom = static_cast<int>( static_cast<int>(classVec[4] * float(_rows)) );
        int xRightTop = static_cast<int>( static_cast<int>(classVec[5] * float(_cols)) );
        int yRightTop = static_cast<int>( static_cast<int>(classVec[6] * float(_rows)) );

        return cv::Rect2i(cv::Point2i(xLeftBottom, yLeftBottom), cv::Point2i(xRightTop, yRightTop));
    }

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

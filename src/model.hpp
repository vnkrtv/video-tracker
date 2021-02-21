#include <iostream>
#include <utility>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <set>

#include <opencv2/opencv.hpp>

namespace detector {

    using std::pair;
    using std::string;
    using std::vector;
    using std::unordered_map;
    using std::set;

    enum ModelClass {
        CL_BACKGROUND = 0,
        CL_AEROPLANE,
        CL_BICYCLE,
        CL_BIRD,
        CL_BOAT,
        CL_BOTTLE,
        CL_BUS,
        CL_CAR,
        CL_CAT,
        CL_CHAIR,
        CL_COW,
        CL_DINING_TABLE,
        CL_DOG,
        CL_HORSE,
        CL_MOTORBIKE,
        CL_PERSON,
        CL_POTTED_PLANT,
        CL_SHEEP,
        CL_SOFA,
        CL_TRAIN,
        CL_TV_MONITOR
    };

    struct DetectionResult {
        int classId;
        int confPercent;
        cv::Rect2i bbox;

        DetectionResult(int _classId, int _confPercent, cv::Rect2i _bbox);

        [[nodiscard]] string getLabel() const;
    };

    class MobileNetSSD {
    private:

        cv::dnn::Net _net;

        int _cols{};
        int _rows{};

        cv::Mat forward(cv::Mat &frame);

        [[nodiscard]] cv::Rect2i getDetectedObjBox(const cv::Mat &frame, const cv::Vec<float, 7> &classVec) const;

    public:

        void loadModel(const string &modelPath);

        vector<DetectionResult> detectObjects(cv::Mat &frame, const set<int> &classesSet, const float &confCoefficient);

    };

}; // namespace detector

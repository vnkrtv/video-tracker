#include <iostream>
#include <utility>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <set>

#include <opencv2/opencv.hpp>


namespace detector {

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

    class MobileNetSSD {
    private:

        cv::dnn::Net _net;
        cv::Size _netInputSize;

        int _cols{};
        int _rows{};

        cv::Mat forward(cv::Mat &);

    public:

        explicit MobileNetSSD(cv::Size);

        void loadModel(const string &);

        void detect(cv::Mat &, const set<int> &, const float &);

    };

}; // namespace detector

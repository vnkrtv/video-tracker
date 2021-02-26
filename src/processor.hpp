#include <chrono>

#include "db.hpp"

namespace detector {

    using namespace std::chrono;

    class VideoProcessor {
    private:

        cv::VideoCapture _cap;
        cv::Size2i _frameSize;

        MobileNetSSD _net;
        set<int> _classesSet;
        float _confCoefficient{};

        MultiTracker _multiTracker;

        void processFrame(cv::Mat &frame, int &frameCounter);

        void process();

        void processToFile(const string &outFileName, const bool &displayNamedWindow);

    public:

        explicit VideoProcessor();

        void loadModel(const string &modelPath, const set<int> &classesSet, const float &confCoefficient);

        void openVideoSrc(const string &videoSrc);
        
        void run(const string &outFileName, const bool &displayNamedWindow);
        
    };
    
} // namespace detector

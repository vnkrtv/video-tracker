#include "args.hpp"
#include "detector.hpp"

using namespace std::chrono;

namespace detector {

    struct Args {
        string _videoSrc;
        string _modelPath;
        set<int> _classesSet{};
        float _confCoefficient = 0.4;
        bool _useGpu = false;

        Args() = default;

        static const char *help() {
            return "Video detection";
        }

        template<class F>
        void parse(F f) {
            f(_videoSrc, "--video-src", "-v",
              args::help("Video source (video file, ip camera, video device)"), args::required());
            f(_modelPath, "--model-path", "-m",
              args::help("MobileNetSSD folder path"), args::required());
            f(_classesSet, "--classes", "-c",
              args::help("Set of detected classes ID. Full set could be found in README. Default classes: persons and cars"));
            f(_confCoefficient, "--confidence", "-t",
              args::help("Model's confidence coefficient. Default value: 0.4"));
            f(_useGpu, "--cuda",
              args::help("Use GPU with CUDA"), args::set(true));
        }

        void run() {
            if (1 <= _confCoefficient || _confCoefficient <= 0) {
                std::cerr << "Incorrect value for model's confidence coefficient. Must be in range(0,1)" << std::endl;
                return;
            }
            if (_classesSet.empty()) {
                _classesSet = set<int>{CL_PERSON, CL_CAR};
            }
            if (_useGpu) {
                cv::cuda::setDevice(cv::cuda::getDevice());
            }
            std::cout << "Video source: " << _videoSrc << std::endl;
            std::cout << "MobileNetSSD folder path: " << _modelPath << std::endl;
            std::cout << "Model's confidence coefficient: " << _confCoefficient << std::endl;
            std::cout << "Use GPU (CUDA): " << _useGpu << std::endl;
            detector::videoDetection(_videoSrc, _modelPath, _classesSet, _confCoefficient);
        }
    };

} // namespace detector

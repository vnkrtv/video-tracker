#include "args.hpp"
#include "processor.hpp"

using namespace std::chrono;

namespace detector {

    struct Args {
        string _videoSrc;
        string _modelPath = "model/MobileNetSSD";
        string _outputFileName;
        set<int> _classesSet{};
        float _confCoefficient = 0.4;
        bool _useGpu = false;
        bool _noNamedWindow = false;

        Args() = default;

        static const char *help() {
            return "Application for tracking various objects and estimating their speed in video stream";
        }

        template<class F>
        void parse(F f) {
            f(_videoSrc, "--video-src", "-v",
              args::help("Video source (video file, ip camera, video device)"), args::required());
            f(_modelPath, "--model-path", "-m",
              args::help("MobileNetSSD folder path"));
            f(_outputFileName, "--output", "-o",
              args::help("Output file name. By default, processed video stream is not saving"));
            f(_classesSet, "--classes", "-c",
              args::help(
                      "Set of detected classes ID. Full set could be found in README. Default classes: persons and cars"));
            f(_confCoefficient, "--confidence", "-t",
              args::help("Model's confidence coefficient. Default value: 0.4"));
            f(_noNamedWindow, "--no-window",
              args::help("Does not show named window with video stream. False by default"), args::set(true));
            f(_useGpu, "--cuda",
              args::help("Use GPU with CUDA"), args::set(true));
        }

        void run() {
            if (1 <= _confCoefficient || _confCoefficient <= 0) {
                std::cerr << "Incorrect value for model's confidence coefficient. Must be in range(0,1)" << std::endl;
                return;
            }
            if (_classesSet.empty()) {
                _classesSet = set<int>{
                        static_cast<int>(ObjectClass::PERSON),
                        static_cast<int>(ObjectClass::CAR)
                };
            }
            if (_useGpu) {
                cv::cuda::setDevice(cv::cuda::getDevice());
            }
            std::cout << "Video source: " << _videoSrc << std::endl;
            std::cout << "Output file: " << (_outputFileName.empty() ? "no" : _outputFileName) << std::endl;
            std::cout << "MobileNetSSD folder path: " << _modelPath << std::endl;
            std::cout << "Model's confidence coefficient: " << _confCoefficient << std::endl;
            std::cout << "Show named window with video stream: " << !_noNamedWindow << std::endl;
            std::cout << "Use GPU (CUDA): " << _useGpu << std::endl;

            VideoProcessor processor;
            processor.loadModel(_modelPath, _classesSet, _confCoefficient);
            processor.openVideoSrc(_videoSrc);
            processor.run(_outputFileName, !_noNamedWindow);

            exit(0);
        }
    };

} // namespace detector

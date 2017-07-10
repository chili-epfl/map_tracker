
#include "VideoSource.hpp"
#include "Map_Tracker.hpp"

static const char window_name[] = "Tracker";

namespace mt = map_tracker;


//work offline on recorded sequence
int main(int argc, char** argv)
{
    mt::Map_Tracker tracker("../data/");

    VideoSourceFile videoSource = VideoSourceFile(argv[1]);
    //VideoSourceLive videoSource = VideoSourceLive(1);
    videoSource.resizeSource(0.5);

    cv::namedWindow( window_name, cv::WINDOW_AUTOSIZE );

    while(1)
    {
        videoSource.grabNewFrame();
        cv::Mat inputImage = videoSource.getFramePointer();


        cv::Mat inputGray;
        cv::cvtColor(inputImage, inputGray, CV_RGB2GRAY);

        tracker.update(inputGray);
        tracker.drawLastDetection(&inputImage);

        imshow(window_name, inputImage);

        auto key = cv::waitKey(5);
        if(key == 27 || key == 'q' || videoSource.isOver())
            break;
    }

    return 0;
}

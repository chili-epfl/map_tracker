
#include "VideoSource.hpp"
#include "Map_Tracker.hpp"

static const char window_name[] = "Tracker";

namespace mt = map_tracker;


///@brief Shows usage info for this tool
void showHelp(){
    printf("Usage: tracker [OPTIONS]\n");
    printf("    -f FILE     Name of the input video file \n");
    printf("    -cg FILE    Name of the Config.xml file containing calibration and landmarks information\n");
    printf("    -v NUM      Video Webcam input (default 0)\n");
    printf("    -h,--help   Shows this help\n");
}


int tracker(const char* inPath = NULL, const char* configPath = NULL, unsigned int videoS = 0){


    mt::Map_Tracker tracker(configPath);

    if(inPath){
          VideoSourceFile videoSource = VideoSourceFile(inPath);
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
        else {
          VideoSourceLive videoSource = VideoSourceLive(videoS);
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
      }



//work offline on recorded sequence
int main(int argc, char** argv)
{
  unsigned int videoS = 0;

  char* input = NULL;
  char* configF = NULL;

    for(int i=1;i<argc;i++){
        if(strcmp(argv[i], "-f") == 0) { //input
            input = argv[i + 1];
            printf("\nInput: %s", input);
        }
        else if(strcmp(argv[i], "-cg") == 0) { //output
            configF = argv[i + 1];
            printf("\nConfig File: %s", configF);
        }
        else if(strcmp(argv[i], "-v") == 0) {
            videoS = atoi(argv[i + 1]);
            printf("\nVideo Source Live", videoS);
        }
        else if(strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            showHelp();
            return 0;
        }
        std::cout << std::endl;
    }


    tracker(input, configF, videoS);


}

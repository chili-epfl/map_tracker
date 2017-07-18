

#include <opencv2/line_descriptor.hpp>

#include "opencv2/core/utility.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace cv::line_descriptor;
using namespace std;

static const char* keys =
{ "{@image_path | | Image path }" };

static void help()
{
  cout << "\nThis example shows the functionalities of lines extraction " << "furnished by BinaryDescriptor class\n"
           << "Please, run this sample using a command in the form\n" << "./example_line_descriptor_lines_extraction <path_to_input_image>" << endl;
}


std::vector<KeyLine> filterLongestLines(std::vector<KeyLine> keylines, float percent){
    std::vector<KeyLine> keylinesL;
    float maxlength = 0.0;
    for ( int i = 0; i < (int) keylines.size(); i++ )
    {
      if( keylines[i].lineLength >= maxlength )
      {
        maxlength =  keylines[i].lineLength;
      }
    }

    for ( int i = 0; i < (int) keylines.size(); i++ )
    {
      if( keylines[i].lineLength >= maxlength*percent )
      {
        keylinesL.push_back(keylines[i]);
      }
    }
return keylinesL;
}

cv::Mat blueFilter(cv::Mat originalImage){
  // Convert input image to HSV
  cv::Mat hsv_image;
  cvtColor(originalImage, hsv_image, cv::COLOR_BGR2HSV);

 // Threshold the HSV image, keep only the blue pixels
 cv::Mat blue_hue_range;
 cv::inRange(hsv_image, cv::Scalar(60, 40, 50), cv::Scalar(125, 255, 255), blue_hue_range);

 return blue_hue_range;

}


cv::Point findIntersection(vector<KeyLine> lines){
  

}

int main( int argc, char** argv )
{
  /* get parameters from comand line */
  cv::CommandLineParser parser( argc, argv, keys );
  String image_path = parser.get<String>( 0 );

  if( image_path.empty() )
  {
        help();
        return -1;
  }

  /* load image */
  cv::Mat imageMat = imread( image_path, 1 );
  if( imageMat.data == NULL )
  {
        std::cout << "Error, image could not be loaded. Please, check its path" << std::endl;
  }

    cv::Mat imageMat_blue = blueFilter(imageMat);
    GaussianBlur( imageMat_blue, imageMat_blue, Size( 9, 9 ), 0, 0 );

    imshow("color filtered", imageMat_blue);

  /* create a ramdom binary mask */
  cv::Mat mask = Mat::ones( imageMat_blue.size(), CV_8UC1 );

  /* create a pointer to a BinaryDescriptor object with deafult parameters */
  Ptr<LSDDetector> bd = LSDDetector::createLSDDetector();
  /* create a structure to store extracted lines */
  vector<KeyLine> lines;

  /* extract lines */
  bd->detect( imageMat_blue, lines,2,1, mask );

  // filter longest lines
  lines = filterLongestLines(lines,0.10);






  /* draw lines extracted from octave 0 */
  cv::Mat output = imageMat.clone();
  if( output.channels() == 1 )
        cv::cvtColor( output, output, COLOR_GRAY2BGR );
  for ( size_t i = 0; i < lines.size(); i++ )
  {
        KeyLine kl = lines[i];
        if( kl.octave == 0)
        {
          /* get a random color */
          int R = ( rand() % (int) ( 255 + 1 ) );
          int G = ( rand() % (int) ( 255 + 1 ) );
          int B = ( rand() % (int) ( 255 + 1 ) );

          /* get extremes of line */
          Point pt1 = Point( kl.startPointX, kl.startPointY );
          Point pt2 = Point( kl.endPointX, kl.endPointY );

          /* draw line */
          line( output, pt1, pt2, Scalar( B, G, R ), 5 );
        }

  }

  /* show lines on image */
  cv::imshow( "Lines", output );

  // wrap perspective matrix   cvWarpPerspective(imageMat, dst, const CvMat* map_matrix, int flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, CvScalar fillval=cvScalarAll(0) )
  //cv::Mat dst;
  //cvWarpPerspective(imageMat, dst, cmap_matrix);


  cv::waitKey();
}

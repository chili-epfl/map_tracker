

#include <opencv2/line_descriptor.hpp>

#include "opencv2/core/utility.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>

#include <iostream>
#include <map>

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

float getAnglesDegree(KeyLine kl){
        return fmod(((kl.angle * 180 / M_PI) + 180), 180);
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

int nb_groups = 0;

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

float my_mean(float new_array[], int num){
        //GET TOTAL & CALCULATE MEAN
        float total = 0.0;
        for(int i=0; i<num; i++) {
                total += new_array[i];
        }
        return total/num;
}
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
float median(float new_array[], int num){
        //CALCULATE THE MEDIAN (middle number)
        if(num % 2 != 0) { // is the # of elements odd?
                int temp = ((num+1)/2)-1;

                return new_array[temp];
        }
        else{ // then it's even! :)

                return new_array[(num/2)-1];
        }
}

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
float mode(float new_array[], int num) {
        int* ipRepetition = new int[num];
        // alocate a new array in memory of the same size (round about way of defining number of elements by a variable)
        for (int i = 0; i < num; i++) {
                ipRepetition[i] = 0; //initialize each element to 0
                int j = 0; //
                while ((j < i) && (new_array[i] != new_array[j])) {
                        if (new_array[i] != new_array[j]) {
                                j++;
                        }
                }
                (ipRepetition[j])++;
        }
        int iMaxRepeat = 0;
        for (int i = 1; i < num; i++) {
                if (ipRepetition[i] > ipRepetition[iMaxRepeat]) {
                        iMaxRepeat = i;
                }
        }
        return new_array[iMaxRepeat];
}


std::map<int,int>  clusterLinesByAngles(std::vector<KeyLine> keylines, int nb_cluster){
  std::map<int,int> grouped_keylines;
  float angle =  getAnglesDegree(keylines[0]);
  float angleList[nb_cluster];
  for (int a = 0; a < nb_cluster; a++)
          angleList[a] = -1;
  angleList[0] = angle;
  int count = 0;
  float klangles[keylines.size()];
  for ( int i = 0; i < (int) keylines.size(); i++ )
          klangles[i] = getAnglesDegree(keylines[i]);



        for ( int i = 0; i < (int) keylines.size(); i++ ) {
                float max_distance = my_mean(klangles,(int) keylines.size() )/2;
                bool added = false;
                for (int a = 0; a <= count; a++) {
                        float angle = angleList[a];
                        if(angle!=-1) {
                                float distance = std::min(abs((angle + 5) - getAnglesDegree(keylines[i])), abs((angle - 5) - getAnglesDegree(keylines[i])));
                                std::cout << " line " << i  << " at angle " << getAnglesDegree(keylines[i]) << " group " << a << " at angle " << angle <<  " distance " << distance << " min " << max_distance << std::endl;
                                if(distance <= max_distance) {
                                        grouped_keylines[i] = a;
                                        distance = min(distance,max_distance);
                                        added = true;
                                }
                        }
                }
                if(added) {
                        std::cout << " ADDING TO EXISTING GROUP  "<<   grouped_keylines[i] << std::endl;

                }
                if(!added) {
                        count +=1;
                        angleList[count]=(getAnglesDegree(keylines[i]));
                        grouped_keylines[i] = count;
                        std::cout << " NEW GROUP  "<< count << std::endl;

                }
        }
        nb_groups = count;
        return grouped_keylines;
}



std::map<int,int>  groupLinesByAngles(std::vector<KeyLine> keylines){
        std::map<int,int> grouped_keylines;
        float angle =  getAnglesDegree(keylines[0]);
        std::vector<float> angleList;
        angleList.push_back(angle);
        for ( int i = 0; i < (int) keylines.size(); i++ ) {
                float min_distance = 80.0;
                float max_distance = 170.0;
                bool added = false;
                for (int a = 0; a < (int) angleList.size(); a++) {
                        float angle = angleList[a];
                        float distance = std::min(abs((angle + 5) - getAnglesDegree(keylines[i])), abs((angle - 5) - getAnglesDegree(keylines[i])));
                        std::cout << " line " << i  << " at angle " << getAnglesDegree(keylines[i]) << " group " << a << " at angle " << angle <<  " distance " << distance << " min " << min_distance << std::endl;

                        if(distance <= min_distance) {
                                min_distance = distance;
                                grouped_keylines[i] = a;
                                std::cout << " ADDING TO EXISTING GROUP  "<< a << std::endl;
                                added = true;
                                break;
                        }
                        if(distance >= max_distance) {
                                max_distance = distance;
                                grouped_keylines[i] = a;
                                std::cout << " ADDING TO EXISTING GROUP  "<< a << std::endl;
                                added = true;
                                break;
                        }
                }
                if(!added) {
                        angleList.push_back(getAnglesDegree(keylines[i]));
                        grouped_keylines[i] = angleList.size()-1;
                        std::cout << " NEW GROUP  "<< angleList.size()-1 << std::endl;

                }
        }
        nb_groups = (int) angleList.size();
        return grouped_keylines;
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


std::vector<Scalar> generateColor(int nb_groups){
        std::vector<Scalar> my_colors;
        for ( int i = 0; i < nb_groups; i++ )
        {
                /* get a random color */
                int R = ( rand() % (int) ( 255 + 1 ) );
                int G = ( rand() % (int) ( 255 + 1 ) );
                int B = ( rand() % (int) ( 255 + 1 ) );

                my_colors.push_back(Scalar( B, G, R ));
        }
        return my_colors;
}



void findTheHomography(){
  // Read source image.

      // Four corners of the book in source image
      vector<Point2f> pts_src;
      pts_src.push_back(Point2f(141, 131));
      pts_src.push_back(Point2f(480, 159));
      pts_src.push_back(Point2f(493, 630));
      pts_src.push_back(Point2f(64, 601));


      // Read destination image.
      Mat im_dst = imread("book1.jpg");
      // Four corners of the book in destination image.
      vector<Point2f> pts_dst;
      pts_dst.push_back(Point2f(318, 256));
      pts_dst.push_back(Point2f(534, 372));
      pts_dst.push_back(Point2f(316, 670));
      pts_dst.push_back(Point2f(73, 473));

      // Calculate Homography
      Mat h = findHomography(pts_src, pts_dst);

      // Output image
      Mat im_out;
      // Warp source image to destination based on homography
      warpPerspective(im_src, im_out, h, im_dst.size());

      // Display images
      imshow("Source Image", im_src);
      imshow("Destination Image", im_dst);
      imshow("Warped Source Image", im_out);

}


int main( int argc, char** argv )
{
        /* get parameters from comand line */
        cv::CommandLineParser parser( argc, argv, keys );
        String srcimg_path = parser.get<String>( 0 );
        String destimg_path  = parser.get<String>( 1 );

        if( srcimg_path.empty() || destimg_path.empty() ){
                help();
                return -1;
        }

        /* load image */
        cv::Mat imageMat = imread( srcimg_path, 1 );
        cv::Mat dest_imageMat = imread( destimg_path, 1 );
        
        if( imageMat.data == NULL ||  dest_imageMat.data == NULL)
        {
                std::cout << "Error, image could not be loaded. Please, check its path" << std::endl;
        }

        cv::Mat imageMat_blue = blueFilter(imageMat);
        //imwrite("./color_filtered.jpg", imageMat_blue);
        GaussianBlur( imageMat_blue, imageMat_blue, Size( 9, 9 ), 0, 0 );

        //imshow("color filtered", imageMat_blue);


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

        // group lines by angle
        //std::map<int,int> grouped_keylines =  groupLinesByAngles(lines);
        std::map<int,int> grouped_keylines = clusterLinesByAngles(lines, 2);

        std::cout << "NB of groups found " <<   grouped_keylines.size() << std::endl;



        std::vector<Scalar> myColors = generateColor(nb_groups);

        /* draw lines extracted from octave 0 */
        cv::Mat output = imageMat.clone();
        if( output.channels() == 1 )
                cv::cvtColor( output, output, COLOR_GRAY2BGR );
        for (auto const& it : grouped_keylines) {

                KeyLine kl = lines[it.first];
                std::cout << "line " << it.first << " in group " << it.second << " at angle " << getAnglesDegree(kl)  << " with length " << kl.lineLength << std::endl;

                if( kl.octave == 0)
                {

                        /* get extremes of line */
                        Point pt1 = Point( kl.startPointX, kl.startPointY );
                        Point pt2 = Point( kl.endPointX, kl.endPointY );

                        /* draw line */
                        line( output, pt1, pt2, myColors[it.second], 5 );
                }

        }

        /* show lines on image */
        cv::imshow( "Lines", output );
        cv::waitKey();
}

#include <opencv2/line_descriptor.hpp>

 #include "opencv2/core/utility.hpp"
 #include <opencv2/imgproc.hpp>
 #include <opencv2/features2d.hpp>
 #include <opencv2/highgui.hpp>

 #include <iostream>
 #include <vector>

 using namespace cv;
 using namespace cv::line_descriptor;

 static const std::string images[] =
 { "p1.png", "p2.png" ,"pacman.png" };

 static const char* keys =
 { "{@image_path | | Image path }" };

 static void help()
 {
   std::cout << "\nThis example shows the functionalities of radius matching " << "Please, run this sample using a command in the form\n"
       << "./example_line_descriptor_radius_matching <path_to_input_images>/" << std::endl;
 }

 int main( int argc, char** argv )
 {
   /* get parameters from comand line */
   CommandLineParser parser( argc, argv, keys );
   String pathToImages = parser.get < String > ( 0 );

   /* create structures for hosting KeyLines and descriptors */
   int num_elements = sizeof ( images ) / sizeof ( images[0] );
   std::vector < Mat > descriptorsMat;
   std::vector < std::vector<KeyLine> > linesMat;

   /*create a pointer to a BinaryDescriptor object */
   Ptr < BinaryDescriptor > bd = BinaryDescriptor::createBinaryDescriptor();

   /* compute lines and descriptors */
   for ( int i = 0; i < num_elements; i++ )
   {
     /* get path to image */
     std::stringstream image_path;
     image_path << pathToImages << images[i];
     std::cout << image_path.str().c_str() << std::endl;

     /* load image */
     Mat loadedImage = imread( image_path.str().c_str(), 1 );
     if( loadedImage.data == NULL )
     {
       std::cout << "Could not load images." << std::endl;
       help();
       exit( -1 );
     }

     /* compute lines and descriptors */
     std::vector < KeyLine > lines;
     Mat computedDescr;
     bd->detect( loadedImage, lines );
     bd->compute( loadedImage, lines, computedDescr );

     descriptorsMat.push_back( computedDescr );
     linesMat.push_back( lines );

   }

   /* compose a queries matrix */
   Mat queries;
   for ( size_t j = 0; j < descriptorsMat.size(); j++ )
   {
     if( descriptorsMat[j].rows >= 5 )
       queries.push_back( descriptorsMat[j].rowRange( 0, 5 ) );

     else if( descriptorsMat[j].rows > 0 && descriptorsMat[j].rows < 5 )
       queries.push_back( descriptorsMat[j] );
   }

   std::cout << "It has been generated a matrix of " << queries.rows << " descriptors" << std::endl;

   /* create a BinaryDescriptorMatcher object */
   Ptr < BinaryDescriptorMatcher > bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

   /* populate matcher */
   bdm->add( descriptorsMat );

   /* compute matches */
   std::vector < std::vector<DMatch> > matches;
   bdm->radiusMatch( queries, matches, 30 );
   std::cout << "size matches sample " << matches.size() << std::endl;

   for ( int i = 0; i < (int) matches.size(); i++ )
   {
     for ( int j = 0; j < (int) matches[i].size(); j++ )
     {
       std::cout << "match: " << matches[i][j].queryIdx << " " << matches[i][j].trainIdx << " " << matches[i][j].distance << std::endl;
     }

   }

   /* plot matches */
   cv::Mat lsd_outImg;
   resize( imageMat1, imageMat1, Size( imageMat1.cols / 2, imageMat1.rows / 2 ) );
   resize( imageMat2, imageMat2, Size( imageMat2.cols / 2, imageMat2.rows / 2 ) );
   std::vector<char> lsd_mask( matches.size(), 1 );
   drawLineMatches( imageMat1, octave0_1, imageMat2, octave0_2, good_matches, lsd_outImg, Scalar::all( -1 ), Scalar::all( -1 ), lsd_mask,
                    DrawLinesMatchesFlags::DEFAULT );

   imshow( "LSD matches", lsd_outImg );
   waitKey();

 }

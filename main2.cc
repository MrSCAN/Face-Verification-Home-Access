#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <thread>

using namespace dlib;
using namespace std;
using namespace cv;

// Face recognition model definition (same as in the original code)
template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET>
using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET>
using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
                                                  alevel0<
                                                      alevel1<
                                                          alevel2<
                                                              alevel3<
                                                                  alevel4<
                                                                      max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2, input_rgb_image_sized<150>>>>>>>>>>>>>;

int main(int argc, char **argv)
try
{
    if (argc != 2)
    {
        cout << "Usage: ./face_match <image_file>" << endl;
        return 1;
    }

    // Load models
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    deserialize("models/shape_predictor_5_face_landmarks.dat") >> sp;
    anet_type net;
    deserialize("models/dlib_face_recognition_resnet_model_v1.dat") >> net;

    // Load reference image
    matrix<rgb_pixel> img;
    load_image(img, argv[1]);
    // image_window win(img);

    // Extract face descriptor for reference image
    std::vector<matrix<rgb_pixel>> ref_faces;
    for (auto face : detector(img))
    {
        auto shape = sp(img, face);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
        ref_faces.push_back(move(face_chip));
    }

    if (ref_faces.empty())
    {
        cout << "No faces found in reference image!" << endl;
        return 1;
    }
    matrix<float, 0, 1> ref_descriptor = net(ref_faces[0]);

    // Start video capture loop
    // VideoCapture cap(0);
    cv::VideoCapture cap("udp://@:2020", cv::CAP_FFMPEG);
    if (!cap.isOpened())
    {
        cerr << "Error: Could not open the camera." << endl;
        return -1;
    }

    while (true)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            cerr << "Error: Blank frame grabbed." << endl;
            continue;
        }

        // Convert the captured frame to Dlib image
        matrix<rgb_pixel> captured_image;
        assign_image(captured_image, cv_image<bgr_pixel>(frame));

        // Detect faces in the captured image
        std::vector<matrix<rgb_pixel>> cap_faces;
        for (auto face : detector(captured_image))
        {
            auto shape = sp(captured_image, face);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(captured_image, get_face_chip_details(shape, 150, 0.25), face_chip);
            cap_faces.push_back(move(face_chip));
        }

        if (!cap_faces.empty())
        {
            matrix<float, 0, 1> cap_descriptor = net(cap_faces[0]);

            // Compare face descriptors
            string match_result = "Not a match";
            if (length(ref_descriptor - cap_descriptor) < 0.6)
            {
                match_result = "Face match";
            }
            // Display the result on the frame
            putText(frame, match_result, Point(20, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

            // Show the reference and live frame side by side
            Mat combined_frame;
            // Convert dlib image to OpenCV
            matrix<rgb_pixel> ref_image = img;
            Mat ref_img_mat(ref_image.nr(), ref_image.nc(), CV_8UC3, (void*)ref_image.begin(), ref_image.nc()*sizeof(rgb_pixel));
            cvtColor(ref_img_mat, ref_img_mat, COLOR_RGB2BGR);

            // Resize the reference image to match the live frame height
            Mat resized_ref_img;
            resize(ref_img_mat, resized_ref_img, Size(frame.cols, frame.rows));

            // Combine reference image and live frame horizontally
            hconcat(resized_ref_img, frame, combined_frame);

            // Display the combined image
            imshow("Face Matching - Reference & Live", combined_frame);
        }
        else
        {
            cout << "No faces found in captured image." << endl;
        }

        // Delay for 5 seconds before the next capture
        // Exit condition: press 'q' to quit
        if (waitKey(1) == 'q')
        {
            break;
        }
    }

    return 0;
}
catch (std::exception &e)
{
    cout << e.what() << endl;
}

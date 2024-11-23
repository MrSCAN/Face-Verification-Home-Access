#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "face_features.hh"

// Define global constants for model paths
const std::string SHAPE_PREDICTOR_PATH = "models/shape_predictor_5_face_landmarks.dat";
const std::string FACE_RECOGNITION_MODEL_PATH = "models/dlib_face_recognition_resnet_model_v1.dat";

// Face recognition model definition
template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;

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

using anet_type = dlib::loss_metric<dlib::fc_no_bias<128, dlib::avg_pool_everything<
                                                         alevel0<
                                                             alevel1<
                                                                 alevel2<
                                                                     alevel3<
                                                                         alevel4<
                                                                             dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<32, 7, 7, 2, 2, dlib::input_rgb_image_sized<150>>>>>>>>>>>>>;

// Define global models
dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
dlib::shape_predictor sp;
anet_type net;

// Initialize models
void initialize_models()
{
    dlib::deserialize(SHAPE_PREDICTOR_PATH) >> sp;
    dlib::deserialize(FACE_RECOGNITION_MODEL_PATH) >> net;
}

// Extract face features
std::vector<float> extract_face_features(const dlib::matrix<dlib::rgb_pixel> &img)
{
    // Detect faces
    std::vector<dlib::matrix<dlib::rgb_pixel>> faces;
    std::vector<dlib::rectangle> detected_faces = detector(img);

    if (detected_faces.empty())
    {
        std::cerr << "Warning: No faces found in the image." << std::endl;
        return {};  // Return empty if no faces are detected
    }

    // Extract face chips for each detected face
    for (auto face : detected_faces)
    {
        // Get the facial landmarks for the detected face
        dlib::full_object_detection shape = sp(img, face);

        // Print the number of landmarks detected for debugging purposes
        std::cout << "Detected " << shape.num_parts() << " landmarks." << std::endl;

        // Ensure that we have a valid face landmark (5 or 68 points)
        if (shape.num_parts() == 68 || shape.num_parts() == 5)
        {
            dlib::matrix<dlib::rgb_pixel> face_chip;
            try
            {
                // Extract the face chip (aligned face image)
                dlib::extract_image_chip(img, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);
                faces.push_back(std::move(face_chip));  // Add the face chip to the list
            }
            catch (const std::exception& e)
            {
                std::cerr << "Error extracting face chip: " << e.what() << std::endl;
            }
        }
        else
        {
            std::cerr << "Invalid number of landmarks detected: " << shape.num_parts() << std::endl;
        }
    }

    if (faces.empty())
    {
        std::cerr << "Warning: No valid face chips could be extracted." << std::endl;
        return {};  // Return empty if no valid face chips could be extracted
    }

    // Get the face descriptor for the first valid face
    dlib::matrix<float, 0, 1> face_descriptor = net(faces[0]);

    // Convert the descriptor to a std::vector<float> for storage
    return std::vector<float>(face_descriptor.begin(), face_descriptor.end());
}

// int main(int argc, char **argv)
// {
//     try
//     {
//         if (argc != 3)
//         {
//             std::cerr << "Usage: ./save_features <image_file> <name>" << std::endl;
//             return 1;
//         }

//         std::string image_file = argv[1];
//         std::string name = argv[2];

//         // Initialize global models
//         initialize_models();

//         // Load image using Dlib
//         dlib::matrix<dlib::rgb_pixel> img;
//         try
//         {
//             dlib::load_image(img, image_file);
//         }
//         catch (const std::exception &e)
//         {
//             std::cerr << "Error loading image: " << e.what() << std::endl;
//             return 1;
//         }

//         // Extract face features
//         std::vector<float> descriptor = extract_face_features(img);

//         if (descriptor.empty())
//         {
//             std::cerr << "Failed to extract features for " << name << " from image: " << image_file << std::endl;
//             return 1;
//         }

//         // Save the descriptor to the database
//         save_to_database(descriptor, name);

//         std::cout << "Features for " << name << " successfully saved to the database." << std::endl;
//         return 0;
//     }
//     catch (const std::exception &e)
//     {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return 1;
//     }
// }

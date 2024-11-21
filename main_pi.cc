#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
// #include <wiringPi.h>
#include <iostream>
#include <thread>
#include <chrono>
#include "db_utils.hh"
#include "face_features.hh"

// Function to simulate LED output
void set_leds(int green, int red, int yellow)
{
    std::cout << "LED Status - Green: " << (green ? "ON" : "OFF")
              << ", Red: " << (red ? "ON" : "OFF")
              << ", Yellow: " << (yellow ? "ON" : "OFF") << std::endl;
}

// Function to match a face descriptor with the database
bool match_face(const std::vector<float> &descriptor, float threshold = 0.6)
{
    std::vector<std::pair<std::string, std::vector<float>>> db_records = load_from_database();

    for (const auto &record : db_records)
    {
        const std::vector<float> &stored_descriptor = record.second;
        if (dlib::length(dlib::mat(descriptor) - dlib::mat(stored_descriptor)) < threshold)
        {
            std::cout << "Match found for: " << record.first << std::endl;
            return true;
        }
    }
    std::cout << "No match found." << std::endl;
    return false;
}

// Main function
int main(int argc, char **argv)
try
{
    // Command-line usage
    if (argc != 1 && argc != 2)
    {
        std::cout << "Usage: ./face_match [<image_file>]" << std::endl;
        return 1;
    }

    initialize_models();

    // Setup GPIO LEDs
    // setup_leds();

    bool use_camera = (argc == 1); // Decide input source: camera or image file
    cv::VideoCapture cap;
    std::string image_file;

    if (use_camera)
    {
        cap.open(0); // Open the default camera
        if (!cap.isOpened())
        {
            std::cerr << "Error: Could not open the camera." << std::endl;
            return -1;
        }
    }
    else
    {
        image_file = argv[1];
    }

    while (true)
    {
        dlib::matrix<dlib::rgb_pixel> img;
        if (use_camera)
        {
            // Capture frame from camera
            cv::Mat frame;
            cap >> frame;
            if (frame.empty())
            {
                std::cerr << "Error: Blank frame grabbed." << std::endl;
                set_leds(0, 0, 1); // Yellow LED
                continue;
            }
            dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(frame));
        }
        else
        {
            // Load image from file
            dlib::load_image(img, image_file);
        }

        try
        {
            // Extract face features from the image
            std::vector<float> descriptor = extract_face_features(img);

            if (descriptor.empty())
            {
                // If no face is detected or descriptor extraction fails
                set_leds(0, 0, 1); // Yellow LED
                continue;
            }

            // Match the extracted features with those in the database
            if (match_face(descriptor))
            {
                set_leds(1, 0, 0); // Green LED
            }
            else
            {
                set_leds(0, 1, 0); // Red LED
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
            set_leds(0, 0, 1); // Yellow LED
        }
    }

    return 0;
}
catch (const std::exception &e)
{
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
}

#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <iostream>
#include <vector>
#include <string>
#include "face_features.hh"
#include "crow.h"
#include <crow/multipart.h>
#include <dlib/image_io.h>
#include <dlib/matrix.h>
#include <fstream> // For std::ofstream

crow::response save_image_upload(const crow::request& req)
{
    try
    {
        // Parse the multipart form data using Crow's multipart namespace
        crow::multipart::message multipart_msg(req);

        // Check if "image" and "name" fields exist
        auto image_part = multipart_msg.get_part_by_name("image");
        auto name_part = multipart_msg.get_part_by_name("name");

        if (image_part.body.empty() || name_part.body.empty())
        {
            return crow::response(400, "Missing 'image' or 'name' field in the request.");
        }

        // Retrieve the image data
        std::string image_data = image_part.body;

        // Retrieve the name
        std::string name = name_part.body;

        // Save image data to a temporary file
        std::string temp_filename = "/tmp/uploaded_image.jpg";
        std::ofstream temp_file(temp_filename, std::ios::binary);
        temp_file.write(image_data.data(), image_data.size());
        temp_file.close();

        // Load the image data from the temporary file
        dlib::matrix<dlib::rgb_pixel> img;
        dlib::load_image(img, temp_filename);

        // Extract features
        std::vector<float> descriptor = extract_face_features(img);
        if (descriptor.empty())
        {
            return crow::response(400, "No faces detected in the image.");
        }

        // Save to database
        save_to_database(descriptor, name);

        // Optionally, delete the temporary file after use
        std::remove(temp_filename.c_str());

        return crow::response(200, "Features saved successfully.");
    }
    catch (const std::exception &e)
    {
        return crow::response(500, std::string("Error processing image: ") + e.what());
    }
}

// API: Save Image and Features
crow::response save_image_api(const crow::request& req)
{
    auto json_body = crow::json::load(req.body);
    if (!json_body)
    {
        return crow::response(400, "Invalid JSON");
    }

    std::string image_path = json_body["image_path"].s();
    std::string name = json_body["name"].s();

    try
    {
        dlib::matrix<dlib::rgb_pixel> img;
        dlib::load_image(img, image_path);

        // Extract features
        std::vector<float> descriptor = extract_face_features(img);
        if (descriptor.empty())
        {
            return crow::response(400, "No faces detected in the image.");
        }

        // Save to database
        save_to_database(descriptor, name);
        return crow::response(200, "Features saved successfully.");
    }
    catch (const std::exception &e)
    {
        return crow::response(500, std::string("Error processing image: ") + e.what());
    }
}

// API: Delete Features
crow::response delete_feature(const crow::request& req)
{
    auto json_body = crow::json::load(req.body);
    if (!json_body)
    {
        return crow::response(400, "Invalid JSON");
    }

    std::string name = json_body["name"].s();

    try
    {
        // delete from database
        delete_from_database(name);
        return crow::response(200, "Features deleted successfully.");
    }
    catch (const std::exception &e)
    {
        return crow::response(500, std::string("Error deleting feature : ") + e.what());
    }
}

// API: Continuous Recognition (Stub Implementation)
crow::response continuous_recognition(const crow::request& req)
{
    // Here, you would interact with your real-time API.
    // Example: Query the current image from a camera or an image buffer.
    return crow::response(200, "Continuous recognition is running.");
}

// Home
crow::response home_page(const crow::request& req)
{
    // Here, you would interact with your real-time API.
    // Example: Query the current image from a camera or an image buffer.
    return crow::response(200, "ok");
}


int main()
{
    // Initialize models
    initialize_models();

    crow::SimpleApp app;

    // Define routes
    CROW_ROUTE(app, "/api/v0/add")
        .methods("POST"_method)(save_image_upload);

        // Define routes
    CROW_ROUTE(app, "/api/v0/add_api")
        .methods("POST"_method)(save_image_api);

    // Define routes
    CROW_ROUTE(app, "/api/v0/remove")
        .methods("POST"_method)(delete_feature);

    CROW_ROUTE(app, "/api/v0/run")
        .methods("GET"_method)(continuous_recognition);

    CROW_ROUTE(app, "/")
        .methods("GET"_method)(home_page);

    // Start server
    app.port(8080).multithreaded().run();
    return 0;
}

#include <sqlite3.h>
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <filesystem>

#include "face_features.hh"

// Constants
const std::string DB_PATH = std::string(getenv("HOME")) + "/pi/fras/face_features.db";

void create_dirs()
{
    // Ensure the directory exists
    try
    {
        // Extract the directory part of the path
        std::filesystem::path db_path(DB_PATH);
        std::filesystem::path dir_path = db_path.parent_path();

        // Create the directory if it doesn't exist
        if (!std::filesystem::exists(dir_path))
        {
            std::filesystem::create_directories(dir_path);
            std::cout << "Created directory: " << dir_path << std::endl;
        }
    }
    catch (const std::filesystem::filesystem_error &e)
    {
        std::cerr << "Error: Failed to create directory: " << e.what() << std::endl;
        return;
    }
}

// Function to load descriptors from the database
std::vector<std::pair<std::string, std::vector<float>>> load_from_database()
{
    sqlite3 *db = nullptr;
    sqlite3_stmt *stmt = nullptr;
    std::vector<std::pair<std::string, std::vector<float>>> records;

    create_dirs();

    // Open the database
    if (sqlite3_open(DB_PATH.c_str(), &db) != SQLITE_OK)
    {
        std::cerr << "Error: Failed to open database: " << sqlite3_errmsg(db) << std::endl;
        return records;
    }

    // Prepare the SQL statement
    const std::string sql = "SELECT name, descriptor FROM face_features";
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK)
    {
        std::cerr << "Error: Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return records;
    }

    // Execute the query and process the results
    while (sqlite3_step(stmt) == SQLITE_ROW)
    {
        std::string name = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0));
        const void *blob = sqlite3_column_blob(stmt, 1);
        int blob_size = sqlite3_column_bytes(stmt, 1);

        if (blob)
        {
            // Deserialize the vector<float>
            std::vector<float> descriptor(blob_size / sizeof(float));
            std::memcpy(descriptor.data(), blob, blob_size);
            records.emplace_back(std::move(name), std::move(descriptor));
        }
    }

    // Cleanup
    sqlite3_finalize(stmt);
    sqlite3_close(db);

    return records;
}

// Function to save descriptors to the database
void save_to_database(const std::vector<float> &descriptor, const std::string &name)
{
    create_dirs();

    sqlite3 *db = nullptr;
    char *err_msg = nullptr;

    // Open the database
    if (sqlite3_open(DB_PATH.c_str(), &db) != SQLITE_OK)
    {
        std::cerr << "Error: Cannot open database: " << sqlite3_errmsg(db) << std::endl;
        return;
    }

    // Create the table if it does not exist
    const char *create_table_sql = R"(
        CREATE TABLE IF NOT EXISTS face_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            descriptor BLOB NOT NULL
        );
    )";

    if (sqlite3_exec(db, create_table_sql, nullptr, nullptr, &err_msg) != SQLITE_OK)
    {
        std::cerr << "Error: Failed to create table: " << err_msg << std::endl;
        sqlite3_free(err_msg);
        sqlite3_close(db);
        return;
    }

    // Prepare the SQL statement to insert data
    sqlite3_stmt *stmt = nullptr;
    const char *insert_sql = "INSERT INTO face_features (name, descriptor) VALUES (?, ?)";

    if (sqlite3_prepare_v2(db, insert_sql, -1, &stmt, nullptr) != SQLITE_OK)
    {
        std::cerr << "Error: Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return;
    }

    // Bind values to the prepared statement
    sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_blob(stmt, 2, descriptor.data(), descriptor.size() * sizeof(float), SQLITE_TRANSIENT);

    // Execute the statement
    if (sqlite3_step(stmt) != SQLITE_DONE)
    {
        std::cerr << "Error: Failed to execute statement: " << sqlite3_errmsg(db) << std::endl;
    }
    else
    {
        std::cout << "Info: Descriptor saved successfully for: " << name << std::endl;
    }

    // Clean up
    sqlite3_finalize(stmt);
    sqlite3_close(db);
}

// Function to delete a row from the database using the name
void delete_from_database(const std::string &name)
{
    sqlite3 *db = nullptr;
    char *err_msg = nullptr;

    // Open the database
    if (sqlite3_open(DB_PATH.c_str(), &db) != SQLITE_OK)
    {
        std::cerr << "Error: Cannot open database: " << sqlite3_errmsg(db) << std::endl;
        return;
    }

    // Prepare the SQL statement to delete data
    sqlite3_stmt *stmt = nullptr;
    const char *delete_sql = "DELETE FROM face_features WHERE name = ?";

    if (sqlite3_prepare_v2(db, delete_sql, -1, &stmt, nullptr) != SQLITE_OK)
    {
        std::cerr << "Error: Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return;
    }

    // Bind the name to the prepared statement
    sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_TRANSIENT);

    // Execute the statement
    if (sqlite3_step(stmt) != SQLITE_DONE)
    {
        std::cerr << "Error: Failed to execute statement: " << sqlite3_errmsg(db) << std::endl;
    }
    else
    {
        std::cout << "Info: Row deleted successfully for: " << name << std::endl;
    }

    // Clean up
    sqlite3_finalize(stmt);
    sqlite3_close(db);
}

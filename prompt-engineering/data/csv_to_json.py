# ---------------------------------------------------------------#
# Script to Convert a CSV file to a JSON file
# (Surya's prompt engineering dataset)
# ---------------------------------------------------------------#

import csv
import json
import os


def convert_csv_to_json(csv_path, json_path):
    # List to store the converted data
    json_data = []

    # Read the CSV file
    with open(csv_path, "r", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)

        # Process each row
        for row in csv_reader:
            json_entry = {
                "input_text": row["English Sentence"],
                "translation_spanish": row["Spanish Translation"],
                "translation_french": row["French Translation"],
                "translation_chinese": row["Chinese Translation"],
                "translation_japanese": row["Japanese Translation"],
                "translation_hindi": row["Hindi Translation"],
                "translation_arabic": row["Arabic Translation"],
                "translation_russian": row["Russian Translation"],
            }

            # Add the entry to our data list
            json_data.append(json_entry)

    # Write the JSON data to a file
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=2)

    print(f"Conversion complete: {csv_path} → {json_path}")
    print(f"Converted {len(json_data)} entries")


if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Ask user for input and output file names
    csv_filename = input("Enter the name of your CSV file (e.g., translations.csv): ")
    json_filename = input(
        "Enter the name for the output JSON file (e.g., dataset.json): "
    )

    # Create full paths
    csv_path = os.path.join(script_dir, csv_filename)
    json_path = os.path.join(script_dir, json_filename)

    # Check if input file exists
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
    else:
        convert_csv_to_json(csv_path, json_path)

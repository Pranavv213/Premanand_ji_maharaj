from pathlib import Path

# Set your fixed folder name or absolute path here
FOLDER_PATH = Path("./transcripts")  # Change this to your desired folder path
OUTPUT_FILE = FOLDER_PATH / "merged_output.txt"

def merge_text_files():
    # Ensure the folder exists before running
    if not FOLDER_PATH.exists():
        print(f"Error: The folder '{FOLDER_PATH}' does not exist.")
        return

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        # Sorts alphabetically for a consistent merge order
        for file_path in sorted(FOLDER_PATH.glob('*.txt')):
            # Skip the output file if it is stored in the same folder
            if file_path.resolve() == OUTPUT_FILE.resolve():
                continue
                
            with open(file_path, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
                # Adds spacing between the contents of different files
                outfile.write('\n\n')
                
    print(f"Successfully merged files into: {OUTPUT_FILE}")

if __name__ == "__main__":
    merge_text_files()
def find_and_count_sscores(file_path):
    count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if 'sScore' in line:
                print(line.strip())
                count += 1
    
    print(f"\nTotal number of lines containing 'sScore': {count}")

# Use the function
try:
    find_and_count_sscores('log2.txt')
except FileNotFoundError:
    print("Error: 'log2.txt' not found in the current directory.")
except Exception as e:
    print(f"An error occurred: {e}")
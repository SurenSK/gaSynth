import pandas as pd
import json

# Function to parse each JSON-like line
def parse_line(line):
    try:
        # Try to parse JSON
        return json.loads(line.strip().rstrip(','))
    except json.JSONDecodeError:
        # Handle the case where line is not a complete JSON object
        return None

# Load and clean the data
file_path = 'survey_responses.jsonl'
data = []
with open(file_path, 'r') as file:
    for line in file:
        parsed_line = parse_line(line)
        if parsed_line:
            data.append(parsed_line)

# Convert to DataFrame
df = pd.DataFrame(data)

# Replace user hashes with names
df['user_hash'] = df['user_hash'].replace({
    '0d61a1fe506c38c11909d216b922d3d6': 'John',
    'e73dee03866ad4a178176b083eed638a': 'Neha',
    "46ffead3f7e16fc76855c5173e9abe33": "Vineet"
})

# Convert the date-time column to datetime format
df['date-time'] = pd.to_datetime(df['date-time'])

# Separate Neha's responses based on the date
df.loc[(df['user_hash'] == 'Neha') & (df['date-time'] >= '2024-07-16'), 'user_hash'] = 'Nehb'

# Convert relevance and completeness to numeric
df['relevance'] = pd.to_numeric(df['relevance'], errors='coerce')
df['completeness'] = pd.to_numeric(df['completeness'], errors='coerce')

# Group by user and generation to count the number of non-null relevance and completeness values
count_table_relevance = df.groupby(['user_hash', 'gen#'])['relevance'].count().unstack(fill_value=0)
count_table_completeness = df.groupby(['user_hash', 'gen#'])['completeness'].count().unstack(fill_value=0)

# Rename the columns for clarity
count_table_relevance.columns = [f'Relv G{col}' for col in count_table_relevance.columns]
count_table_completeness.columns = [f'Comp G{col}' for col in count_table_completeness.columns]

# Combine the tables into one
count_table = pd.concat([count_table_relevance, count_table_completeness], axis=1)

# Display the count table
print("Count Table:")
print(count_table)

# Group by user and generation to calculate the mean relevance and completeness values
mean_table_relevance = df.groupby(['user_hash', 'gen#'])['relevance'].mean().unstack(fill_value=0).round(1)
mean_table_completeness = df.groupby(['user_hash', 'gen#'])['completeness'].mean().unstack(fill_value=0).round(1)

# Rename the columns for clarity
mean_table_relevance.columns = [f'Relv G{col}' for col in mean_table_relevance.columns]
mean_table_completeness.columns = [f'Comp G{col}' for col in mean_table_completeness.columns]

# Combine the tables into one
mean_table = pd.concat([mean_table_relevance, mean_table_completeness], axis=1)

# Display the mean table
print("\nMean Table:")
print(mean_table)

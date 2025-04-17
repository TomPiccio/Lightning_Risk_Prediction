import os
import pandas as pd
from bs4 import BeautifulSoup
import re
from datetime import datetime, time, timedelta
from collections import defaultdict
from tqdm import tqdm

data_folder = "data/lightning_risk_data_raw_test"
output_folder = "data/lightning_risk_data_test"

source_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
codes = ['1N', '1S', '2', '3N', '3S', '4', '5', '6', '7', '8N', '8S', '9', 'L1', 'L2', 'L3', 'L4', '10N', '10S', '11E', '11W', '12', '13N', '13S', '14', '15', '16N', '16S', '17', '18E', '18W', '19N', '19S']
columns = ['DateTime', 'Date', 'Time'] + codes  # Add the codes as columns

def extract_datetime(message_div : BeautifulSoup):
    details = message_div.find("span", class_="date details")
    if details != None:
        details = pd.to_datetime(details.get("title"), format='%d.%m.%Y %H:%M:%S UTC%z')
    else:
        details = message_div.find("div", class_="pull_right date details")
        if details != None:
            details = pd.to_datetime(details.get("title"), format='%d.%m.%Y %H:%M:%S UTC%z')
    return details

def extract_multiple_groups(text: str):
    # Define regex patterns for both formats
    pattern_1 = r"([A-Za-z0-9,]+):\(?(\d{4}-\d{4})\)?" # Format 1: '1N,L1,L2,...:1715-1800'
    pattern_2 = r"\((\d{4}-\d{4})\)\n\s*([A-Za-z0-9,]+)"  # Format 2: '(1020-1100)\n L2,L3,L4,...'

    #Standardize Format
    text = re.sub(r' \(', r':(', text)  # Replace ' (' with ':('
    text = re.sub(r'\(\n', r'(', text)   # Replace '(\n' with '('
    text = re.sub(r'\n\)', r')', text)   # Replace '\n)' with ')'
    text = re.sub(r'\n\)', r')', text)   # Replace '\n)' with ')'
    text = re.sub(r":\n(\d{4}-\d{4})", r":\1", text)
    
    # Find all matches for both formats
    matches_1 = re.findall(pattern_1, text)
    matches_2 = re.findall(pattern_2, text)

    result = []

    if len(matches_1) > 0  and len(matches_2)> 0:
        print(matches_1,matches_2)

    if len(matches_1) > len(matches_2):
        # Process all matches for Format 1
        for match in matches_1:
            if "May" in match or "May" in match[0]:
                print(matches_1)
                break
            result.append({
                "area_codes": match[0].split(','),
                "time_range": match[1]
            })
    else:
        # Process all matches for Format 2
        for match in matches_2:
            
            if "May" in match or "May" in match[0]:
                print(matches_2,"2")
                break
            result.append({
                "time_range": match[0],
                "area_codes": match[1].split(',')
            })

    return result


def row_generation(date: pd.Timestamp, time_range: str, groups: list):
    row = defaultdict(int)
    row['DateTime'] = date
    row['Date'] = date.date()
    row['Time'] = date.time()
    start_str, end_str = time_range.split("-")
    try:
        start_time = time(int(start_str[:2])%24, int(start_str[2:]))
        end_time = time(int(end_str[:2])%24, int(end_str[2:]))
    except:
        print("Error parse time:", start_str, end_str, time_range)
    start_datetime = pd.Timestamp(datetime.combine(date.date(), start_time), tz=date.tz) 
    end_datetime = pd.Timestamp(datetime.combine(date.date(), end_time), tz=date.tz) 
    # If end time is earlier than start time, it means it's on the next day
    if end_datetime < start_datetime:
        end_datetime += timedelta(days=1)
    
    delta_minutes = int((end_datetime - date).total_seconds() / 60)

    #Special Cases

    if delta_minutes > 1000:
        delta_minutes -= 24*60

    if delta_minutes<0:
        return None
    
    if not 'Clear' in groups:
        for group in groups:
            group = group.lstrip("0")
            if not group in codes:
                print(group,start_str, end_str)
            row[group] = delta_minutes
    for column in columns:
        if column in row:
            continue
        row[column] = 0
    return row
            
def process_file(source_file_path,file_num):
    file_name = os.path.basename(source_file_path)

    with open(source_file_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Find all div elements with class="text"
    messages = soup.find_all("div", class_=lambda x: x == "body")
    
    dates = [extract_datetime(div) for div in messages]
    groups = [extract_multiple_groups(content.get_text(separator="\n")) if content != None else None for content in messages]
    data = []
    for i in tqdm(range(len(dates)), desc=f"Processing file {file_name} ({file_num+1}/{len(source_files)})"):
        if groups[i] == None or dates[i] == None:
            continue
        for group in groups[i]:
            try:
                row = row_generation(dates[i],group["time_range"],group["area_codes"])
                if row == None:
                    continue
                data.append(dict(row))
            except Exception as e:
                print("Error occured:",e)
                print(group)
    print(len(data))
    return data

compiled_data = []

for file_num in range(len(source_files)):
    try:
        compiled_data.extend(process_file(source_files[file_num],file_num))
    except Exception as e:
        print(source_files[file_num],e)

#Initialize columns
df = pd.DataFrame(compiled_data)
print(df.head())
df['DateTime'] = pd.to_datetime(df['DateTime'])  # Convert to datetime
df["DateTime"] = df["DateTime"].dt.tz_localize(None)
df['Date'] = pd.to_datetime(df['Date']).dt.date  # Convert to date only
df['Time'] = df['Time'].astype(str)  # Ensure Time is a string (you can convert to timedelta if needed)
# Convert code columns to int
for code in codes:
    if code not in df.columns:
        print(code,"is missing")
    df[code] = df[code].astype(int)  # Ensure the code columns are integers
df.fillna(0, inplace=True)
df["DateTime"] = df["DateTime"].dt.strftime("%Y-%m-%dT%H:%M:%S+08:00")
df_agg = df.groupby(["DateTime", "Date", "Time"], as_index=False).max()

# Save to CSV
df_agg.sort_values(by="DateTime", inplace=True)
df_agg.to_csv(os.path.join(output_folder,"lightning_risk_data_mins.csv"),index=False)


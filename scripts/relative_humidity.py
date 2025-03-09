import aiohttp
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import json
import os

# API settings
BASE_URL = "https://api-open.data.gov.sg/v2/real-time/api/relative-humidity"
start_date = datetime(2020, 4, 20)
end_date = datetime(2025, 2, 17)

# Store data
weather_data_list = []

def store_weather_data(new_data, folder_path, date: datetime):
    '''save the data as a json file'''
    folder_path = "data/relative_humidity/"

    # create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, f"{date.strftime('%Y-%m-%d')}.json")

    # write a json file
    with open(file_path, "w") as file:
        json.dump(new_data, file, indent=4)


async def fetch_weather_async(session, date):
    """Fetch lightning data for a specific date."""
    formatted_date = date.strftime("%Y-%m-%d")  # Correct date format
    params = {"date": formatted_date}

    async with session.get(BASE_URL, params=params) as response:
        if response.status == 200:
            return await response.json()
        else:
            print(f"Error {response.status} for {formatted_date}: {await response.text()}")
            return None

async def main():
    folder_path = "data/relative_humidity/"

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_weather_async(session, start_date + timedelta(days=i)) for i in range((end_date - start_date).days + 1)]
        results = await asyncio.gather(*tasks)
        for i, data in enumerate(results):
            store_weather_data(data, folder_path, start_date + timedelta(days=i))

asyncio.run(main())

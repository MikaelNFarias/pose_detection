import requests
import os
import sys
from dotenv import load_dotenv
sys.path.append('..')
from src.directories import *

try:
    load_dotenv()
except Exception as e:
    print('Error loading .env file. Please make sure you have created one with your Unsplash API key.')
    print('You can create one by running the command: `cp .env.example .env`')
    sys.exit(1)

def download_images(query, count,page,dest_folder, api_key):
    os.makedirs(dest_folder, exist_ok=True)
    BASE_URL = 'https://api.unsplash.com'
    ROUTE = 'search/photos'
    URL = f'{BASE_URL}/{ROUTE}'
    QUERY_PARAMS = {
        "query": "living_room",
        "per_page": int(count),
        "page": int(page),
        "client_id": os.getenv("UNSPLASH_API_KEY")
    }

    try:
        for i in range(1,page+1):
            QUERY_PARAMS['page'] = i
            response = requests.get(URL,params=QUERY_PARAMS)
            data = response.json()
            for i, image in enumerate(data['results']):
                image_url = image['urls']['regular']
                image_response = requests.get(image_url)
                with open(os.path.join(dest_folder, f'background_{i+1}.jpg'), 'wb') as f:
                    f.write(image_response.content)
                print(f'Downloaded {i+1}/{count} images')
    except Exception as e:
        print('Error downloading images:', e)

# Set your Unsplash API key here
unsplash_api_key = os.getenv("UNSPLASH_API_KEY")

# Define the search query and number of images you need
search_query = 'livingroom'
num_images = 30 # max per hour
page = 2
destination_folder = os.path.join(ROOT_DIR,'downloads')

# Download the images
download_images(search_query, num_images,page, destination_folder, unsplash_api_key)

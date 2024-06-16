import requests
import os
import sys
from dotenv import load_dotenv
sys.path.append('..')
from src.directories import *
import random

try:
    load_dotenv()
except Exception as e:
    print('Error loading .env file. Please make sure you have created one with your Unsplash API key.')
    print('You can create one by running the command: `cp .env.example .env`')
    sys.exit(1)

def download_images(count: int,
                    dest_folder: str,
                    pages: int = 5) -> None\
        :

    os.makedirs(dest_folder, exist_ok=True)
    BASE_URL = 'https://api.unsplash.com'
    ROUTE = 'search/photos'
    QUERIES = ['living_room',
                  "blank_wall",
                  "neutral_wall",
                  "modern_furniture",
                  "scandinavian_interior",
                  "cozy_corner",
                  "textured_wall"]

    URL = f'{BASE_URL}/{ROUTE}'
    QUERY_PARAMS = {
        "query":random.choice(QUERIES),
        "per_page": int(count),
        "client_id": os.getenv("UNSPLASH_API_KEY")
    }

    try:
        for i in range(1,pages+1):
            QUERY_PARAMS['page'] = i
            response = requests.get(URL,params=QUERY_PARAMS)
            data = response.json()
            for k, image in enumerate(data['results']):
                image_url = image['urls']['regular']
                image_response = requests.get(image_url)
                with open(os.path.join(dest_folder, f'background_{(k + 1) * i}.jpg'), 'wb') as f:
                    f.write(image_response.content)
                print(f'Downloaded {(k+1) * i}/{count} images')
    except Exception as e:
        print('Error downloading images:', e)

# Set your Unsplash API key here
unsplash_api_key = os.getenv("UNSPLASH_API_KEY")

# Define the search query and number of images you need
num_images = 30 # max per hour
destination_folder = os.path.join(ROOT_DIR,'downloads')

# Download the images
download_images(count=30,
                dest_folder = destination_folder,
                pages = 5)

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
        "per_page": int(count),
        "client_id": os.getenv("UNSPLASH_API_KEY")
    }

    try:
        for page in range(1, pages+1):
            QUERY_PARAMS['page'] = page
            QUERY_PARAMS['query'] = random.choice(QUERIES)
            response = requests.get(URL, params=QUERY_PARAMS)
            response.raise_for_status()
            images = response.json()['results']
            for image in images:
                image_url = image['urls']['regular']
                image_id = image['id']
                image_path = os.path.join(dest_folder, f'{image_id}.jpg')
                with open(image_path, 'wb') as file:
                    image_response = requests.get(image_url)
                    file.write(image_response.content)
                    print(f'Downloaded image {image_id} to {image_path}')
    except Exception as e:
        print(f'Error downloading images: {e}')

# Set your Unsplash API key here
unsplash_api_key = os.getenv("UNSPLASH_API_KEY")

# Define the search query and number of images you need
num_images = 30 # max per hour
destination_folder = os.path.join(ROOT_DIR,'downloads')

# Download the images
download_images(count=30,
                dest_folder = destination_folder,
                pages = 5)

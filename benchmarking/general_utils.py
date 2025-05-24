from nuclia import sdk
import os
from dotenv import load_dotenv
import requests
from tqdm import tqdm
import json


USER_TOKEN = os.getenv('NUCLIA_USER_TOKEN')
KB_URL = os.getenv('NUCLIA_KB_URL')
KB_ID = os.getenv('NUCLIA_KB_ID')
API_KEY = os.getenv('NUCLIA_API_KEY')


def extract_triplets(kb_url:str):

    # sdk.NucliaAuth().set_user_token(USER_TOKEN)
    # sdk.NucliaAuth().kb(url=KB_URL, token=API_KEY)
    #
    # kbs = sdk.NucliaKBS()
    # kbs.default(KB_ID)
    #
    # resource = sdk.NucliaResource()
    # knowledge_graph = resource.get(rid=RID)

    headers = {
        "accept": "*/*",  # Accept all response types
        "accept-language": "en-US,en;q=0.9",  # Language preference
        "X-NUCLIA-SERVICEACCOUNT": f"Bearer {API_KEY}",  # Token for authentication
        "content-type": "application/json",  # Ensures we're sending JSON requests
        "x-ndb-client": "dashboard",  # Identifies the client making the request
    }

    # Make a GET request to fetch all resources from the knowledge box
    response = requests.get(f"{kb_url}/resources", headers=headers)

    # Print the response status to check if the request was successful
    print(f"Response Status Code: {response.status_code}")

    # Handle unsuccessful requests by printing the error message and stopping execution
    if response.status_code != 200:
        print(f"Error: {response.text}")
        raise Exception("Failed to fetch resources from Nuclia")

    # Parse the JSON response and extract resource IDs
    resources = response.json()
    resources_ids = [resource["id"] for resource in resources.get("resources", [])]

    # Display the number of resources found
    print(f"Total Resources Found: {len(resources_ids)}")

    # Initialize an empty list to store all relations
    relations = []

    # Iterate over each resource ID using tqdm for a progress bar
    for resource_id in tqdm(resources_ids, desc="Extracting Relations"):
        # Construct the URL for fetching detailed resource data
        url = f"{kb_url}/resource/{resource_id}?show=basic&show=origin&show=extracted"

        # Make a GET request to fetch the resource data
        response = requests.get(url, headers=headers)

        # If the request fails, print an error and continue to the next resource
        if response.status_code != 200:
            print(f"Error fetching resource {resource_id}: {response.text}")
            continue

        # Parse the JSON response
        result = response.json()

        # Extract relations from the resource
        for tag in ["links", "files", "texts"]:
            if tag in result["data"]:
                # Extract relations if metadata is present
                rels = [
                    result["data"][tag][f]["extracted"]["metadata"]["metadata"]["relations"]
                    for f in result["data"][tag]
                    if "metadata" in result["data"][tag][f]["extracted"]
                       and "metadata" in result["data"][tag][f]["extracted"]["metadata"]
                ]

                # Flatten the list (convert list of lists into a single list)
                rels = [item for sublist in rels for item in sublist]

                # Filter out relations that do not have "data_augmentation_task_id"
                rels = [rel for rel in rels if "data_augmentation_task_id" in rel["metadata"]]

                # Append the extracted relations to the main list
                relations.extend(rels)

                # Print how many relations were found for this specific tag
                print(f"{len(rels)} relations found in {tag} for resource {resource_id}")


    return relations


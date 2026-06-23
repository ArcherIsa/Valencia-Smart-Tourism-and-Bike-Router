import pandas as pd
import requests
import urllib3
import time  

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_wiki_image(monument_name):
    """Searches Wikipedia for a monument and returns its main image URL."""
    
    time.sleep(2)
    
    headers = {
        "User-Agent": "ValenciaSmartBikeTour/1.0 (Contact: your_email@example.com)"
    }
    
    try:
        search_url = f"https://es.wikipedia.org/w/api.php?action=query&list=search&srsearch={monument_name} Valencia&utf8=&format=json"
        res = requests.get(search_url, headers=headers, verify=False, timeout=10)
        
        if res.status_code != 200:
            print(f"[{monument_name}] Blocked at Search: HTTP {res.status_code}")
            return None
            
        res_json = res.json()
        
        if not res_json.get('query', {}).get('search'):
            print(f"[{monument_name}] No Wikipedia article found.")
            return None
            
        title = res_json['query']['search'][0]['title']
        
        img_url = f"https://es.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&piprop=original&titles={title}"
        img_res = requests.get(img_url, headers=headers, verify=False, timeout=10)
        
        if img_res.status_code != 200:
            print(f"[{monument_name}] Blocked at Image Fetch: HTTP {img_res.status_code}")
            return None
            
        img_json = img_res.json()
        pages = img_json['query']['pages']
        
        for page_id, page_data in pages.items():
            if 'original' in page_data:
                print(f"[{monument_name}] Success!")
                return page_data['original']['source']
                
        print(f"[{monument_name}] Article found, but no main image available.")
        return None
        
    except Exception as e:
        print(f"[{monument_name}] Python Error: {e}")
        return None

print("Loading CSV...")
try:
    df = pd.read_csv('monument_tags.csv')
    
    print("Fetching images from Wikipedia...")
    df['image_url'] = df['name'].apply(lambda x: get_wiki_image(str(x)) if pd.notna(x) else None)

    df.to_csv('monument_tags_images.csv', index=False)
    print("Done! Saved to monument_tags_images.csv")
    
except FileNotFoundError:
    print("Error: Could not find 'monument_tags.csv'. Make sure the script is in the same folder as the CSV!")
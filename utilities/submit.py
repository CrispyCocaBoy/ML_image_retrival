import json
import requests
from typing import Dict, List

def submit(data: Dict[str, List[str]], group_name = "Simple_Guys", url="http://tatooine.disi.unitn.it:3001/retrieval/"):
    """
    Invia i risultati del retrieval al server per la valutazione.

    Args:
        data: dizionario {query_image_name: [top_k_retrieved_images]}
        group_name: nome del gruppo per l’identificazione
        url: endpoint per la sottomissione (di default quello fornito)
    """
    payload = {
        "groupname": group_name,
        "images": data
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        print(f"accuracy is {result.get('accuracy', 'N/A')}")
        return result.get('accuracy', None)
    except requests.RequestException as e:
        print(f"Request failed: {e}")
    except json.JSONDecodeError:
        print(f"Server response is not JSON:\n{response.text}")

import json
import requests

def submit(results, groupname, url="http://tatooine.disi.unitn.it:3001/retrieval/"):
    res = {
        "groupname": groupname,
        "images": results
    }
    res = json.dumps(res)
    response = requests.post(url, res)

    try:
        result = json.loads(response.text)
        print(f"✅ Accuracy: {result['accuracy']}")
        return result['accuracy']
    except json.JSONDecodeError:
        print(f"❌ Error decoding response: {response.text}")
        return None

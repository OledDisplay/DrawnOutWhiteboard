import json, urllib.request

def api_get_images(prompts: dict, host="127.0.0.1", port=8787, timeout=300):
    url = f"http://{host}:{port}/get_images"
    payload = json.dumps({"prompts": prompts}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

print(api_get_images({"Eukaryotic cell":"Biology"}))

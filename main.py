import requests

BOARD_TOKEN = "cloudflare"

url = (
    f"https://boards-api.greenhouse.io/v1/"
    f"boards/{BOARD_TOKEN}/jobs?content=true"
)

response = requests.get(url, timeout=10)
response.raise_for_status()

data = response.json()


for job in data["jobs"]:
    print(job["title"])
    print(job["location"]["name"])
    print(job["absolute_url"])
    print()
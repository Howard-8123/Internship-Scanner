import requests

BOARD_TOKEN = "cloudflare"
COMPANY_NAME = "Cloudflare"


def fetch_greenhouse_jobs(board_token):
    url = (
        f"https://boards-api.greenhouse.io/v1/"
        f"boards/{board_token}/jobs?content=true"
    )

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    return response.json().get("jobs", [])


def normalise_greenhouse_job(raw_job, company_name):
    return {
        "source": "greenhouse",
        "source_job_id": str(raw_job["id"]),
        "company": company_name,
        "title": raw_job.get("title", ""),
        "location": raw_job.get("location", {}).get("name", "Unknown"),
        "description_html": raw_job.get("content", ""),
        "updated_at": raw_job.get("updated_at"),
        "application_url": raw_job.get("absolute_url", "")
    }


raw_jobs = fetch_greenhouse_jobs(BOARD_TOKEN)

normalised_jobs = [
    normalise_greenhouse_job(raw_job, COMPANY_NAME)
    for raw_job in raw_jobs
]

print(f"Downloaded {len(raw_jobs)} jobs")
print(f"Normalised {len(normalised_jobs)} jobs\n")

for job in normalised_jobs[:5]:
    print(job["company"])
    print(job["title"])
    print(job["location"])
    print(job["application_url"])
    print()
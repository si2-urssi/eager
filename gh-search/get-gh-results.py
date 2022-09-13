import time
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from fastcore.net import HTTP4xxClientError
from ghapi.all import GhApi

load_dotenv()

###############################################################################

SEARCH_QUERIES = [
    "National Science Foundation",
    "NSF Award",
    "NSF Grant",
    "Supported by the NSF",
    "Supported by NSF",
]

BATCH_SIZE = 10

###############################################################################

# Load GH API
# Env variable of GITHUB_TOKEN with a PAT must be in `.env` file.
api = GhApi()

# Get all results for each term
results = []
query_start_time = time.time()
for query in SEARCH_QUERIES:
    complete_query = f'"{query}" filename:README.md'

    # Get initial
    page = 0
    all_gathered = False
    while not all_gathered:
        try:
            print(f"Querying: '{complete_query}', Page: {page}")
            page_results = api("/search/code", "GET ", query=dict(q=complete_query, per_page=BATCH_SIZE, page=page))
            total_count = page_results["total_count"]
            items_returned = page_results["items"]

            # Unpack results
            for item in items_returned:
                repo_details = item["repository"]
                repo_name = repo_details["name"]
                owner_details = repo_details["owner"]
                owner_name = owner_details["login"]
                full_name = f"{owner_name}/{repo_name}"

                # Get languages
                languages = api(f"/repos/{full_name}/languages")

                # Get latest commit datetime
                commits = api(f"/repos/{full_name}/commits")
                most_recent_commit = commits[0]["commit"]
                most_recent_committer = most_recent_commit["committer"]
                most_recent_committer_name = most_recent_committer["name"]
                most_recent_committer_email = most_recent_committer["email"]
                most_recent_commit_dt = datetime.fromisoformat(
                    # We remove last character because it is 'Z' for "Zulu"
                    # Datetimes are naturally UTC/Zulu
                    most_recent_committer["date"][:-1]
                )

                # Append this result to all results
                results.append(
                    {
                        "owner": owner_name,
                        "name": repo_name,
                        "link": f"https://github.com/{full_name}",
                        "languages": "; ".join(languages.keys()),
                        "most_recent_committer_name": most_recent_committer_name,
                        "most_recent_committer_email": most_recent_committer_email,
                        "most_recent_commit_datetime": most_recent_commit_dt.isoformat(),
                        "most_recent_commit_timestamp": most_recent_commit_dt.timestamp(), 
                        "query": query,
                    }
                )

            # Store partial results always
            pd.DataFrame(results).to_csv("gh-results.csv", index=False)

            # Increase page and keep going
            page += 1

            # Wait to avoid rate limiting
            print("Sleeping for forty seconds...")
            time.sleep(40)

            # Update time estimate
            batch_time = time.time()
            seconds_diff = batch_time - query_start_time
            seconds_diff_per_page = seconds_diff / page
            total_pages_required = total_count / BATCH_SIZE
            remaining_pages = total_pages_required - page
            estimated_remaining_seconds = seconds_diff_per_page * remaining_pages
            estimated_remained_pages = remaining_pages
            print(
                f"Remaining pages: {estimated_remained_pages} "
                f"(of {total_pages_required} -- "
                f"est. {timedelta(seconds=estimated_remaining_seconds)})"
            )

            # Break because we are done
            if len(items_returned) == 0:
                break

        except HTTP4xxClientError as e:
            print(f"Caught exception: {e}")
            pass

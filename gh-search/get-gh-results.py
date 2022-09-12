import time

import pandas as pd
from dotenv import load_dotenv
from ghapi.all import GhApi

load_dotenv()

SEARCH_QUERIES = [
    "National Science Foundation",
    "NSF Award",
    "NSF Grant",
    "Supported by the NSF",
    "Supported by NSF",
]

TEMP_STORAGE_FILE = "gh-page-results.json"

# Load GH API
# Env variable of GITHUB_TOKEN with a PAT must be in `.env` file.
api = GhApi()

# Get all results for each term
results = []
for query in SEARCH_QUERIES:
    complete_query = f'"{query}" filename:README.md'

    # Get initial
    page = 0
    all_gathered = False
    while not all_gathered:
        print(f"Querying: '{complete_query}', Page: {page}")
        try:
            results = api("/search/code", "GET ", query=dict(q=complete_query, per_page=30, page=page))
            items_returned = results["items"]
            if len(items_returned) != 30:
                print(
                    f"Query failed to return all 30 requested items. "
                    f"(Instead returned {len(items_returned)} items)"
                )

            # Unpack results
            for item in items_returned:
                repo_details = item["repository"]
                repo_name = repo_details["name"]
                owner_details = repo_details["owner"]
                owner_name = owner_details["login"]
                results.append(
                    {
                        "repo_owner": owner_name,
                        "repo_name": repo_name,
                        "repo_link": f"https://github.com/{owner_name}/{repo_name}",
                        "found_from_query": query,
                    }
                )

            # Store partial results always
            pd.DataFrame(results).to_csv("gh-results.csv")

            # Wait to avoid rate limiting
            print("Sleeping for one minute...")
            time.sleep(60)

            # Increase page and keep going
            page += 1

            # Break because we are done
            if len(items_returned) == 0:
                break

        except Exception as e:
            print(f"Caught exception: {e}")
            pass

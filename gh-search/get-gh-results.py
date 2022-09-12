import time
from ghapi.all import GhApi

SEARCH_TERMS = [
    '"National Science Foundation"',
    '"NSF Award"',
    '"NSF Grant"',
    '"Supported by the NSF"',
    '"Supported by NSF"',
]

TEMP_STORAGE_FILE = "gh-page-results.json"

# Load GH API
api = GhApi(token="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")

# Get all results for each term
results = []
for search_term in SEARCH_TERMS:
    search_query = f"{search_term} filename:README.md"
    command_template = (
        "gh api -X /search/code "
        "-f q='{search_query}' "
        "-f per_page=100 "
        "-f page={page} "
        "> {storage_file}"
    )

    # Get initial
    page = 0
    all_gathered = False
    while not all_gathered:
        query_q = f'{search_term} filename:README.md'
        print(query_q)
        page = api("/search/code", "GET ", query=dict(q='"National Science Foundation" filename:README.md', per_page=100, page=5))
        print(page["total_count"])
        print(len(page["items"]))

        # check that results are present
        # add columns to results

        # Wait to avoid rate limiting
        time.sleep(10)

        if page == 4:
            all_gathered = True

        page += 1
        
        

    
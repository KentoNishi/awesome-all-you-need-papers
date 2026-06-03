# %%

import time
import arxiv
import latexcodec
from tomark import Tomark
from additional_papers import *

# %%


def d(s):
    return s.encode("latex").decode("utf8")


def fetch_arxiv_results(search):
    backoff_seconds = (60, 180, 300)

    for attempt in range(1, len(backoff_seconds) + 2):
        try:
            client = arxiv.Client(delay_seconds=10.0, num_retries=5)
            return list(client.results(search))
        except arxiv.HTTPError as err:
            if err.status != 429 or attempt > len(backoff_seconds):
                raise

            delay = backoff_seconds[attempt - 1]
            print(
                f"arXiv returned HTTP 429; retrying in {delay}s "
                f"(attempt {attempt}/{len(backoff_seconds) + 1})"
            )
            time.sleep(delay)


# %%

search = arxiv.Search(
    query='ti:"all you need"',
    max_results=None,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Ascending,
)

table = []

for result in fetch_arxiv_results(search):
    etal = " et al." if len(result.authors) > 1 else ""
    if any([s.startswith("cs") for s in result.categories]):
        table.append(
            {
                "Title": f"[{d(result.title)}]({result.entry_id})",
                "Authors": f"{result.authors[0].name}{etal}",
                "Date": str(result.published).split(" ")[0],
            }
        )

for paper in additional_papers:
    table.append(
        {
            "Title": f"[{paper[0]}]({paper[1]})",
            "Authors": paper[2],
            "Date": paper[3],
        }
    )

table = sorted(table, key=lambda x: x["Date"])

# %%

with open("template.md", "r", encoding="utf8") as file:
    data = file.read().replace("[PAPER LIST]", Tomark.table(table))
with open("readme.md", "w", encoding="utf8") as f:
    f.writelines(
        data.replace(
            "DATEHERE",
            table[-1]["Date"].replace("-", "--")
            # str(datetime.today()).split(" ")[0].replace("-", "--")
        )
    )

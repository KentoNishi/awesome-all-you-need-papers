# %%

import arxiv
import latexcodec
from tomark import Tomark

# %%


def d(s):
    return s.encode("ascii").decode("latex")


# %%

search = arxiv.Search(
    query='ti:"is all you need"',
    max_results=float("inf"),
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Ascending,
)

table = []

for result in search.get():
    etal = " et al." if len(result.authors) > 1 else ""
    table.append(
        {
            "Title": f"[{d(result.title)}]({result.entry_id})",
            "Authors": f"{result.authors[0].name}{etal}",
            "Date": str(result.published).split(" ")[0],
        }
    )

# %%

with open("template.md", "r", encoding="utf8") as file:
    data = file.read().replace("[PAPER LIST]", Tomark.table(table))
with open("readme.md", "w", encoding="utf8") as file:
    file.writelines(data)

# %%

import arxiv
import latexcodec

# %%


def d(s):
    return s.encode("ascii").decode("latex")


# %%

search = arxiv.Search(
    query='ti:"is all you need"',
    max_results=float('inf'),
    sort_by=arxiv.SortCriterion.SubmittedDate,
)

for result in search.get():
    print(d(result.title), result)

# %%

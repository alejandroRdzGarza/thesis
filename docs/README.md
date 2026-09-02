# Project page

Static site for *Distilling Runtime Safety into VLA Robot Policies*. No build step —
plain HTML/CSS/JS, served as-is.

```
docs/
  index.html      the page
  style.css       all styling
  script.js       demo grid: filtering, lazy loading, autoplay on scroll
  data.js         manifest of the 68 clips (suite, level, task, init state, instruction)
  assets/         thesis PDF, UCL crest, the results figure, and the stills the
                  system diagram embeds (dg_*.png, cut from a demo clip)
  videos/         68 side-by-side comparison clips, ~12 MB total
  .nojekyll       stops GitHub Pages running the files through Jekyll
```

## Publishing

Settings → Pages → *Deploy from a branch* → branch `main`, folder `/docs`. The site
appears at `https://alejandroRdzGarza.github.io/thesis/` a minute or so later.

`.gitignore` ignores `videos/` and `*.mp4` project-wide, so it carries a matching pair
of negations for `docs/videos/` — keep them if you reorganise it.

## Regenerating content

The clips come from the paired-rollout traces and are rebuilt with:

```
PYTHONPATH=. python -m experiments.make_demo_videos --all   # -> figures/demo_videos/
cp figures/demo_videos/*.mp4 docs/videos/
```

`--all` selects every held-out episode where the base policy collided and the distilled
policy did not, which is why the page says outright that these are the difference cases
rather than a random sample. If the set changes, regenerate `data.js` — it is just a
JSON array parsed out of the filenames (`safelibero_<suite>_<level>_t<task>_g<init>.mp4`)
with the task instructions from `libero_suite_task_map.py`.

Figures come from `figures/`; re-copy them after rerunning any `experiments/make_fig_*.py`.

## Numbers

Every number on the page is quoted from the submitted dissertation, `assets/thesis.pdf` —
the internalisation table (§4.4), the demonstration-source comparison (§4.6) and the
per-body attribution (§4.7). Change them there first.

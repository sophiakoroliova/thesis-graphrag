# Data

Nothing in this folder (except this file) is committed: the textbook is copyrighted.

```
data/
  raw/Lay.pdf                     full textbook (Lay, Lay, McDonald, Linear Algebra and Its Applications, 5th ed., 2016)
  pdf/lay_ch1-5_theory.pdf        corpus, 238 pages             <- src/cut_pdf.py
  pdf/chapters/lay_ch1..5.pdf     chapters for parsing          <- src/cut_pdf.py
  parsed/hybrid_full/lay_chN/     MinerU output                 <- colab/parse_mineru.ipynb
```

To reproduce, place your own copy of the textbook at `data/raw/Lay.pdf` and follow the steps in the main README.
Corpus pages are defined in `src/config.py` (`CHAPTER_RANGES`).

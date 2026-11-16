# AI Generated Text in the SEC Corpus

This is a follow-up article to [Using Modal & Open Pangram to detect AI generated text across 300bn+ tokens in the SEC Corpus for $15](https://datamule.xyz/articles/detecting-ai-generated-text-across-the-entire-sec-corpus/).

## Data

Scored using an 14M Electra model trained on [Open Pangram's](https://www.pangram.com/blog/introducing-open-pangram) Dataset. Model processed 330 billion tokens of text for $15 using [Modal](https://modal.com/) free credits applied to RTX Pro 6000. SEC text was downloaded using [datamule](https://datamule.xyz/articles/sec-standardized-text/).

filings.parquet

| Column | Type | Meaning |
|---|---|---|
| accession | uint64 | original SEC filing identifier |
| ciks | list of uint32 | company identifiers associated with filing |
| filingdate | date | filing date |
| sic | string | industry code |
| submission_type | string | filing submission type |

ai_score_filings.parquet and no_ai_score_{year}.parquet

| Column | Type | Meaning |
|---|---|---|
| sequence | uint16 | sequence of document within filing |
| filename | string | filename, useful for URL construction |
| document_type | string | document type within a submission |
| num_fragments | uint32 | number of text fragments used for scoring within a document |
| bin95 | uint16 | fragment count with 0.95 <= score < 0.96 |
| bin96 | uint16 | fragment count with 0.96 <= score < 0.97 |
| bin97 | uint16 | fragment count with 0.97 <= score < 0.98 |
| bin98 | uint16 | fragment count with 0.98 <= score < 0.99 |
| bin99 | uint16 | fragment count with 0.99 <= score <= 1.00 |
| mean_score | uint16 | mean score rounded to three decimal places and multiplied by 1000 |

## Results

Scores are accurate enough to show general trends. They are not correct, and are often wrong.

Sanity check: AI score increases as LLMs are adopted.

![](plots\ai_generated_score_over_time.png)

Companies that have the most text scored as AI.

![](plots\companies.png)

Companies that have the most text scored as AI, filtered on having filings from 2022-2026.

![](plots\companies_all_years.png)

Software industry codes have more AI adoption.

![](plots\industries.png)

Industry groups show some adoption, but hard to see due to noise of scorer.

![](plots\industry_groups.png)

Crowdfunding has the most AI adoption followed by public offerings for small amounts of money (mini IPO). F-1 (foreign) IPO filings have more AI adoption than domestic (S-1). 

Annual Report to Shareholders has some AI adoption as well. It is interesting that ARS filings have more AI adoption than 10-K (Annual Report) filings.

![](plots\submission_types.png)

Ethics code documents have the most AI adoption. The other documents observe high noise with EX-19.1. This is mostly due to a noisy scorer over few EX-19.1 filings.

> Thank you to [Matt Robinson](https://www.linkedin.com/in/robinsonmatt/) of [AI Street](https://www.ai-street.co/) for the law firm idea. 

![](plots\document_types.png)

By rudimentarily linking the law firm associated with a filing, we can naively estimate what submission types firms use AI the most on. This appears to be S-1 and F-1, by far.


| Rank | Law firm | Approx. flagged-client count, 2025–26 |
|---:|---|---:|
| 1 | Ogier | 64 |
| 2 | Ellenoff Grossman & Schole | 54 |
| 3 | Loeb & Loeb | 43 |
| 4 | Maples and Calder | 40 |
| 5 | Appleby | 24 |
| 6 | Greenberg Traurig | 21 |
| 7 | Harney Westwood & Riegels | 21 |
| 8 | Sichenzia Ross Ference Carmel | 18 |
| 9 | Winston & Strawn | 13 |
| 10 | Conyers Dill & Pearman | 12 |
| 11 | Latham & Watkins | 12 |
| 12 | Lucosky Brookman | 11 |

![](plots\law_firms_by_submission_type.png)

Ethics code, S-1, and F-1 are the document types with most AI adoption.

![](plots\law_firms_by_document_type.png)


## License

Our contributions to this `sec_ai_text.parquet` are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) due to training a model on [Pangram Labs’ EditLens dataset](https://huggingface.co/datasets/pangram/editlens_iclr) which has that license.

## Acknowledgements
Thank you to:

- Katherine Thai, Bradley Emi, Elyas Masrour and Mohit Iyyer for their work on Open Pangram. Also Ben Glickenhaus, Max Spero and Pangram Labs for making a nice commercial AI generated detection product.
- Daniel Shaar at Modal. Also Modal in general for making a great product with a generous free tier.
- Kevin Clark and Minh-Thang Luong and Quoc V. Le and Christopher D. Manning for Electra
- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov for RoBERTa.

# Datasheet for RivalBench

Following *Datasheets for Datasets* (Gebru et al., 2021).

## Motivation

- **Purpose.** To evaluate citation predictors for newly published papers under a realistic, leakage-controlled protocol, and to make the
  competition among contemporaneous papers explicit: every paper comes with its own set of rivals and the papers that share its demand.
  Existing benchmarks score papers in isolation and rarely report within-subtopic ranking quality.
- **Creators and funding.** Withheld for anonymous review.

## Composition

- **Instances.** Research papers first made public in 2017–2021 in three OpenAlex subfields: Artificial Intelligence (313,260 papers),
  Oncology (155,569), and Applied Mathematics (72,259), restricted to journal, conference, book-series, and arXiv publications that pass
  de-duplication and junk filters (see `code/code/` for the pipeline).
- **Per instance.** OpenAlex work ID; publication year and date precision; topic and subtopic assignments; features available at the end
  of the publication year (for AI including the quality of cited references, which the other fields lack); citation counts in each of the
  three following years; train/validation/test assignments for two temporal protocols.
- **Relations.** For each paper, its 50 most similar papers from the same and the two previous years that were visible at prediction time,
  with their similarity, timing, citations by prediction time, venue type, shared authorship, bibliographic coupling, and mutual citation;
  near-identical twin pairs; SPECTER2 embeddings of all focal papers and rivals.
- **Labels.** Citation counts are computed from the citation links of the same OpenAlex snapshot (all citing works) and, for AI, also
  within the corpus.
- **Missing data.** Dates known only to the year have no lead-day values (`lead_days`, `c_pb365_95` missing); author histories are missing
  for authors without earlier papers; reference-quality features exist only for AI.
- **Confidentiality / sensitive data.** The data contain bibliographic metadata of published works only. No titles, abstracts, author
  names, or personal identifiers are redistributed; author-level information enters only as aggregate track-record statistics.
- **Known errors and artifacts.** OpenAlex dates, venues, and topics contain errors (for example, January-1 placeholder dates).
  Because the corpus starts in 2016, counts that look back one or two years are truncated for 2017–2018 papers. Citation counts of 2021 papers
  are affected by a corpus-wide drop in citations from calendar year 2022 onward.

## Collection process

- **Source.** A public OpenAlex snapshot (CC0), processed with DuckDB; SPECTER2 embeddings computed from titles and abstracts.
- **Time frame.** Papers 2017–2021; citations observed through 2024.
- **Pre-registration.** The evidence tests of the paper and the prediction protocol (splits, metrics) were fixed before the test year
  was evaluated.

## Preprocessing

- Duplicate works (e.g., preprint and published versions) merged; papers assigned to the venue known at prediction time.
- Subtopics: k-means (K = 2000) on SPECTER2 embeddings of papers up to 2019 (model input); a second clustering on all papers is used only
  for evaluation.
- Rivals: cosine similarity of SPECTER2 embeddings among papers of the field from years *Y*−2 to *Y*, restricted to papers visible at the
  end of year *Y*.

## Uses

- **Intended.** Benchmarking cold-start citation prediction and within-subtopic ranking of new papers; studying competition and
  visibility among contemporaneous papers; developing paper recommendation and discovery methods.
- **Not intended.** Evaluating individual researchers, hiring, funding, or promotion decisions. Citation predictions can reinforce
  cumulative advantage and should be used to surface promising work early, not to judge people.
- **Fair use of the splits.** Use the validation year for model selection and evaluate the test year once; report within-subtopic
  metrics alongside absolute errors.

## Distribution and maintenance

- **Access.** Anonymous link during review; a permanent archive with a DOI upon publication.
- **Licenses.** Data: CC BY 4.0 (derived from CC0 OpenAlex data). Code: MIT. SPECTER2: Apache-2.0. *(To be confirmed before public release.)*
- **Maintenance.** Issues and corrections through the repository; versioned releases with checksums (`MD5SUMS`).

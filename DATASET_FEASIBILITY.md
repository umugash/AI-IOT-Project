# Dataset research and feasibility

Research snapshot: 2026-11-01

> **Scope update:** the assessment below documents the historical Uganda prototype and is not the current geographic selection. The India project now has a separate local historical intelligence pipeline with Chennai station-level behaviour, chronological backtests, virtual-event replay, and SQLite persistence; see [FINAL_PROJECT_SCOPE.md](./FINAL_PROJECT_SCOPE.md), [FORECASTING_BASELINE_REPORT.md](./FORECASTING_BASELINE_REPORT.md), and [ARCHITECTURE.md](./ARCHITECTURE.md). These are not current forecasts or geographic maps. Uganda measurements, maps, and metrics remain supplementary and must not be presented as Indian results.

## Assessment

**Urban Noise Uganda 61K was selected as the primary dataset for the earlier prototype.** It is not the current geographic selection. The official Figshare download is open under CC BY 4.0 and contains OGG audio, source labels, GPS coordinates, and a noise measurement field. Direct inspection of the public archive confirms the audio and spatial/label metadata, but also reveals material discrepancies that affected the original time-series work:

- The Figshare landing description claims 61,821 samples, but its actual `metadata.csv` contains 61,685 rows and the archive contains 61,685 OGG recordings.
- The publicly downloadable Figshare `metadata.csv` has **no timestamp field**. It has 61,685 rows, 60,581 exact coordinate pairs, and only 1,622 observations among the 518 coordinate pairs that repeat.
- The Hugging Face dataset card lists a `timestamp` feature, but the repository is gated and requires agreeing to share account information. Anonymous dataset-server requests return 401, so timestamp values and coverage could not be independently inspected. Its card also describes 59,731 large-config examples plus a 1,000-row small config, which does not reconcile with the Figshare count.

The public Figshare release therefore supports acoustic classification and spatial noise mapping. **Reliable emergence/persistence/disappearance analysis and future forecasting are not supported by the verified public files**, because observation times are absent. Do not infer timestamps from archive order, file names, or the Hugging Face card's schema. This project will preserve the metadata as delivered and will refuse temporal replay/forecasting until a lawful, accessible source of genuine timestamps is verified.

The `noise_measurement` field is described by the dataset record/card as SPL, but the public CSV has no unit/calibration field and observed values range from 1 to 84 (median 48). Treat it as the dataset-reported measurement; document the provenance and do not claim independent calibration or regulatory compliance without supporting methodology.

## Primary candidate: Urban Noise Uganda 61K

| Property | Verified finding |
|---|---|
| Source | [Figshare dataset record](https://doi.org/10.6084/m9.figshare.30168901.v1), [Hugging Face dataset card](https://huggingface.co/datasets/Sunbird/urban-noise-uganda-61k), and [Scientific Data article](https://doi.org/10.1038/s41597-026-06658-w) |
| Acquisition | Mobile-based field recordings in Kampala and Entebbe, Uganda. Public archive includes original OGG audio files. |
| Actual public archive | One open Figshare ZIP, 2,971,638,528 bytes (about 2.77 GiB), CC BY 4.0. Its ZIP64 central directory contains 61,685 OGG recordings plus macOS sidecar files and `metadata.csv`. The metadata CSV is 5,626,799 bytes uncompressed. |
| Actual metadata | 61,685 rows; fields are `filename`, `class`, `class_id`, `noise_measurement`, `latitude`, `longitude`, `altitude`, `accuracy`, `submitter_id`, and `region`. There is no timestamp/date/time field in this public CSV. |
| Geography | 37,512 Kampala and 24,173 Entebbe records. 60,581 exact coordinate pairs; 518 pairs repeat, covering 1,622 samples (maximum 91 at one pair). GPS accuracy is supplied, so any grid aggregation should account for it. |
| Labels | 19 observed categories including `other`, `boda-boda-motorcycle`, `car-or-truck`, `crowd-noise`, `animal`, construction, sirens, vendors, and venue/activity categories. The Hugging Face card says Kampala labels are hand-labelled while Entebbe labels are model-predicted; this distinction must be respected and independently verified before using Entebbe for ground-truth model evaluation. |
| SPL field | The dataset describes `noise_measurement` as SPL in decibels. In the actual CSV values range from 1 to 84, median 48. The CSV itself provides no unit, weighting, or calibration column; use it as dataset-reported SPL and report this limitation. |
| Time | Figshare's public CSV has no timestamp. The Hugging Face card describes an ISO 8601 timestamp field, but that dataset is gated behind an agreement/login; the anonymous rows endpoint returns 401. Timestamp coverage, row alignment, and permission to access were therefore not verified. |
| Count discrepancies | The Figshare description says 61,821 while its downloadable metadata/archive give 61,685. Hugging Face metadata reports 59,731 rows for `large` and 1,000 for `small`. Use the inspected Figshare row/file counts for the public-download pipeline and document these differences. |
| Hotspot suitability | Strong for retrospective geographic event-category and reported-intensity maps, subject to location coverage, class-label provenance, sampling bias, and measurement caveats. Not sufficient for temporal hotspot behaviour or forecasting unless real timestamps are obtained and validated. |

### Download inspection performed

The Figshare API verified public status, CC BY 4.0 licensing, file size, and MD5. HTTP byte-range reads first inspected the ZIP64 directory and extracted only the 5.6 MB metadata member. The full 2.97 GB archive was subsequently downloaded and verified against the published MD5, then the metadata and OGG recordings were extracted. The acquisition script confirmed 61,685 metadata rows, 61,685 OGG files, and matching filenames. The actual metadata lacks timestamps, so timestamp-based forecasting remains unsupported.

## Optional supplement: SONYC-UST v2.3

| Property | Verified finding |
|---|---|
| Source | [Zenodo v2.3 record](https://doi.org/10.5281/zenodo.3966543), [SONYC resources](https://wp.nyu.edu/sonyc/resources/), and the [DCASE 2020 task](https://dcase.community/challenge2020/task-urban-sound-tagging-with-spatiotemporal-context) |
| Acquisition | Recordings from the SONYC urban acoustic sensor network in New York City; identical microphones and gain settings are reported. This is sensor-recorded audio, not crowdsourced phone audio. |
| Recordings | The downloaded v2.3 `annotations.csv` contains **18,510 unique audio filenames**: 13,538 train, 4,308 validation, and 664 test recordings. |
| Annotation table | 62,022 annotation rows (multiple annotators per recording); 1,380 rows use annotator ID `0` for the agreed SONYC ground-truth annotations. The file has 37 presence columns: 29 fine-level taxonomy entries (including six “unknown/other” buckets) plus eight coarse-level categories. |
| Audio | Ten-second audio files distributed across 19 `audio-*.tar.gz` files. The archive files total approximately **12.37 GiB** (about 13.28 GB); `annotations.csv` is 14,488,305 bytes. The data are accessible without credentials, but a full local copy is sizeable. |
| Time/location | The CSV includes year, week of year, day of week, hour, borough, block, and block-level latitude/longitude. In the inspected metadata there are 56 sensor IDs and 37 borough/block pairs. Time is quantized to an hour and location to a city block for privacy. |
| Labels | Multi-label presence and proximity annotations for urban sound sources, including engine, machinery impact, powered saw, alert signal, music, human voice, and dog, at fine and coarse levels. Labels are not direct SPL measurements. |
| Selection and split caveats | The release documentation says recordings were selected by nearest-neighbour audio features around recordings known to have classes of interest. Train/validation sensors are disjoint; the test data are later in time. This supports a temporal test but does not remove selection bias. |
| Licensing | Zenodo reports open access and **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Cite the dataset DOI/version and the dataset paper when using it. |
| Hotspot suitability | Optional supplementary data for multilabel acoustic tagging and NYC sensor-context comparison. Not aligned geographically with Uganda and must not be merged as if observations came from the same city. |

### Download inspection performed

The official Zenodo record API was checked for file names, byte sizes, access status, and license. The 14.5 MB `annotations.csv` was downloaded and parsed locally to verify the recording counts, split names, fields, label columns, time/location coverage, and annotation-row counts above. The audio archives were verified in the official file listing, but **not** downloaded: all audio chunks together are approximately 12.37 GiB. Consequently, this inspection does not yet verify individual audio headers or train an audio model.

The record landing page/summary says 18,515 recordings and 1,385 verified recordings; the actual v2.3 CSV inspected here has 18,510 unique filenames and 1,380 annotator-ID-0 rows. The CSV counts are used in this assessment; this discrepancy should be recorded and checked against any later mirror or release before reporting results.

## Other candidates and supplements

| Dataset | Contents and accessibility | Spatial/temporal/audio evidence | Suitability |
|---|---|---|---|
| [SONYC-UST v1.0](https://doi.org/10.5281/zenodo.3692954) | Earlier release; official source describes 2,794 ten-second recordings and a 23-source multilabel taxonomy. | Audio and tags, but the v1.0 documentation says time and location are omitted/distributed for privacy. | Useful for acoustic-tagging experiments, not hotspot analysis. Prefer v2.3. |
| [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) | 8,732 WAV excerpts of up to four seconds, ten urban classes, and a metadata CSV; obtainable via the dataset download form or `soundata`. Audio sample rates and formats follow their source recordings. The dataset site asks academic users to acknowledge/cite it; Freesound source licenses can vary. | No observation GPS or recording timestamp. Its CSV identifies source Freesound IDs, excerpt start/end, class, salience, and predefined folds. | Good optional classification benchmark only. It cannot support geographic hotspot discovery or forecasting. Keep its recommended predefined 10-fold evaluation if used. |
| [NoiseCapture](https://noise-planet.org/noisecapture.html) | Android app measures dB(A), noise indicators, and GPS tracks; users are encouraged to calibrate their phones. A community map is public. A verified, bulk, long-duration per-measure export suitable for this project was not established in this research. | Potential georeferenced noise-level measurements, but crowdsourced phone hardware/calibration and uneven survey coverage complicate comparisons. It does not provide SONYC-style audio/tag labels in the reviewed materials. | Promising for a separate, carefully qualified noise-level study if a suitable export and its license can be verified; not selected as the core dataset now. |
| [URBANOME personal-exposure release](https://zenodo.org/records/16964643) | Open 1.7 MB Zenodo ZIP with daily aggregate personal exposure fields, including NoiseCapture-derived `L_mean`, `L_den`, and `L_night`; its description covers campaign phases across cities. | Daily participant aggregates rather than a block-level acoustic event stream; no reviewed audio/source labels or coordinates suitable for a hotspot map. | An accessible dB-related supplement, but unsuitable for this project's source classification and spatial hotspot forecasting. Its measures are not evidence of calibrated fixed-sensor SPL. |
| [NYC 311 service requests](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9) | Public, queryable complaint records; an API record was confirmed to expose complaint timestamp, type/descriptor, and coordinates. | These are human reports, not acoustic observations or sound-level measurements. | Optional contextual comparison only. Do not mix complaints into acoustic labels or call them measured noise. |

## Recommended research scope

Use the public Figshare Uganda release as the primary dataset. Implement multiclass source classification on the Kampala human-labelled subset, with Entebbe kept for exploratory geographic analysis unless the original annotation provenance can be established. Create spatial cells and noise-intensity summaries from the reported measurement field, retain observation counts and GPS accuracy, and clearly label the resulting maps as sampled dataset observations rather than unbiased area-wide exposure.

Do not implement or report temporal forecasting using the public Figshare CSV: its timestamp is absent. Historical replay must fail clearly rather than order by file/ZIP order or invent times. The gated Hugging Face card advertises timestamps, but the actual values and their coverage remain unverified; if a lawful, authorized download is later supplied, first validate actual timestamps, repeated locations, and chronological train/validation/test coverage. Only then should time-based behaviour labels or forecast models be enabled. Without that evidence, focus the scientific contribution on classification and retrospective spatial hotspot/intensity analysis, as requested.

For SONYC-UST v2.3, keep its multilabel semantics and official splits if used as a supplementary, separate-domain classifier benchmark. Do not combine it with Uganda's class labels, measurements, or locations as if the taxonomies and collection methods were equivalent.

## Syllabus-aligned feasibility

| Project component | Syllabus concept | Evidence-based implementation direction |
|---|---|---|
| Metadata and audio ingest | IoT collection design, CSV, JSON, HDF5 | Parse Uganda metadata CSV and OGG audio; retain original values and metadata provenance. |
| Sensor replay | Edge devices, IoT ingestion, windowing | Implement a simulated historical event envelope; timestamp-ordered replay is unavailable and must be guarded until timestamps are actually obtained. It is not live sensor acquisition. |
| Acoustic classification | Deep learning, CNN, feature engineering, model comparison, PyTorch/scikit-learn | Compare a multiclass Uganda log-Mel CNN with an MFCC-summary Random Forest baseline, using verified human-labelled samples and geographic-group evaluation to reduce location leakage. |
| Hotspot behaviour | Data science, XGBoost/LSTM | Retrospective spatial intensity and source-category aggregation is supported. Temporal emergence/persistence and XGBoost/LSTM forecasting are blocked by missing public timestamps. |
| Storage/streaming | IoT Hub, Kafka, Spark, Delta Lake, NoSQL | Keep local execution the default; make Kafka/Spark/Delta optional scale-out adapters. Do not claim a timestamp-ordered stream or require a cloud subscription. |
| Graph modelling | GraphFrames/GraphX | Defer unless real repeated regional observations support a stable graph and it improves over non-graph baselines. |
| Edge deployment | Edge AI | Assess model size and CPU latency after a working classifier exists; no physical hardware or cloud service is assumed. |

## References

1. Cartwright et al., “SONYC-UST-V2: An Urban Sound Tagging Dataset with Spatiotemporal Context,” DCASE 2020; [paper](https://arxiv.org/abs/2009.05188), [dataset DOI](https://doi.org/10.5281/zenodo.3966543).
2. [SONYC-UST v2.3 Zenodo record and README](https://zenodo.org/records/3966543).
3. [DCASE 2020 Task 5 description and results](https://dcase.community/challenge2020/task-urban-sound-tagging-with-spatiotemporal-context).
4. [SONYC DCASE baseline code and setup](https://github.com/sonyc-project/dcase2020task5-uststc-baseline).
5. Salamon, Jacoby, and Bello, “A Dataset and Taxonomy for Urban Sound Research,” ACM Multimedia 2014; [UrbanSound8K dataset page](https://urbansounddataset.weebly.com/urbansound8k.html).
6. [NoiseCapture project and protocol](https://noise-planet.org/noisecapture.html), [URBANOME open data record](https://zenodo.org/records/16964643), and [NYC 311 Open Data](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9).
7. Nsumba et al., “Noise mapping and ambient sound recordings of the urban environment in Uganda,” *Scientific Data* 13, 345 (2026), [article DOI](https://doi.org/10.1038/s41597-026-06658-w), [Figshare data DOI](https://doi.org/10.6084/m9.figshare.30168901.v1), and [Hugging Face dataset card](https://huggingface.co/datasets/Sunbird/urban-noise-uganda-61k).

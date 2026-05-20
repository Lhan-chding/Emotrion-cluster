# Affect-aware Retrieval Evaluation

## Task Definition

This downstream module evaluates whether fixed Dataset-S v20.3 VA+tension representations retrieve songs with complex affective structure better than a VA-only baseline. It does not retrain the upstream VA model, rerun clustering, change cluster assignments, change tension subtype assignments, or alter the post-hoc v3 interpretation rules.

## Query Definitions

| query_id | description | target_cluster | target_tension_strength | target_complexity |
|---|---|---|---|---|
| volatile_high_tension | low valence + high arousal + high cross-modal tension | C2 | high | complex_or_high |
| bittersweet_lyrical_lift | subdued melancholy with lyric-side uplift or intensification | C0 | moderate_or_high | mild_or_complex |
| gentle_warmth_lyrical_lift | gentle warmth with lyric-side warmth or emotional lift | C1 | moderate_or_high |  |
| audio_led_exuberance_dark_lyrics | positive energetic song with darker or softer lyric-side undercurrent | C3 | moderate_or_high | complex_or_high |
| boundary_or_ambivalent_affect | boundary blend or emotionally ambivalent song | any |  | high |
| concordant_region_prototype | clean affective prototype with low tension | any | low | simple |

## Systems Compared

- System A (`va_only`): balanced valence/arousal only, scored by an exponential normalized VA distance.
- System B (`va_cluster`): VA similarity plus target region soft weights, region typicality, and geometry role matching.
- System C (`va_tension_complexity`): fixed VA, region soft weights, calibrated audio-lyric tension direction/strength, affective complexity score, and geometry role matching.

## Scoring Formula

`score_va = exp(-0.5 * normalized_va_distance^2)`.

`score_cluster = 0.50*score_va + 0.30*region_match + 0.10*region_typicality_match + 0.10*boundary_or_mixture_match`.

`score_full = 0.30*score_va + 0.20*region_match + 0.25*tension_match + 0.15*complexity_match + 0.10*geometry_role_match`.

All weights are fixed a priori. External reviews and relevance labels are not used in ranking.

## External Annotation Protocol

External professional reviews are used only after retrieval outputs are fixed. Accepted sources are professional music criticism or mainstream media music reviews. Lyrics websites, Genius annotations, Reddit, forums, user comments, Spotify tags, YouTube comments, Amazon reviews, unattributed snippets, and generated content are not accepted as primary evidence. Evidence summaries are paraphrases and do not quote lyrics.

Grades: region relevance and tension relevance are each 0-3. For simple prototype queries, overall relevance equals region relevance. For tension or complex queries, overall relevance is rounded from `0.45*region + 0.55*tension`.

## Metric Table

No verified external relevance labels are available yet, so retrieval metrics are gated off.

## Summary

No cross-query summary is available until at least one external label is verified.

## Per-query Qualitative Examples

| query_id | title | artist | system_sources | cluster_name | tension_strength_percentile | affective_complexity_score | annotation_status |
|---|---|---|---|---|---|---|---|
| audio_led_exuberance_dark_lyrics | Bright Dark | Artist E | va_cluster;va_only;va_tension_complexity | Playful Vitality | 1.00 | 0.82 | needs_external_review |
| audio_led_exuberance_dark_lyrics | Prototype | Artist F | va_cluster;va_only | Gentle Warmth | 1.00 | 0.20 | needs_external_review |
| audio_led_exuberance_dark_lyrics | High Volatile | Artist A | va_tension_complexity | Volatile Intensity | 1.00 | 0.91 | needs_external_review |
| audio_led_exuberance_dark_lyrics | Warm Song | Artist D | va_cluster;va_only;va_tension_complexity | Gentle Warmth | 1.00 | 0.20 | needs_external_review |
| bittersweet_lyrical_lift | Lifted Melancholy | Artist C | va_cluster;va_only;va_tension_complexity | Subdued Melancholy | 1.00 | 0.65 | needs_external_review |
| bittersweet_lyrical_lift | Prototype | Artist F | va_cluster;va_only | Gentle Warmth | 1.00 | 0.20 | needs_external_review |
| bittersweet_lyrical_lift | High Volatile | Artist A | va_tension_complexity | Volatile Intensity | 1.00 | 0.91 | needs_external_review |
| bittersweet_lyrical_lift | Warm Song | Artist D | va_cluster;va_only;va_tension_complexity | Gentle Warmth | 1.00 | 0.20 | needs_external_review |
| boundary_or_ambivalent_affect | Bright Dark | Artist E | va_cluster;va_only;va_tension_complexity | Playful Vitality | 1.00 | 0.82 | needs_external_review |
| boundary_or_ambivalent_affect | Lifted Melancholy | Artist C | va_cluster;va_only;va_tension_complexity | Subdued Melancholy | 1.00 | 0.65 | needs_external_review |
| boundary_or_ambivalent_affect | Prototype | Artist F | va_cluster;va_only | Gentle Warmth | 1.00 | 0.20 | needs_external_review |
| boundary_or_ambivalent_affect | High Volatile | Artist A | va_tension_complexity | Volatile Intensity | 1.00 | 0.91 | needs_external_review |
| concordant_region_prototype | Bright Dark | Artist E | va_cluster;va_only | Playful Vitality | 1.00 | 0.82 | needs_external_review |
| concordant_region_prototype | Lifted Melancholy | Artist C | va_cluster;va_only | Subdued Melancholy | 1.00 | 0.65 | needs_external_review |
| concordant_region_prototype | Prototype | Artist F | va_cluster;va_only;va_tension_complexity | Gentle Warmth | 1.00 | 0.20 | needs_external_review |
| concordant_region_prototype | Low Volatile | Artist B | va_tension_complexity | Volatile Intensity | 0.50 | 0.20 | needs_external_review |
| concordant_region_prototype | Warm Song | Artist D | va_tension_complexity | Gentle Warmth | 1.00 | 0.20 | needs_external_review |
| gentle_warmth_lyrical_lift | Lifted Melancholy | Artist C | va_cluster;va_only;va_tension_complexity | Subdued Melancholy | 1.00 | 0.65 | needs_external_review |

## Limitations

- This is not a final user satisfaction evaluation for a music recommender.
- Metrics depend on post-hoc professional review coverage; unverified rows are excluded from judged-label metrics.
- External evidence is not allowed to tune scoring weights, construct query-specific rankings, or revise model-side outputs.
- The experiment evaluates retrieval-oriented downstream validity of a fixed representation, not causal preference or listening behavior.

## Sanity Check

```json
{
  "total_songs": 6,
  "missing_feature_counts": {
    "title": 0,
    "artist": 0,
    "cluster_id": 0,
    "cluster_name": 0,
    "balanced_valence": 0,
    "balanced_arousal": 0,
    "region_typicality": 0,
    "region_confidence": 0,
    "region_margin": 0,
    "nearest_alt_cluster": 0,
    "w_region_C0": 0,
    "w_region_C1": 0,
    "w_region_C2": 0,
    "w_region_C3": 0,
    "tension_label": 0,
    "tension_name": 0,
    "tension_dv": 0,
    "tension_da": 0,
    "tension_norm": 0,
    "tension_strength_percentile": 0,
    "affective_complexity_score": 0,
    "complexity_level": 0,
    "final_interpretation_label": 0
  },
  "cluster_conflict_counts": {},
  "tension_conflict_counts": {
    "tension_label": 4,
    "tension_name": 4,
    "tension_dv": 2,
    "tension_da": 2,
    "tension_norm": 2
  },
  "input_paths": {
    "cluster_csv": "C:\\Users\\LHan1\\Desktop\\CVCL\\cluster\\.pytest_affect_retrieval_full_env\\test_run_affect_retrieval_eval0\\dataset_s_cluster_assignments.csv",
    "tension_csv": "C:\\Users\\LHan1\\Desktop\\CVCL\\cluster\\.pytest_affect_retrieval_full_env\\test_run_affect_retrieval_eval0\\dataset_s_tension_assignments.csv",
    "interpretation_csv": "C:\\Users\\LHan1\\Desktop\\CVCL\\cluster\\.pytest_affect_retrieval_full_env\\test_run_affect_retrieval_eval0\\song_affective_interpretation_all_v3.csv"
  },
  "number_of_queries": 6,
  "systems": [
    "va_only",
    "va_cluster",
    "va_tension_complexity"
  ],
  "topK": [
    2,
    3
  ],
  "annotation_pool_size": 23,
  "verified_external_labels_count": 0,
  "unverified_count": 23,
  "contradiction_count": 0,
  "metrics_available": false,
  "whether_any_external_label_used_in_scoring": false
}
```

## Recommended LaTeX Insertion Text

\paragraph{Affect-aware retrieval.}
We evaluate the fixed Dataset-S v20.3 representation in a retrieval-oriented downstream setting. A VA-only baseline ranks songs by integrated balanced valence and arousal, which captures the song's overall affective position but cannot directly express cross-modal affective structure. The full affect-aware retrieval system additionally uses calibrated audio-lyric tension, region soft weights, boundary margin, and affective complexity. This allows queries for structures such as subdued affect with lyrical lift, volatile intensity with semantic amplification, and bright audio-led vitality with a darker lyrical undertone. External professional reviews are used only after retrieval outputs are fixed, as post-hoc critical relevance labels. We therefore interpret the results as a downstream validation of representation quality for complex-affect retrieval, not as a final evaluation of user satisfaction in a recommendation system.

\caption{Affect-aware retrieval evaluation. VA-only retrieval ranks songs by balanced affective position, while the full representation additionally uses calibrated audio-lyric tension and affective complexity. Relevance grades are assigned from external professional descriptions after retrieval outputs are fixed.}

# Song Affective Profile Report

## Method

This report is post-hoc only. It reads fixed Dataset-S v20.3 cluster assignments and fixed tension subtype assignments; it does not retrain models and does not change assignments.

## Distance Definitions

- Region prototypes are cluster means in balanced valence-arousal space.
- For each candidate region k, region distance is Euclidean distance to region prototype k divided by the median in-cluster radius of region k.
- Region soft weights use exp(-0.5 * D^2) over all main regions.
- Tension subtype prototypes are computed in region-local robust-scaled (tension_dv, tension_da) space.
- tension_norm is used only as a strength percentile, not as a subtype-distance coordinate.

## Typicality Definitions

- Region typicality is a centrality percentile within the assigned cluster: higher values mean closer to that cluster prototype.
- Tension typicality is the same centrality percentile within the assigned tension subtype.
- Tension strength percentile is the empirical percentile of tension_norm within the same main cluster.

## Selected Songs


### Region prototype songs

| song_id | title | artist | cluster | region role | tension | main text | region typicality | tension strength | top descriptors |
|---|---|---|---|---|---|---|---:|---:|---|
| A157-71 | Yesterday | The Beatles | Subdued Melancholy | prototype | C0-T0 / modality-consistent | yes | 95.1% | 13.8% | introspective, low arousal, melancholic, somber, subdued melancholy |
| A061-66 | Mad World | Gary Jules | Subdued Melancholy | prototype | C0-T1 / lyric-brightened + lyric-intensified + high cross-modal tension | yes | 98.5% | 59.3% | introspective, low arousal, melancholic, somber, subdued melancholy |
| MT0003274067 | Driving | Arab Strap | Subdued Melancholy | prototype | C0-T1 / lyric-brightened + lyric-intensified + high cross-modal tension | yes | 87.2% | 98.4% | introspective, low arousal, melancholic, somber, subdued melancholy |
| A151-172 | War Pigs | Black Sabbath | Volatile Intensity | representative | C2-T1 / lyric-intensified + high cross-modal tension | yes | 66.1% | 76.4% | aggressive, high arousal, negative-active, tense, volatile intensity |
| MT0007075838 | Sleep Now in the Fire | Rage Against the Machine | Volatile Intensity | prototype | C2-T2 / lyric-brightened + high cross-modal tension | yes | 97.2% | 92.1% | aggressive, high arousal, negative-active, tense, volatile intensity |
| A014 | Feels Just Like It Should | Jamiroquai | Playful Vitality | representative | C3-T0 / modality-consistent | yes | 69.0% | 28.6% | bright, danceable, energetic, playful vitality, positive-active |
| A019 | California Girls | The Beach Boys | Playful Vitality | representative | C3-T1 / lyric-darkened + lyric-softened + high cross-modal tension | yes | 78.9% | 61.3% | bright, danceable, energetic, playful vitality, positive-active |
| MT0006683875 | Mary's Place | Bruce Springsteen | Playful Vitality | prototype | C3-T1 / lyric-darkened + lyric-softened + high cross-modal tension | yes | 97.7% | 91.5% | bright, danceable, energetic, playful vitality, positive-active |
| MT0027072382 | Express Yourself | Madonna | Playful Vitality | prototype | C3-T0 / modality-consistent | yes | 94.8% | 19.2% | bright, danceable, energetic, playful vitality, positive-active |
| MT0004141823 | Conga | Miami Sound Machine | Playful Vitality | prototype | C3-T0 / modality-consistent | yes | 96.9% | 23.5% | bright, danceable, energetic, playful vitality, positive-active |
| MT0033512773 | Do You Believe in Love | Huey Lewis & the News | Playful Vitality | prototype | C3-T0 / modality-consistent | yes | 99.5% | 20.4% | bright, danceable, energetic, playful vitality, positive-active |
| MT0002360219 | Keepin' the Summer Alive | The Beach Boys | Gentle Warmth | prototype | C1-T1 / lyric-brightened + lyric-intensified | yes | 98.7% | 97.2% | calm-positive, gentle warmth, romantic, soft, warm |
| A085-69 | Dying in the Sun | The Cranberries | Volatile Intensity | prototype | C2-T1 / lyric-intensified + high cross-modal tension | yes | 93.3% | 65.8% | aggressive, high arousal, negative-active, tense, volatile intensity |

### Tension case-study songs

| song_id | title | artist | cluster | region role | tension | main text | region typicality | tension strength | top descriptors |
|---|---|---|---|---|---|---|---:|---:|---|
| A045 | Hurt | Johnny Cash | Subdued Melancholy | peripheral | C0-T1 / lyric-brightened + lyric-intensified + high cross-modal tension | yes | 1.3% | 89.9% | audio-lyric contrast, high cross-modal tension, lyric arousal intensification, lyric valence uplift, lyrics brighten the affect |
| A034 | Natural Disaster | Anathema | Subdued Melancholy | peripheral | C0-T1 / lyric-brightened + lyric-intensified + high cross-modal tension | yes | 12.0% | 75.9% | audio-lyric contrast, high cross-modal tension, lyric arousal intensification, lyric valence uplift, lyrics brighten the affect |
| A171 | Say Something | A Great Big World | Subdued Melancholy | prototype | C0-T1 / lyric-brightened + lyric-intensified + high cross-modal tension | yes | 98.9% | 83.4% | introspective, low arousal, melancholic, somber, subdued melancholy |
| A021 | That's What Friends Are For | Dionne Warwick | Gentle Warmth | representative | C1-T1 / lyric-brightened + lyric-intensified | yes | 52.7% | 81.7% | audio-lyric contrast, high cross-modal tension, lyric arousal intensification, lyric valence uplift, lyrics brighten the affect |
| A029 | Tears in Heaven | Eric Clapton | Gentle Warmth | prototype | C1-T1 / lyric-brightened + lyric-intensified | yes | 88.6% | 88.1% | calm-positive, gentle warmth, romantic, soft, warm |
| MT0005344437 | Send One Your Love | Stevie Wonder | Gentle Warmth | prototype | C1-T1 / lyric-brightened + lyric-intensified | yes | 99.6% | 86.4% | calm-positive, gentle warmth, romantic, soft, warm |
| A005 | U Got the Look | Prince | Playful Vitality | peripheral | C3-T1 / lyric-darkened + lyric-softened + high cross-modal tension | yes | 10.3% | 89.2% | audio-lyric contrast, high cross-modal tension, lyric arousal softening, lyric valence tempering, lyrics darken the affect |
| MT0000151451 | Love You 'Till the End | The Pogues | Gentle Warmth | prototype | C1-T1 / lyric-brightened + lyric-intensified | yes | 96.8% | 81.0% | calm-positive, gentle warmth, romantic, soft, warm |

### Boundary / ambiguity cases

| song_id | title | artist | cluster | region role | tension | main text | region typicality | tension strength | top descriptors |
|---|---|---|---|---|---|---|---:|---:|---|
| A004 | I'm So Lonesome I Could Cry | Johnny Cash | Subdued Melancholy | boundary | C0-T0 / modality-consistent | no | 5.0% | 14.1% | boundary between assigned region and nearest alternative, nearest alternative: C2 Volatile Intensity, assigned region: Subdued Melancholy, affective concordance, audio-lyric agreement |
| A048 | Everybody Hurts | R.E.M. | Subdued Melancholy | boundary | C0-T0 / modality-consistent | no | 50.2% | 1.5% | boundary between assigned region and nearest alternative, assigned region: Subdued Melancholy, nearest alternative: C1 Gentle Warmth, affective concordance, audio-lyric agreement |
| A049 | Highway to Hell | AC/DC | Volatile Intensity | boundary | C2-T1 / lyric-intensified + high cross-modal tension | no | 1.6% | 23.6% | boundary between assigned region and nearest alternative, nearest alternative: C3 Playful Vitality, assigned region: Volatile Intensity, affective concordance, audio-lyric agreement |

### Appendix-only candidates

| song_id | title | artist | cluster | region role | tension | main text | region typicality | tension strength | top descriptors |
|---|---|---|---|---|---|---|---:|---:|---|
| A024 | Paradise | Sade | Gentle Warmth | peripheral | C1-T0 / modality-consistent | no | 8.1% | 10.2% | affective concordance, audio-lyric agreement, calm-positive, gentle warmth, romantic |
| A017 | Just The Way You Are | Billy Joel | Gentle Warmth | peripheral | C1-T0 / modality-consistent | no | 2.4% | 76.8% | lyric valence tempering, lyrics darken the affect, lyric arousal intensification, lyrics intensify the affect, calm-positive |
| A013 | London Calling | The Clash | Volatile Intensity | peripheral | C2-T0 / lyric-brightened | no | 45.2% | 22.9% | aggressive, high arousal, negative-active, tense, volatile intensity |
| A010 | Animal | Pearl Jam | Volatile Intensity | peripheral | C2-T1 / lyric-intensified + high cross-modal tension | no | 2.6% | 98.0% | lyric arousal intensification, lyrics intensify the affect, audio-lyric contrast, high cross-modal tension, aggressive |
| A183 | Killing in the Name | Rage Against the Machine | Volatile Intensity | peripheral | C2-T0 / lyric-brightened | no | 49.9% | 36.5% | aggressive, high arousal, negative-active, tense, volatile intensity |
| A001 | What a Wonderful World | Louis Armstrong | Playful Vitality | peripheral | C3-T0 / modality-consistent | no | 4.9% | 39.7% | bright, danceable, energetic, playful vitality, positive-active |
| A028 | I Fucking Hate You | Godsmack | Volatile Intensity | peripheral | C2-T0 / lyric-brightened | no | 0.5% | 67.8% | lyric arousal softening, lyrics soften the affect, aggressive, high arousal, negative-active |

### Selected Song Interpretations


**A157-71 Chinese**: 该歌曲属于 Subdued Melancholy 区域，是高度原型样本；region typicality 为 95.1%，其 balanced VA 位置相对该区域原型的 soft confidence 为 83.4%。该样本与最近替代区域 C1 Gentle Warmth 明显分离（region margin=1.586）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C0-T0 / modality-consistent，tension strength percentile 为 13.8%；张力强度较低，因此不把小的音频-歌词方向差解释为主要证据。综合 descriptor profile 的高权重词包括：introspective、low arousal、melancholic、somber、subdued melancholy。

**A157-71 English**: The song is assigned to Subdued Melancholy and is highly prototypical, with region typicality 95.1% and assigned-region soft confidence 83.4%; it is clearly separated from the nearest alternative region, C1 Gentle Warmth (region margin=1.586). The calibrated cross-modal tension / audio-lyric contrast profile is C0-T0 / modality-consistent, with tension strength percentile 13.8%; tension strength is low, so small audio-lyric directional differences are not used as primary evidence. The top descriptor profile is: introspective, low arousal, melancholic, somber, subdued melancholy.


**A004 Chinese**: 该歌曲属于 Subdued Melancholy 区域，是边界样本，不应用作主区域证据；region typicality 为 5.0%，其 balanced VA 位置相对该区域原型的 soft confidence 为 47.3%。该样本在 post-hoc prototype 距离下更接近 C2 Volatile Intensity，只能作为边界/歧义案例使用（region margin=-0.051）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C0-T0 / modality-consistent，tension strength percentile 为 14.1%；张力强度较低，因此不把小的音频-歌词方向差解释为主要证据。综合 descriptor profile 的高权重词包括：boundary between assigned region and nearest alternative、nearest alternative: C2 Volatile Intensity、assigned region: Subdued Melancholy、affective concordance、audio-lyric agreement。

**A004 English**: The song is assigned to Subdued Melancholy and is a boundary case, not used as main region evidence, with region typicality 5.0% and assigned-region soft confidence 47.3%; post-hoc prototype distance places it closer to C2 Volatile Intensity; use it only as a boundary or ambiguity case (region margin=-0.051). The calibrated cross-modal tension / audio-lyric contrast profile is C0-T0 / modality-consistent, with tension strength percentile 14.1%; tension strength is low, so small audio-lyric directional differences are not used as primary evidence. The top descriptor profile is: boundary between assigned region and nearest alternative, nearest alternative: C2 Volatile Intensity, assigned region: Subdued Melancholy, affective concordance, audio-lyric agreement.


**A048 Chinese**: 该歌曲属于 Subdued Melancholy 区域，是边界样本，不应用作主区域证据；region typicality 为 50.2%，其 balanced VA 位置相对该区域原型的 soft confidence 为 58.1%。该样本与最近替代区域 C1 Gentle Warmth 有一定接近性，但仍保留正 margin（region margin=0.319）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C0-T0 / modality-consistent，tension strength percentile 为 1.5%；张力强度较低，因此不把小的音频-歌词方向差解释为主要证据。综合 descriptor profile 的高权重词包括：boundary between assigned region and nearest alternative、assigned region: Subdued Melancholy、nearest alternative: C1 Gentle Warmth、affective concordance、audio-lyric agreement。

**A048 English**: The song is assigned to Subdued Melancholy and is a boundary case, not used as main region evidence, with region typicality 50.2% and assigned-region soft confidence 58.1%; it is moderately close to the nearest alternative region, C1 Gentle Warmth, while retaining a positive margin (region margin=0.319). The calibrated cross-modal tension / audio-lyric contrast profile is C0-T0 / modality-consistent, with tension strength percentile 1.5%; tension strength is low, so small audio-lyric directional differences are not used as primary evidence. The top descriptor profile is: boundary between assigned region and nearest alternative, assigned region: Subdued Melancholy, nearest alternative: C1 Gentle Warmth, affective concordance, audio-lyric agreement.


**A045 Chinese**: 该歌曲属于 Subdued Melancholy 区域，是所属区域内的外围样本；region typicality 为 1.3%，其 balanced VA 位置相对该区域原型的 soft confidence 为 93.7%。该样本与最近替代区域 C2 Volatile Intensity 有一定接近性，但仍保留正 margin（region margin=0.955）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C0-T1 / lyric-brightened + lyric-intensified + high cross-modal tension，tension strength percentile 为 89.9%；歌词侧相对音频侧更明亮/更正向；歌词侧更激活。综合 descriptor profile 的高权重词包括：audio-lyric contrast、high cross-modal tension、lyric arousal intensification、lyric valence uplift、lyrics brighten the affect。

**A045 English**: The song is assigned to Subdued Melancholy and is peripheral within the assigned region, with region typicality 1.3% and assigned-region soft confidence 93.7%; it is moderately close to the nearest alternative region, C2 Volatile Intensity, while retaining a positive margin (region margin=0.955). The calibrated cross-modal tension / audio-lyric contrast profile is C0-T1 / lyric-brightened + lyric-intensified + high cross-modal tension, with tension strength percentile 89.9%; the lyrics are brighter or more positive than the audio; the lyrics are more activating. The top descriptor profile is: audio-lyric contrast, high cross-modal tension, lyric arousal intensification, lyric valence uplift, lyrics brighten the affect.


**A034 Chinese**: 该歌曲属于 Subdued Melancholy 区域，是所属区域内的外围样本；region typicality 为 12.0%，其 balanced VA 位置相对该区域原型的 soft confidence 为 84.3%。该样本与最近替代区域 C2 Volatile Intensity 有一定接近性，但仍保留正 margin（region margin=0.851）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C0-T1 / lyric-brightened + lyric-intensified + high cross-modal tension，tension strength percentile 为 75.9%；歌词侧相对音频侧更明亮/更正向；歌词侧更激活。综合 descriptor profile 的高权重词包括：audio-lyric contrast、high cross-modal tension、lyric arousal intensification、lyric valence uplift、lyrics brighten the affect。

**A034 English**: The song is assigned to Subdued Melancholy and is peripheral within the assigned region, with region typicality 12.0% and assigned-region soft confidence 84.3%; it is moderately close to the nearest alternative region, C2 Volatile Intensity, while retaining a positive margin (region margin=0.851). The calibrated cross-modal tension / audio-lyric contrast profile is C0-T1 / lyric-brightened + lyric-intensified + high cross-modal tension, with tension strength percentile 75.9%; the lyrics are brighter or more positive than the audio; the lyrics are more activating. The top descriptor profile is: audio-lyric contrast, high cross-modal tension, lyric arousal intensification, lyric valence uplift, lyrics brighten the affect.


**A061-66 Chinese**: 该歌曲属于 Subdued Melancholy 区域，是高度原型样本；region typicality 为 98.5%，其 balanced VA 位置相对该区域原型的 soft confidence 为 84.3%。该样本与最近替代区域 C1 Gentle Warmth 明显分离（region margin=1.870）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C0-T1 / lyric-brightened + lyric-intensified + high cross-modal tension，tension strength percentile 为 59.3%；歌词侧相对音频侧更明亮/更正向；歌词侧更激活。综合 descriptor profile 的高权重词包括：introspective、low arousal、melancholic、somber、subdued melancholy。

**A061-66 English**: The song is assigned to Subdued Melancholy and is highly prototypical, with region typicality 98.5% and assigned-region soft confidence 84.3%; it is clearly separated from the nearest alternative region, C1 Gentle Warmth (region margin=1.870). The calibrated cross-modal tension / audio-lyric contrast profile is C0-T1 / lyric-brightened + lyric-intensified + high cross-modal tension, with tension strength percentile 59.3%; the lyrics are brighter or more positive than the audio; the lyrics are more activating. The top descriptor profile is: introspective, low arousal, melancholic, somber, subdued melancholy.


**A171 Chinese**: 该歌曲属于 Subdued Melancholy 区域，是高度原型样本；region typicality 为 98.9%，其 balanced VA 位置相对该区域原型的 soft confidence 为 88.4%。该样本与最近替代区域 C1 Gentle Warmth 明显分离（region margin=2.080）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C0-T1 / lyric-brightened + lyric-intensified + high cross-modal tension，tension strength percentile 为 83.4%；歌词侧相对音频侧更明亮/更正向；歌词侧更激活。综合 descriptor profile 的高权重词包括：introspective、low arousal、melancholic、somber、subdued melancholy。

**A171 English**: The song is assigned to Subdued Melancholy and is highly prototypical, with region typicality 98.9% and assigned-region soft confidence 88.4%; it is clearly separated from the nearest alternative region, C1 Gentle Warmth (region margin=2.080). The calibrated cross-modal tension / audio-lyric contrast profile is C0-T1 / lyric-brightened + lyric-intensified + high cross-modal tension, with tension strength percentile 83.4%; the lyrics are brighter or more positive than the audio; the lyrics are more activating. The top descriptor profile is: introspective, low arousal, melancholic, somber, subdued melancholy.


**MT0003274067 Chinese**: 该歌曲属于 Subdued Melancholy 区域，是高度原型样本；region typicality 为 87.2%，其 balanced VA 位置相对该区域原型的 soft confidence 为 76.4%。该样本与最近替代区域 C1 Gentle Warmth 明显分离（region margin=1.194）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C0-T1 / lyric-brightened + lyric-intensified + high cross-modal tension，tension strength percentile 为 98.4%；歌词侧相对音频侧更明亮/更正向；歌词侧更激活。综合 descriptor profile 的高权重词包括：introspective、low arousal、melancholic、somber、subdued melancholy。

**MT0003274067 English**: The song is assigned to Subdued Melancholy and is highly prototypical, with region typicality 87.2% and assigned-region soft confidence 76.4%; it is clearly separated from the nearest alternative region, C1 Gentle Warmth (region margin=1.194). The calibrated cross-modal tension / audio-lyric contrast profile is C0-T1 / lyric-brightened + lyric-intensified + high cross-modal tension, with tension strength percentile 98.4%; the lyrics are brighter or more positive than the audio; the lyrics are more activating. The top descriptor profile is: introspective, low arousal, melancholic, somber, subdued melancholy.


**A024 Chinese**: 该歌曲属于 Gentle Warmth 区域，是所属区域内的外围样本；region typicality 为 8.1%，其 balanced VA 位置相对该区域原型的 soft confidence 为 90.0%。该样本与最近替代区域 C3 Playful Vitality 明显分离（region margin=1.012）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C1-T0 / modality-consistent，tension strength percentile 为 10.2%；张力强度较低，因此不把小的音频-歌词方向差解释为主要证据。综合 descriptor profile 的高权重词包括：affective concordance、audio-lyric agreement、calm-positive、gentle warmth、romantic。

**A024 English**: The song is assigned to Gentle Warmth and is peripheral within the assigned region, with region typicality 8.1% and assigned-region soft confidence 90.0%; it is clearly separated from the nearest alternative region, C3 Playful Vitality (region margin=1.012). The calibrated cross-modal tension / audio-lyric contrast profile is C1-T0 / modality-consistent, with tension strength percentile 10.2%; tension strength is low, so small audio-lyric directional differences are not used as primary evidence. The top descriptor profile is: affective concordance, audio-lyric agreement, calm-positive, gentle warmth, romantic.


**A017 Chinese**: 该歌曲属于 Gentle Warmth 区域，是所属区域内的外围样本；region typicality 为 2.4%，其 balanced VA 位置相对该区域原型的 soft confidence 为 75.8%。该样本与最近替代区域 C3 Playful Vitality 有一定接近性，但仍保留正 margin（region margin=0.482）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C1-T0 / modality-consistent，tension strength percentile 为 76.8%；歌词侧相对音频侧更暗/更负向；歌词侧更激活。综合 descriptor profile 的高权重词包括：lyric valence tempering、lyrics darken the affect、lyric arousal intensification、lyrics intensify the affect、calm-positive。

**A017 English**: The song is assigned to Gentle Warmth and is peripheral within the assigned region, with region typicality 2.4% and assigned-region soft confidence 75.8%; it is moderately close to the nearest alternative region, C3 Playful Vitality, while retaining a positive margin (region margin=0.482). The calibrated cross-modal tension / audio-lyric contrast profile is C1-T0 / modality-consistent, with tension strength percentile 76.8%; the lyrics darken the affect relative to the audio; the lyrics are more activating. The top descriptor profile is: lyric valence tempering, lyrics darken the affect, lyric arousal intensification, lyrics intensify the affect, calm-positive.


**A021 Chinese**: 该歌曲属于 Gentle Warmth 区域，是有代表性但并非最中心的样本；region typicality 为 52.7%，其 balanced VA 位置相对该区域原型的 soft confidence 为 90.6%。该样本与最近替代区域 C0 Subdued Melancholy 明显分离（region margin=1.388）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C1-T1 / lyric-brightened + lyric-intensified，tension strength percentile 为 81.7%；歌词侧相对音频侧更明亮/更正向；歌词侧更激活。综合 descriptor profile 的高权重词包括：audio-lyric contrast、high cross-modal tension、lyric arousal intensification、lyric valence uplift、lyrics brighten the affect。

**A021 English**: The song is assigned to Gentle Warmth and is representative but not central, with region typicality 52.7% and assigned-region soft confidence 90.6%; it is clearly separated from the nearest alternative region, C0 Subdued Melancholy (region margin=1.388). The calibrated cross-modal tension / audio-lyric contrast profile is C1-T1 / lyric-brightened + lyric-intensified, with tension strength percentile 81.7%; the lyrics are brighter or more positive than the audio; the lyrics are more activating. The top descriptor profile is: audio-lyric contrast, high cross-modal tension, lyric arousal intensification, lyric valence uplift, lyrics brighten the affect.


**A029 Chinese**: 该歌曲属于 Gentle Warmth 区域，是高度原型样本；region typicality 为 88.6%，其 balanced VA 位置相对该区域原型的 soft confidence 为 91.7%。该样本与最近替代区域 C0 Subdued Melancholy 明显分离（region margin=1.847）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C1-T1 / lyric-brightened + lyric-intensified，tension strength percentile 为 88.1%；歌词侧相对音频侧更明亮/更正向；歌词侧更激活。综合 descriptor profile 的高权重词包括：calm-positive、gentle warmth、romantic、soft、warm。

**A029 English**: The song is assigned to Gentle Warmth and is highly prototypical, with region typicality 88.6% and assigned-region soft confidence 91.7%; it is clearly separated from the nearest alternative region, C0 Subdued Melancholy (region margin=1.847). The calibrated cross-modal tension / audio-lyric contrast profile is C1-T1 / lyric-brightened + lyric-intensified, with tension strength percentile 88.1%; the lyrics are brighter or more positive than the audio; the lyrics are more activating. The top descriptor profile is: calm-positive, gentle warmth, romantic, soft, warm.


**MT0005344437 Chinese**: 该歌曲属于 Gentle Warmth 区域，是高度原型样本；region typicality 为 99.6%，其 balanced VA 位置相对该区域原型的 soft confidence 为 93.4%。该样本与最近替代区域 C3 Playful Vitality 明显分离（region margin=2.493）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C1-T1 / lyric-brightened + lyric-intensified，tension strength percentile 为 86.4%；歌词侧相对音频侧更明亮/更正向；歌词侧更激活。综合 descriptor profile 的高权重词包括：calm-positive、gentle warmth、romantic、soft、warm。

**MT0005344437 English**: The song is assigned to Gentle Warmth and is highly prototypical, with region typicality 99.6% and assigned-region soft confidence 93.4%; it is clearly separated from the nearest alternative region, C3 Playful Vitality (region margin=2.493). The calibrated cross-modal tension / audio-lyric contrast profile is C1-T1 / lyric-brightened + lyric-intensified, with tension strength percentile 86.4%; the lyrics are brighter or more positive than the audio; the lyrics are more activating. The top descriptor profile is: calm-positive, gentle warmth, romantic, soft, warm.


**A013 Chinese**: 该歌曲属于 Volatile Intensity 区域，是所属区域内的外围样本；region typicality 为 45.2%，其 balanced VA 位置相对该区域原型的 soft confidence 为 98.8%。该样本与最近替代区域 C3 Playful Vitality 明显分离（region margin=2.213）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C2-T0 / lyric-brightened，tension strength percentile 为 22.9%；张力强度较低，因此不把小的音频-歌词方向差解释为主要证据。综合 descriptor profile 的高权重词包括：aggressive、high arousal、negative-active、tense、volatile intensity。

**A013 English**: The song is assigned to Volatile Intensity and is peripheral within the assigned region, with region typicality 45.2% and assigned-region soft confidence 98.8%; it is clearly separated from the nearest alternative region, C3 Playful Vitality (region margin=2.213). The calibrated cross-modal tension / audio-lyric contrast profile is C2-T0 / lyric-brightened, with tension strength percentile 22.9%; tension strength is low, so small audio-lyric directional differences are not used as primary evidence. The top descriptor profile is: aggressive, high arousal, negative-active, tense, volatile intensity.


**A010 Chinese**: 该歌曲属于 Volatile Intensity 区域，是所属区域内的外围样本；region typicality 为 2.6%，其 balanced VA 位置相对该区域原型的 soft confidence 为 97.9%。该样本与最近替代区域 C0 Subdued Melancholy 明显分离（region margin=1.179）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C2-T1 / lyric-intensified + high cross-modal tension，tension strength percentile 为 98.0%；歌词侧相对音频侧更明亮/更正向；歌词侧更激活。综合 descriptor profile 的高权重词包括：lyric arousal intensification、lyrics intensify the affect、audio-lyric contrast、high cross-modal tension、aggressive。

**A010 English**: The song is assigned to Volatile Intensity and is peripheral within the assigned region, with region typicality 2.6% and assigned-region soft confidence 97.9%; it is clearly separated from the nearest alternative region, C0 Subdued Melancholy (region margin=1.179). The calibrated cross-modal tension / audio-lyric contrast profile is C2-T1 / lyric-intensified + high cross-modal tension, with tension strength percentile 98.0%; the lyrics are brighter or more positive than the audio; the lyrics are more activating. The top descriptor profile is: lyric arousal intensification, lyrics intensify the affect, audio-lyric contrast, high cross-modal tension, aggressive.


**A151-172 Chinese**: 该歌曲属于 Volatile Intensity 区域，是有代表性但并非最中心的样本；region typicality 为 66.1%，其 balanced VA 位置相对该区域原型的 soft confidence 为 99.3%。该样本与最近替代区域 C0 Subdued Melancholy 明显分离（region margin=2.507）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C2-T1 / lyric-intensified + high cross-modal tension，tension strength percentile 为 76.4%；歌词侧相对音频侧更明亮/更正向；歌词侧更激活。综合 descriptor profile 的高权重词包括：aggressive、high arousal、negative-active、tense、volatile intensity。

**A151-172 English**: The song is assigned to Volatile Intensity and is representative but not central, with region typicality 66.1% and assigned-region soft confidence 99.3%; it is clearly separated from the nearest alternative region, C0 Subdued Melancholy (region margin=2.507). The calibrated cross-modal tension / audio-lyric contrast profile is C2-T1 / lyric-intensified + high cross-modal tension, with tension strength percentile 76.4%; the lyrics are brighter or more positive than the audio; the lyrics are more activating. The top descriptor profile is: aggressive, high arousal, negative-active, tense, volatile intensity.


**A183 Chinese**: 该歌曲属于 Volatile Intensity 区域，是所属区域内的外围样本；region typicality 为 49.9%，其 balanced VA 位置相对该区域原型的 soft confidence 为 99.8%。该样本与最近替代区域 C3 Playful Vitality 明显分离（region margin=2.787）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C2-T0 / lyric-brightened，tension strength percentile 为 36.5%；歌词侧相对音频侧更明亮/更正向；歌词侧更柔和/低唤醒。综合 descriptor profile 的高权重词包括：aggressive、high arousal、negative-active、tense、volatile intensity。

**A183 English**: The song is assigned to Volatile Intensity and is peripheral within the assigned region, with region typicality 49.9% and assigned-region soft confidence 99.8%; it is clearly separated from the nearest alternative region, C3 Playful Vitality (region margin=2.787). The calibrated cross-modal tension / audio-lyric contrast profile is C2-T0 / lyric-brightened, with tension strength percentile 36.5%; the lyrics are brighter or more positive than the audio; the lyrics soften the arousal profile. The top descriptor profile is: aggressive, high arousal, negative-active, tense, volatile intensity.


**MT0007075838 Chinese**: 该歌曲属于 Volatile Intensity 区域，是高度原型样本；region typicality 为 97.2%，其 balanced VA 位置相对该区域原型的 soft confidence 为 96.9%。该样本与最近替代区域 C0 Subdued Melancholy 明显分离（region margin=2.530）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C2-T2 / lyric-brightened + high cross-modal tension，tension strength percentile 为 92.1%；歌词侧相对音频侧更明亮/更正向；歌词侧更激活。综合 descriptor profile 的高权重词包括：aggressive、high arousal、negative-active、tense、volatile intensity。

**MT0007075838 English**: The song is assigned to Volatile Intensity and is highly prototypical, with region typicality 97.2% and assigned-region soft confidence 96.9%; it is clearly separated from the nearest alternative region, C0 Subdued Melancholy (region margin=2.530). The calibrated cross-modal tension / audio-lyric contrast profile is C2-T2 / lyric-brightened + high cross-modal tension, with tension strength percentile 92.1%; the lyrics are brighter or more positive than the audio; the lyrics are more activating. The top descriptor profile is: aggressive, high arousal, negative-active, tense, volatile intensity.


**A049 Chinese**: 该歌曲属于 Volatile Intensity 区域，是边界样本，不应用作主区域证据；region typicality 为 1.6%，其 balanced VA 位置相对该区域原型的 soft confidence 为 45.3%。该样本在 post-hoc prototype 距离下更接近 C3 Playful Vitality，只能作为边界/歧义案例使用（region margin=-0.064）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C2-T1 / lyric-intensified + high cross-modal tension，tension strength percentile 为 23.6%；张力强度较低，因此不把小的音频-歌词方向差解释为主要证据。综合 descriptor profile 的高权重词包括：boundary between assigned region and nearest alternative、nearest alternative: C3 Playful Vitality、assigned region: Volatile Intensity、affective concordance、audio-lyric agreement。

**A049 English**: The song is assigned to Volatile Intensity and is a boundary case, not used as main region evidence, with region typicality 1.6% and assigned-region soft confidence 45.3%; post-hoc prototype distance places it closer to C3 Playful Vitality; use it only as a boundary or ambiguity case (region margin=-0.064). The calibrated cross-modal tension / audio-lyric contrast profile is C2-T1 / lyric-intensified + high cross-modal tension, with tension strength percentile 23.6%; tension strength is low, so small audio-lyric directional differences are not used as primary evidence. The top descriptor profile is: boundary between assigned region and nearest alternative, nearest alternative: C3 Playful Vitality, assigned region: Volatile Intensity, affective concordance, audio-lyric agreement.


**A014 Chinese**: 该歌曲属于 Playful Vitality 区域，是有代表性但并非最中心的样本；region typicality 为 69.0%，其 balanced VA 位置相对该区域原型的 soft confidence 为 88.8%。该样本与最近替代区域 C1 Gentle Warmth 明显分离（region margin=1.441）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C3-T0 / modality-consistent，tension strength percentile 为 28.6%；张力强度较低，因此不把小的音频-歌词方向差解释为主要证据。综合 descriptor profile 的高权重词包括：bright、danceable、energetic、playful vitality、positive-active。

**A014 English**: The song is assigned to Playful Vitality and is representative but not central, with region typicality 69.0% and assigned-region soft confidence 88.8%; it is clearly separated from the nearest alternative region, C1 Gentle Warmth (region margin=1.441). The calibrated cross-modal tension / audio-lyric contrast profile is C3-T0 / modality-consistent, with tension strength percentile 28.6%; tension strength is low, so small audio-lyric directional differences are not used as primary evidence. The top descriptor profile is: bright, danceable, energetic, playful vitality, positive-active.


**A001 Chinese**: 该歌曲属于 Playful Vitality 区域，是所属区域内的外围样本；region typicality 为 4.9%，其 balanced VA 位置相对该区域原型的 soft confidence 为 71.9%。该样本与最近替代区域 C1 Gentle Warmth 有一定接近性，但仍保留正 margin（region margin=0.363）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C3-T0 / modality-consistent，tension strength percentile 为 39.7%；该样本标记为 modality-consistent，方向差未达到门控阈值，因此解释为音频与歌词基本一致。综合 descriptor profile 的高权重词包括：bright、danceable、energetic、playful vitality、positive-active。

**A001 English**: The song is assigned to Playful Vitality and is peripheral within the assigned region, with region typicality 4.9% and assigned-region soft confidence 71.9%; it is moderately close to the nearest alternative region, C1 Gentle Warmth, while retaining a positive margin (region margin=0.363). The calibrated cross-modal tension / audio-lyric contrast profile is C3-T0 / modality-consistent, with tension strength percentile 39.7%; the sample is modality-consistent and directional deltas do not pass the gate, so it is interpreted as broad audio-lyric agreement. The top descriptor profile is: bright, danceable, energetic, playful vitality, positive-active.


**A019 Chinese**: 该歌曲属于 Playful Vitality 区域，是有代表性但并非最中心的样本；region typicality 为 78.9%，其 balanced VA 位置相对该区域原型的 soft confidence 为 98.0%。该样本与最近替代区域 C1 Gentle Warmth 明显分离（region margin=2.273）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C3-T1 / lyric-darkened + lyric-softened + high cross-modal tension，tension strength percentile 为 61.3%；歌词侧相对音频侧更暗/更负向；歌词侧更柔和/低唤醒。综合 descriptor profile 的高权重词包括：bright、danceable、energetic、playful vitality、positive-active。

**A019 English**: The song is assigned to Playful Vitality and is representative but not central, with region typicality 78.9% and assigned-region soft confidence 98.0%; it is clearly separated from the nearest alternative region, C1 Gentle Warmth (region margin=2.273). The calibrated cross-modal tension / audio-lyric contrast profile is C3-T1 / lyric-darkened + lyric-softened + high cross-modal tension, with tension strength percentile 61.3%; the lyrics darken the affect relative to the audio; the lyrics soften the arousal profile. The top descriptor profile is: bright, danceable, energetic, playful vitality, positive-active.


**MT0006683875 Chinese**: 该歌曲属于 Playful Vitality 区域，是高度原型样本；region typicality 为 97.7%，其 balanced VA 位置相对该区域原型的 soft confidence 为 99.2%。该样本与最近替代区域 C1 Gentle Warmth 明显分离（region margin=2.939）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C3-T1 / lyric-darkened + lyric-softened + high cross-modal tension，tension strength percentile 为 91.5%；歌词侧相对音频侧更暗/更负向；歌词侧更柔和/低唤醒。综合 descriptor profile 的高权重词包括：bright、danceable、energetic、playful vitality、positive-active。

**MT0006683875 English**: The song is assigned to Playful Vitality and is highly prototypical, with region typicality 97.7% and assigned-region soft confidence 99.2%; it is clearly separated from the nearest alternative region, C1 Gentle Warmth (region margin=2.939). The calibrated cross-modal tension / audio-lyric contrast profile is C3-T1 / lyric-darkened + lyric-softened + high cross-modal tension, with tension strength percentile 91.5%; the lyrics darken the affect relative to the audio; the lyrics soften the arousal profile. The top descriptor profile is: bright, danceable, energetic, playful vitality, positive-active.


**A005 Chinese**: 该歌曲属于 Playful Vitality 区域，是所属区域内的外围样本；region typicality 为 10.3%，其 balanced VA 位置相对该区域原型的 soft confidence 为 100.0%。该样本与最近替代区域 C1 Gentle Warmth 明显分离（region margin=3.074）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C3-T1 / lyric-darkened + lyric-softened + high cross-modal tension，tension strength percentile 为 89.2%；歌词侧相对音频侧更暗/更负向；歌词侧更柔和/低唤醒。综合 descriptor profile 的高权重词包括：audio-lyric contrast、high cross-modal tension、lyric arousal softening、lyric valence tempering、lyrics darken the affect。

**A005 English**: The song is assigned to Playful Vitality and is peripheral within the assigned region, with region typicality 10.3% and assigned-region soft confidence 100.0%; it is clearly separated from the nearest alternative region, C1 Gentle Warmth (region margin=3.074). The calibrated cross-modal tension / audio-lyric contrast profile is C3-T1 / lyric-darkened + lyric-softened + high cross-modal tension, with tension strength percentile 89.2%; the lyrics darken the affect relative to the audio; the lyrics soften the arousal profile. The top descriptor profile is: audio-lyric contrast, high cross-modal tension, lyric arousal softening, lyric valence tempering, lyrics darken the affect.


**MT0027072382 Chinese**: 该歌曲属于 Playful Vitality 区域，是高度原型样本；region typicality 为 94.8%，其 balanced VA 位置相对该区域原型的 soft confidence 为 97.5%。该样本与最近替代区域 C1 Gentle Warmth 明显分离（region margin=2.488）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C3-T0 / modality-consistent，tension strength percentile 为 19.2%；张力强度较低，因此不把小的音频-歌词方向差解释为主要证据。综合 descriptor profile 的高权重词包括：bright、danceable、energetic、playful vitality、positive-active。

**MT0027072382 English**: The song is assigned to Playful Vitality and is highly prototypical, with region typicality 94.8% and assigned-region soft confidence 97.5%; it is clearly separated from the nearest alternative region, C1 Gentle Warmth (region margin=2.488). The calibrated cross-modal tension / audio-lyric contrast profile is C3-T0 / modality-consistent, with tension strength percentile 19.2%; tension strength is low, so small audio-lyric directional differences are not used as primary evidence. The top descriptor profile is: bright, danceable, energetic, playful vitality, positive-active.


**MT0004141823 Chinese**: 该歌曲属于 Playful Vitality 区域，是高度原型样本；region typicality 为 96.9%，其 balanced VA 位置相对该区域原型的 soft confidence 为 98.5%。该样本与最近替代区域 C1 Gentle Warmth 明显分离（region margin=2.714）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C3-T0 / modality-consistent，tension strength percentile 为 23.5%；张力强度较低，因此不把小的音频-歌词方向差解释为主要证据。综合 descriptor profile 的高权重词包括：bright、danceable、energetic、playful vitality、positive-active。

**MT0004141823 English**: The song is assigned to Playful Vitality and is highly prototypical, with region typicality 96.9% and assigned-region soft confidence 98.5%; it is clearly separated from the nearest alternative region, C1 Gentle Warmth (region margin=2.714). The calibrated cross-modal tension / audio-lyric contrast profile is C3-T0 / modality-consistent, with tension strength percentile 23.5%; tension strength is low, so small audio-lyric directional differences are not used as primary evidence. The top descriptor profile is: bright, danceable, energetic, playful vitality, positive-active.


**MT0033512773 Chinese**: 该歌曲属于 Playful Vitality 区域，是高度原型样本；region typicality 为 99.5%，其 balanced VA 位置相对该区域原型的 soft confidence 为 98.3%。该样本与最近替代区域 C1 Gentle Warmth 明显分离（region margin=2.749）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C3-T0 / modality-consistent，tension strength percentile 为 20.4%；张力强度较低，因此不把小的音频-歌词方向差解释为主要证据。综合 descriptor profile 的高权重词包括：bright、danceable、energetic、playful vitality、positive-active。

**MT0033512773 English**: The song is assigned to Playful Vitality and is highly prototypical, with region typicality 99.5% and assigned-region soft confidence 98.3%; it is clearly separated from the nearest alternative region, C1 Gentle Warmth (region margin=2.749). The calibrated cross-modal tension / audio-lyric contrast profile is C3-T0 / modality-consistent, with tension strength percentile 20.4%; tension strength is low, so small audio-lyric directional differences are not used as primary evidence. The top descriptor profile is: bright, danceable, energetic, playful vitality, positive-active.


**MT0002360219 Chinese**: 该歌曲属于 Gentle Warmth 区域，是高度原型样本；region typicality 为 98.7%，其 balanced VA 位置相对该区域原型的 soft confidence 为 93.6%。该样本与最近替代区域 C3 Playful Vitality 明显分离（region margin=2.380）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C1-T1 / lyric-brightened + lyric-intensified，tension strength percentile 为 97.2%；歌词侧相对音频侧更明亮/更正向；歌词侧更激活。综合 descriptor profile 的高权重词包括：calm-positive、gentle warmth、romantic、soft、warm。

**MT0002360219 English**: The song is assigned to Gentle Warmth and is highly prototypical, with region typicality 98.7% and assigned-region soft confidence 93.6%; it is clearly separated from the nearest alternative region, C3 Playful Vitality (region margin=2.380). The calibrated cross-modal tension / audio-lyric contrast profile is C1-T1 / lyric-brightened + lyric-intensified, with tension strength percentile 97.2%; the lyrics are brighter or more positive than the audio; the lyrics are more activating. The top descriptor profile is: calm-positive, gentle warmth, romantic, soft, warm.


**MT0000151451 Chinese**: 该歌曲属于 Gentle Warmth 区域，是高度原型样本；region typicality 为 96.8%，其 balanced VA 位置相对该区域原型的 soft confidence 为 89.4%。该样本与最近替代区域 C0 Subdued Melancholy 明显分离（region margin=1.997）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C1-T1 / lyric-brightened + lyric-intensified，tension strength percentile 为 81.0%；歌词侧相对音频侧更明亮/更正向；歌词侧更激活。综合 descriptor profile 的高权重词包括：calm-positive、gentle warmth、romantic、soft、warm。

**MT0000151451 English**: The song is assigned to Gentle Warmth and is highly prototypical, with region typicality 96.8% and assigned-region soft confidence 89.4%; it is clearly separated from the nearest alternative region, C0 Subdued Melancholy (region margin=1.997). The calibrated cross-modal tension / audio-lyric contrast profile is C1-T1 / lyric-brightened + lyric-intensified, with tension strength percentile 81.0%; the lyrics are brighter or more positive than the audio; the lyrics are more activating. The top descriptor profile is: calm-positive, gentle warmth, romantic, soft, warm.


**A028 Chinese**: 该歌曲属于 Volatile Intensity 区域，是所属区域内的外围样本；region typicality 为 0.5%，其 balanced VA 位置相对该区域原型的 soft confidence 为 93.7%。该样本与最近替代区域 C3 Playful Vitality 有一定接近性，但仍保留正 margin（region margin=0.661）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C2-T0 / lyric-brightened，tension strength percentile 为 67.8%；歌词侧相对音频侧更明亮/更正向；歌词侧更柔和/低唤醒。综合 descriptor profile 的高权重词包括：lyric arousal softening、lyrics soften the affect、aggressive、high arousal、negative-active。

**A028 English**: The song is assigned to Volatile Intensity and is peripheral within the assigned region, with region typicality 0.5% and assigned-region soft confidence 93.7%; it is moderately close to the nearest alternative region, C3 Playful Vitality, while retaining a positive margin (region margin=0.661). The calibrated cross-modal tension / audio-lyric contrast profile is C2-T0 / lyric-brightened, with tension strength percentile 67.8%; the lyrics are brighter or more positive than the audio; the lyrics soften the arousal profile. The top descriptor profile is: lyric arousal softening, lyrics soften the affect, aggressive, high arousal, negative-active.


**A085-69 Chinese**: 该歌曲属于 Volatile Intensity 区域，是高度原型样本；region typicality 为 93.3%，其 balanced VA 位置相对该区域原型的 soft confidence 为 98.8%。该样本与最近替代区域 C0 Subdued Melancholy 明显分离（region margin=2.890）。其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 C2-T1 / lyric-intensified + high cross-modal tension，tension strength percentile 为 65.8%；歌词侧相对音频侧更暗/更负向；歌词侧更激活。综合 descriptor profile 的高权重词包括：aggressive、high arousal、negative-active、tense、volatile intensity。

**A085-69 English**: The song is assigned to Volatile Intensity and is highly prototypical, with region typicality 93.3% and assigned-region soft confidence 98.8%; it is clearly separated from the nearest alternative region, C0 Subdued Melancholy (region margin=2.890). The calibrated cross-modal tension / audio-lyric contrast profile is C2-T1 / lyric-intensified + high cross-modal tension, with tension strength percentile 65.8%; the lyrics darken the affect relative to the audio; the lyrics are more activating. The top descriptor profile is: aggressive, high arousal, negative-active, tense, volatile intensity.


## Final Paper Candidate Table

| role | rank | song_id | title | artist | cluster | candidate type | region typicality | confidence | tension strength | external evidence |
|---|---:|---|---|---|---|---|---:|---:|---:|---|
| region_candidate | 1 | MT0010468178 | My Sweet Prince | Placebo | Subdued Melancholy | region_prototype | 100.0% | 87.3% | 39.8% | not_checked |
| region_candidate | 2 | MT0009769814 | Tortoise Regrets Hare | James Yorkston | Subdued Melancholy | region_prototype | 99.8% | 86.8% | 66.5% | not_checked |
| region_candidate | 3 | MT0004485367 | Create Me | Neil Diamond | Subdued Melancholy | region_prototype | 99.7% | 87.3% | 35.8% | not_checked |
| tension_case | 1 | MT0003504525 | Ain't No Santa | Trick Daddy | Subdued Melancholy | tension_case | 4.4% | 61.9% | 94.5% | not_checked |
| tension_case | 2 | MT0002986568 | When Things Go Wrong | Robin Lane & The Chartbusters | Subdued Melancholy | tension_case | 98.0% | 86.4% | 92.7% | not_checked |
| region_candidate | 1 | MT0030499533 | Lately | Tyrese | Gentle Warmth | tension_case | 100.0% | 93.1% | 88.8% | not_checked |
| region_candidate | 2 | MT0006137300 | How Far Am I from Canaan? | Sam Cooke | Gentle Warmth | region_prototype | 99.9% | 91.5% | 43.1% | not_checked |
| region_candidate | 3 | MT0003044417 | Just Because | Paul McCartney | Gentle Warmth | region_prototype | 99.7% | 91.4% | 63.1% | not_checked |
| tension_case | 1 | MT0000638634 | Heaven Help | Lenny Kravitz | Gentle Warmth | tension_case | 62.1% | 86.6% | 94.1% | not_checked |
| tension_case | 2 | MT0028335228 | Love of My Life | Jim Brickman | Gentle Warmth | tension_case | 43.5% | 64.0% | 93.8% | not_checked |
| region_candidate | 1 | MT0001738256 | Price You Pay | Agnostic Front | Volatile Intensity | tension_case | 100.0% | 98.6% | 89.4% | not_checked |
| region_candidate | 2 | MT0030184605 | Give Me It | The Cure | Volatile Intensity | region_prototype | 99.3% | 98.7% | 57.1% | not_checked |
| region_candidate | 3 | A055 | Born In The U.S.A. | Bruce Springsteen | Volatile Intensity | region_prototype | 98.7% | 97.6% | 27.2% | not_checked |
| tension_case | 1 | MT0010857270 | Reaganomics | D.R.I. | Volatile Intensity | tension_case | 1.1% | 100.0% | 92.6% | not_checked |
| tension_case | 2 | MT0030435905 | My Face Would Crack | Alec Empire | Volatile Intensity | tension_case | 55.6% | 81.8% | 91.3% | not_checked |
| region_candidate | 1 | MT0005674518 | Ain't That a Shame | John Lennon | Playful Vitality | region_prototype | 100.0% | 98.3% | 98.8% | not_checked |
| region_candidate | 2 | MT0001169715 | Red Blue Jeans and a Pony Tail | Gene Vincent | Playful Vitality | tension_case | 99.8% | 99.1% | 77.5% | not_checked |
| region_candidate | 3 | MT0033512773 | Do You Believe in Love | Huey Lewis & the News | Playful Vitality | region_prototype | 99.5% | 98.3% | 20.4% | not_checked |
| tension_case | 1 | MT0000948111 | Hot Dog | Elvis Presley | Playful Vitality | tension_case | 20.9% | 100.0% | 92.0% | not_checked |
| tension_case | 2 | MT0001217651 | Choppa Style | Choppa | Playful Vitality | tension_case | 4.7% | 100.0% | 91.8% | not_checked |

### Full-table boundary coverage

| song_id | title | artist | cluster | region role | tension | main text | region typicality | tension strength | top descriptors |
|---|---|---|---|---|---|---|---:|---:|---|
| MT0027002641 | All We Got | Michael McDonald | Playful Vitality | boundary | C3-T0 / modality-consistent | no | 29.1% | 8.0% | boundary between assigned region and nearest alternative, affective concordance, audio-lyric agreement, assigned region: Playful Vitality, nearest alternative: C1 Gentle Warmth |
| MT0013612461 | The Christmas Song | David Banner | Gentle Warmth | boundary | C1-T1 / lyric-brightened + lyric-intensified | no | 2.8% | 91.9% | boundary between assigned region and nearest alternative, lyric valence uplift, lyrics brighten the affect, nearest alternative: C3 Playful Vitality, assigned region: Gentle Warmth |
| MT0026973618 | I'm Satisfied | Mississippi John Hurt | Subdued Melancholy | boundary | C0-T0 / modality-consistent | no | 1.6% | 37.2% | boundary between assigned region and nearest alternative, nearest alternative: C1 Gentle Warmth, affective concordance, audio-lyric agreement, assigned region: Subdued Melancholy |
| MT0012900592 | This Is Radio Clash | The Clash | Volatile Intensity | boundary | C2-T2 / lyric-brightened + high cross-modal tension | no | 1.8% | 75.1% | boundary between assigned region and nearest alternative, lyric arousal softening, lyrics soften the affect, nearest alternative: C3 Playful Vitality, assigned region: Volatile Intensity |

## Cluster Representatives

### C3 Playful Vitality

| rank | song_id | title | artist | region typicality | confidence |
|---:|---|---|---|---:|---:|
| 1 | MT0005674518 | Ain't That a Shame | John Lennon | 100.0% | 98.3% |
| 2 | MT0001169715 | Red Blue Jeans and a Pony Tail | Gene Vincent | 99.8% | 99.1% |
| 3 | MT0033512773 | Do You Believe in Love | Huey Lewis & the News | 99.5% | 98.3% |
| 4 | MT0010705334 | I Feel for You | Prince | 99.3% | 98.7% |
| 5 | A146-118 | Twist and Shout | The Beatles | 99.1% | 98.4% |
| 6 | MT0014781093 | I Got Stung | Elvis | 98.8% | 99.1% |
| 7 | MT0026753827 | I Believe | Third Day | 98.6% | 98.0% |
| 8 | MT0008113039 | Don't Leave, I Think I Love You | Toby Keith | 98.4% | 98.7% |
| 9 | MT0002113056 | Ac-Cent-Tchu-Ate the Positive | The Andrews Sisters | 98.1% | 98.5% |
| 10 | MT0008363189 | Magic Road | Al Green | 97.9% | 97.9% |

### C1 Gentle Warmth

| rank | song_id | title | artist | region typicality | confidence |
|---:|---|---|---|---:|---:|
| 1 | MT0030499533 | Lately | Tyrese | 100.0% | 93.1% |
| 2 | MT0006137300 | How Far Am I from Canaan? | Sam Cooke | 99.9% | 91.5% |
| 3 | MT0003044417 | Just Because | Paul McCartney | 99.7% | 91.4% |
| 4 | MT0005344437 | Send One Your Love | Stevie Wonder | 99.6% | 93.4% |
| 5 | MT0013095380 | Way Beyond | Morcheeba | 99.4% | 91.6% |
| 6 | MT0005827376 | I Want You | Marvin Gaye | 99.3% | 92.6% |
| 7 | MT0015506396 | His Eyes | Steven Curtis Chapman | 99.1% | 92.6% |
| 8 | MT0015465230 | I'm Through with Love | Bing Crosby | 99.0% | 92.3% |
| 9 | MT0003410384 | No Speech, No Language | Gregory Isaacs | 98.8% | 93.3% |
| 10 | MT0002360219 | Keepin' the Summer Alive | The Beach Boys | 98.7% | 93.6% |

### C0 Subdued Melancholy

| rank | song_id | title | artist | region typicality | confidence |
|---:|---|---|---|---:|---:|
| 1 | MT0010468178 | My Sweet Prince | Placebo | 100.0% | 87.3% |
| 2 | MT0009769814 | Tortoise Regrets Hare | James Yorkston | 99.8% | 86.8% |
| 3 | MT0004485367 | Create Me | Neil Diamond | 99.7% | 87.3% |
| 4 | MT0019766193 | In a Modern World | Fischerspooner | 99.5% | 86.6% |
| 5 | MT0000619767 | It's All Over But the Crying | Garbage | 99.3% | 85.3% |
| 6 | MT0034005433 | Everybody Loves a Nut | Johnny Cash | 99.2% | 88.2% |
| 7 | MT0009495280 | If I Don't See You Again | Neil Diamond | 99.0% | 85.9% |
| 8 | A171 | Say Something | A Great Big World | 98.9% | 88.4% |
| 9 | MT0006833696 | When You Leave That Way You Can Never Go Back | Confederate Railroad | 98.7% | 85.8% |
| 10 | A061-66 | Mad World | Gary Jules | 98.5% | 84.3% |

### C2 Volatile Intensity

| rank | song_id | title | artist | region typicality | confidence |
|---:|---|---|---|---:|---:|
| 1 | MT0001738256 | Price You Pay | Agnostic Front | 100.0% | 98.6% |
| 2 | MT0003444822 | Hate Me | The Distillers | 99.8% | 98.1% |
| 3 | MT0000900387 | Let's Ride | Ja Rule | 99.7% | 98.6% |
| 4 | MT0001641166 | Solitaire | Face to Face | 99.5% | 98.7% |
| 5 | MT0030184605 | Give Me It | The Cure | 99.3% | 98.7% |
| 6 | MT0006761971 | Mac and Brad | Scarface | 99.2% | 98.9% |
| 7 | MT0014135173 | Resurrection #9 | Burn the Priest | 99.0% | 99.0% |
| 8 | MT0030233759 | Juggernaut | Adam West | 98.9% | 98.9% |
| 9 | A055 | Born In The U.S.A. | Bruce Springsteen | 98.7% | 97.6% |
| 10 | MT0006903607 | Waiting for the Meat | Fear | 98.5% | 98.3% |


## Tension Subtype Representatives

### C0-T0 / modality-consistent

| rank | song_id | title | artist | cluster | tension typicality | strength percentile |
|---:|---|---|---|---|---:|---:|
| 1 | MT0002969144 | All Through the Night | Ella Fitzgerald | Subdued Melancholy | 100.0% | 17.9% |
| 2 | MT0008969183 | Sub-Culture | New Order | Subdued Melancholy | 99.4% | 20.2% |
| 3 | MT0015664499 | Rivers of Babylon | Don Carlos | Subdued Melancholy | 99.1% | 16.9% |
| 4 | MT0010630017 | This Heartache Never Sleeps | Mark Chesnutt | Subdued Melancholy | 98.8% | 19.2% |
| 5 | A157-71 | Yesterday | The Beatles | Subdued Melancholy | 98.0% | 13.8% |
| 6 | MT0000829765 | When a Love Song Sings the Blues | Trisha Yearwood | Subdued Melancholy | 97.4% | 22.3% |
| 7 | MT0034992958 | Slavery Days | Burning Spear | Subdued Melancholy | 96.8% | 11.7% |
| 8 | MT0009495285 | Forgotten | Neil Diamond | Subdued Melancholy | 96.5% | 9.6% |
| 9 | MT0003420996 | Ballad of the Sad Young Men | Roberta Flack | Subdued Melancholy | 96.2% | 20.5% |
| 10 | MT0005941732 | How Low? | Against Me! | Subdued Melancholy | 95.6% | 24.2% |

### C0-T1 / lyric-brightened + lyric-intensified + high cross-modal tension

| rank | song_id | title | artist | cluster | tension typicality | strength percentile |
|---:|---|---|---|---|---:|---:|
| 1 | MT0003367947 | Islands | Arab Strap | Subdued Melancholy | 100.0% | 80.2% |
| 2 | MT0001747053 | The Eulogy | CPO | Subdued Melancholy | 99.6% | 75.4% |
| 3 | MT0008953247 | My Hometown | Bruce Springsteen | Subdued Melancholy | 99.3% | 79.7% |
| 4 | MT0005760259 | Young, Gifted and Skint | New Model Army | Subdued Melancholy | 98.9% | 79.3% |
| 5 | A171 | Say Something | A Great Big World | Subdued Melancholy | 98.5% | 83.4% |
| 6 | MT0004317784 | Weightless Again | The Handsome Family | Subdued Melancholy | 98.2% | 83.9% |
| 7 | MT0004869271 | This Place | Joni Mitchell | Subdued Melancholy | 97.8% | 82.4% |
| 8 | MT0008421900 | Over the Next Hill (We'll Be Home) | Johnny Cash | Subdued Melancholy | 96.7% | 77.9% |
| 9 | MT0003907045 | Various Stages | Great Lake Swimmers | Subdued Melancholy | 96.3% | 73.5% |
| 10 | MT0000091781 | Love in the Afternoon | Marianne Faithfull | Subdued Melancholy | 96.0% | 81.0% |

### C1-T0 / modality-consistent

| rank | song_id | title | artist | cluster | tension typicality | strength percentile |
|---:|---|---|---|---|---:|---:|
| 1 | MT0003792475 | Rifles | Black Rebel Motorcycle Club | Gentle Warmth | 99.8% | 1.9% |
| 2 | MT0003081304 | Time Travelin' (A Tribute to Fela) | Common | Gentle Warmth | 99.3% | 2.4% |
| 3 | MT0000854722 | Mr. Man | Alicia Keys | Gentle Warmth | 99.1% | 1.2% |
| 4 | MT0007039082 | Five Candles (You Were There) | Jars of Clay | Gentle Warmth | 98.4% | 5.2% |
| 5 | MT0008673460 | Feels Like the First Time | Isaac Hayes | Gentle Warmth | 97.2% | 1.5% |
| 6 | MT0006044445 | The Unclouded Day | Randy Travis | Gentle Warmth | 97.0% | 2.9% |
| 7 | MT0000850050 | Baby, Now That I've Found You | Alison Krauss | Gentle Warmth | 96.5% | 7.4% |
| 8 | MT0029133325 | Tomorrow Started | Talk Talk | Gentle Warmth | 96.0% | 0.3% |
| 9 | MT0030438262 | London You're a Lady | The Pogues | Gentle Warmth | 95.1% | 6.0% |
| 10 | MT0000714708 | Music | Madonna | Gentle Warmth | 94.7% | 9.3% |

### C1-T1 / lyric-brightened + lyric-intensified

| rank | song_id | title | artist | cluster | tension typicality | strength percentile |
|---:|---|---|---|---|---:|---:|
| 1 | MT0032093528 | East of the Sun | Ella Fitzgerald | Gentle Warmth | 100.0% | 77.9% |
| 2 | MT0003213617 | Got to Make a Comeback | Robert Cray | Gentle Warmth | 99.6% | 79.1% |
| 3 | MT0013955066 | Scarborough Fair | Sergio Mendes | Gentle Warmth | 98.8% | 81.4% |
| 4 | MT0000906888 | The Way I Am | Merle Haggard | Gentle Warmth | 98.4% | 82.0% |
| 5 | MT0008236217 | It's a Beautiful Day (Reprise) | Queen | Gentle Warmth | 98.0% | 82.9% |
| 6 | MT0006813360 | Is This Desire? | PJ Harvey | Gentle Warmth | 97.6% | 82.6% |
| 7 | MT0028697763 | Have You Seen Her? | MC Hammer | Gentle Warmth | 97.2% | 78.6% |
| 8 | MT0028603801 | Mighty Love | Lisa Stansfield | Gentle Warmth | 96.8% | 72.1% |
| 9 | MT0015709701 | Don't Go Breaking My Heart | Burt Bacharach | Gentle Warmth | 96.4% | 71.8% |
| 10 | MT0007556029 | Hallelujah, I Love Her So | Harry Belafonte | Gentle Warmth | 96.0% | 77.6% |

### C2-T0 / lyric-brightened

| rank | song_id | title | artist | cluster | tension typicality | strength percentile |
|---:|---|---|---|---|---:|---:|
| 1 | MT0002711792 | Skinfather | Dismember | Volatile Intensity | 99.7% | 17.2% |
| 2 | MT0003728288 | Odious & Devious | ...And Oceans | Volatile Intensity | 99.4% | 16.7% |
| 3 | MT0008084656 | Pennywise | Pennywise | Volatile Intensity | 98.8% | 12.3% |
| 4 | MT0005287506 | Rentboy | Within Reach | Volatile Intensity | 98.5% | 14.1% |
| 5 | MT0012030863 | Another Nigger in the Morgue | Geto Boys | Volatile Intensity | 97.6% | 11.3% |
| 6 | A174 | Renegade | Eminem | Volatile Intensity | 96.8% | 19.8% |
| 7 | MT0028348785 | Straight to Hell | Flotsam and Jetsam | Volatile Intensity | 96.2% | 19.6% |
| 8 | MT0003762339 | Fading Dimensions | Darkane | Volatile Intensity | 95.9% | 19.5% |
| 9 | MT0006409530 | Use the Man | Megadeth | Volatile Intensity | 95.3% | 21.6% |
| 10 | MT0001974130 | Wormwood | Easy Rider | Volatile Intensity | 95.0% | 20.0% |

### C2-T1 / lyric-intensified + high cross-modal tension

| rank | song_id | title | artist | cluster | tension typicality | strength percentile |
|---:|---|---|---|---|---:|---:|
| 1 | MT0007627521 | Beneath the Crimson Vaults of Cydonia | Bal-Sagoth | Volatile Intensity | 93.5% | 57.8% |
| 2 | MT0009631750 | Break Down | Final Breath | Volatile Intensity | 89.1% | 25.5% |
| 3 | MT0002949712 | Inflikted | Cavalera Conspiracy | Volatile Intensity | 87.0% | 86.9% |
| 4 | MT0012009054 | Evil Walks | AC/DC | Volatile Intensity | 84.8% | 32.9% |
| 5 | A085-69 | Dying in the Sun | The Cranberries | Volatile Intensity | 82.6% | 65.8% |
| 6 | MT0008999728 | Rush of Deliverance | Vital Remains | Volatile Intensity | 76.1% | 32.7% |
| 7 | MT0010489498 | Mi Vida | Gipsy Kings | Volatile Intensity | 71.7% | 82.5% |
| 8 | MT0027320606 | La Dona | Gipsy Kings | Volatile Intensity | 71.7% | 82.5% |
| 9 | MT0031914705 | Vamos a Bailar | Gipsy Kings | Volatile Intensity | 71.7% | 82.5% |
| 10 | MT0007413949 | Tribulation | Divine Empire | Volatile Intensity | 63.0% | 51.7% |

### C2-T2 / lyric-brightened + high cross-modal tension

| rank | song_id | title | artist | cluster | tension typicality | strength percentile |
|---:|---|---|---|---|---:|---:|
| 1 | MT0002842988 | Train of Consequences | Megadeth | Volatile Intensity | 99.1% | 74.8% |
| 2 | MT0006695707 | Beast and the Harlot | Avenged Sevenfold | Volatile Intensity | 98.7% | 76.3% |
| 3 | MT0011945841 | Foreclosure of a Dream | Megadeth | Volatile Intensity | 97.8% | 77.1% |
| 4 | MT0026987140 | Pistol Grip Pump | Rage Against the Machine | Volatile Intensity | 97.3% | 76.8% |
| 5 | MT0027317557 | I Ain't With Being Broke | Geto Boys | Volatile Intensity | 96.9% | 73.2% |
| 6 | MT0010612023 | Dirty Mack | Ice Cube | Volatile Intensity | 96.4% | 69.1% |
| 7 | MT0026985012 | AmeriKKKa's Most Wanted | Ice Cube | Volatile Intensity | 95.6% | 68.4% |
| 8 | MT0026681751 | Catatonic | Slayer | Volatile Intensity | 95.1% | 80.5% |
| 9 | MT0010616215 | The Predator | Ice Cube | Volatile Intensity | 94.7% | 71.8% |
| 10 | MT0031964947 | Berkertex Bribe | Crass | Volatile Intensity | 94.2% | 77.9% |

### C3-T0 / modality-consistent

| rank | song_id | title | artist | cluster | tension typicality | strength percentile |
|---:|---|---|---|---|---:|---:|
| 1 | MT0011950450 | Hark! The Herald Angels Sing | Neil Diamond | Playful Vitality | 100.0% | 10.8% |
| 2 | MT0008363189 | Magic Road | Al Green | Playful Vitality | 98.8% | 14.6% |
| 3 | MT0004469135 | Satisfaction Guaranteed (Or Take Your Love Back) | Harold Melvin & the Blue Notes | Playful Vitality | 98.0% | 11.3% |
| 4 | MT0030362299 | Lover | Frank Sinatra | Playful Vitality | 97.2% | 6.6% |
| 5 | MT0026900439 | 8 Days of Christmas | Destiny's Child | Playful Vitality | 96.8% | 16.2% |
| 6 | MT0015680193 | Love It When You Call | The Feeling | Playful Vitality | 96.4% | 12.9% |
| 7 | MT0000225806 | I Need Your Love Tonight | Elvis Presley | Playful Vitality | 94.9% | 17.6% |
| 8 | MT0027072382 | Express Yourself | Madonna | Playful Vitality | 94.5% | 19.2% |
| 9 | MT0008029439 | I Love This Bar | Toby Keith | Playful Vitality | 93.7% | 16.0% |
| 10 | MT0033512773 | Do You Believe in Love | Huey Lewis & the News | Playful Vitality | 92.5% | 20.4% |

### C3-T1 / lyric-darkened + lyric-softened + high cross-modal tension

| rank | song_id | title | artist | cluster | tension typicality | strength percentile |
|---:|---|---|---|---|---:|---:|
| 1 | MT0000664362 | Hello Dolly | Harry Connick, Jr. | Playful Vitality | 99.4% | 73.9% |
| 2 | MT0028445777 | Early Mornin' | Britney Spears | Playful Vitality | 98.8% | 71.8% |
| 3 | MT0007274054 | Sex Machine (Part 1) | James Brown | Playful Vitality | 97.7% | 83.1% |
| 4 | MT0006406298 | Sally Sue Brown | Bob Dylan | Playful Vitality | 97.1% | 80.3% |
| 5 | MT0006431807 | Beer for My Horses | Toby Keith | Playful Vitality | 96.5% | 82.9% |
| 6 | MT0013064857 | Evening of the Day | Supergrass | Playful Vitality | 96.0% | 70.0% |
| 7 | MT0010290924 | Objection [Tango] | Shakira | Playful Vitality | 95.4% | 79.8% |
| 8 | MT0012863101 | You're Only Human (Second Wind) | Billy Joel | Playful Vitality | 94.8% | 81.9% |
| 9 | MT0013389935 | Shut Up | The Black Eyed Peas | Playful Vitality | 94.2% | 67.6% |
| 10 | MT0000110175 | Over You | Aaron Neville | Playful Vitality | 93.6% | 83.8% |

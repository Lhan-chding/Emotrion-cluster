from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from cluster.utils import find_column


@dataclass
class AlignConfig:
    order_csv: str
    audio_csv: str
    lyrics_csv: str
    audio_meta_csv: str | None
    lyrics_meta_csv: str | None
    out_dir: str


def _load_order_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    audio_col = find_column(df.columns, ["audio_song", "audio_id", "audio"], required=True)
    lyric_col = find_column(df.columns, ["lyric_song", "lyrics_song", "lyric_id", "lyrics_id", "lyric"], required=True)
    out = df[[audio_col, lyric_col]].copy()
    out.columns = ["Audio_Song", "Lyric_Song"]
    out["Audio_Song"] = out["Audio_Song"].astype(str).str.strip()
    out["Lyric_Song"] = out["Lyric_Song"].astype(str).str.strip()
    out = out.dropna().drop_duplicates(subset=["Audio_Song", "Lyric_Song"], keep="first").reset_index(drop=True)
    out["pair_index"] = range(len(out))
    return out


def _load_av_table(path: str, song_key: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    song_col = find_column(df.columns, ["song", "audio_song", "lyric_song", "identifier"], required=True)
    arousal_col = find_column(df.columns, ["arousal"], required=True)
    valence_col = find_column(df.columns, ["valence"], required=True)
    out = df[[song_col, arousal_col, valence_col]].copy()
    out.columns = [song_key, "Arousal", "Valence"]
    out[song_key] = out[song_key].astype(str).str.strip()
    out["Arousal"] = pd.to_numeric(out["Arousal"], errors="coerce")
    out["Valence"] = pd.to_numeric(out["Valence"], errors="coerce")
    out = out.dropna(subset=[song_key, "Arousal", "Valence"]).drop_duplicates(subset=[song_key], keep="first")
    return out


def _load_metadata_table(path: str, song_key: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    song_col = find_column(df.columns, ["song", "audio_song", "lyric_song", "identifier"], required=True)
    out = df.copy()
    out[song_col] = out[song_col].astype(str).str.strip()
    out = out.dropna(subset=[song_col]).drop_duplicates(subset=[song_col], keep="first")
    out = out.rename(columns={song_col: song_key})
    return out


def align_audio_lyrics_av(cfg: AlignConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    order_df = _load_order_table(cfg.order_csv)
    audio_df = _load_av_table(cfg.audio_csv, "Audio_Song")
    lyrics_df = _load_av_table(cfg.lyrics_csv, "Lyric_Song")

    merged = (
        order_df
        .merge(audio_df, on="Audio_Song", how="inner")
        .merge(lyrics_df, on="Lyric_Song", how="inner", suffixes=("_audio", "_lyrics"))
        .sort_values("pair_index", kind="stable")
        .reset_index(drop=True)
    )

    audio_aligned = merged[["Audio_Song", "Lyric_Song", "Arousal_audio", "Valence_audio"]].copy()
    audio_aligned.columns = ["Audio_Song", "Lyric_Song", "Arousal", "Valence"]

    lyrics_aligned = merged[["Audio_Song", "Lyric_Song", "Arousal_lyrics", "Valence_lyrics"]].copy()
    lyrics_aligned.columns = ["Audio_Song", "Lyric_Song", "Arousal", "Valence"]

    paired_summary = merged[
        [
            "pair_index",
            "Audio_Song",
            "Lyric_Song",
            "Arousal_audio",
            "Valence_audio",
            "Arousal_lyrics",
            "Valence_lyrics",
        ]
    ].copy()
    paired_summary.columns = [
        "pair_index",
        "Audio_Song",
        "Lyric_Song",
        "Audio_Arousal",
        "Audio_Valence",
        "Lyrics_Arousal",
        "Lyrics_Valence",
    ]
    return audio_aligned, lyrics_aligned, paired_summary


def align_audio_lyrics_metadata(cfg: AlignConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not cfg.audio_meta_csv or not cfg.lyrics_meta_csv:
        raise ValueError("Both audio_meta_csv and lyrics_meta_csv are required to align metadata.")

    order_df = _load_order_table(cfg.order_csv)
    audio_meta_df = _load_metadata_table(cfg.audio_meta_csv, "Audio_Song")
    lyrics_meta_df = _load_metadata_table(cfg.lyrics_meta_csv, "Lyric_Song")

    audio_merged = (
        order_df
        .merge(audio_meta_df, on="Audio_Song", how="inner")
        .sort_values("pair_index", kind="stable")
        .reset_index(drop=True)
    )
    lyrics_merged = (
        order_df
        .merge(lyrics_meta_df, on="Lyric_Song", how="inner")
        .sort_values("pair_index", kind="stable")
        .reset_index(drop=True)
    )

    audio_cols = ["Audio_Song", "Lyric_Song"] + [
        col for col in audio_merged.columns if col not in {"pair_index", "Audio_Song", "Lyric_Song"}
    ]
    lyrics_cols = ["Audio_Song", "Lyric_Song"] + [
        col for col in lyrics_merged.columns if col not in {"pair_index", "Audio_Song", "Lyric_Song"}
    ]
    return audio_merged[audio_cols].copy(), lyrics_merged[lyrics_cols].copy()


def save_outputs(
    audio_aligned: pd.DataFrame,
    lyrics_aligned: pd.DataFrame,
    paired_summary: pd.DataFrame,
    audio_meta_aligned: pd.DataFrame | None,
    lyrics_meta_aligned: pd.DataFrame | None,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    audio_out = os.path.join(out_dir, "aligned_audio_av.csv")
    lyrics_out = os.path.join(out_dir, "aligned_lyrics_av.csv")
    pairs_out = os.path.join(out_dir, "aligned_pairs_summary.csv")
    audio_aligned.to_csv(audio_out, index=False, encoding="utf-8-sig")
    lyrics_aligned.to_csv(lyrics_out, index=False, encoding="utf-8-sig")
    paired_summary.to_csv(pairs_out, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved aligned audio AV CSV:   {audio_out}")
    print(f"[OK] Saved aligned lyrics AV CSV:  {lyrics_out}")
    print(f"[OK] Saved pair summary CSV:       {pairs_out}")
    if audio_meta_aligned is not None and lyrics_meta_aligned is not None:
        audio_meta_out = os.path.join(out_dir, "aligned_audio_metadata.csv")
        lyrics_meta_out = os.path.join(out_dir, "aligned_lyrics_metadata.csv")
        audio_meta_aligned.to_csv(audio_meta_out, index=False, encoding="utf-8-sig")
        lyrics_meta_aligned.to_csv(lyrics_meta_out, index=False, encoding="utf-8-sig")
        print(f"[OK] Saved aligned audio metadata CSV:  {audio_meta_out}")
        print(f"[OK] Saved aligned lyrics metadata CSV: {lyrics_meta_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align MERGE audio and lyrics AV CSVs by a one-to-one Audio_Song <-> Lyric_Song mapping."
    )
    parser.add_argument("--order_csv", type=str, required=True, help="Path to order.csv")
    parser.add_argument("--audio_csv", type=str, required=True, help="Path to MERGE audio complete AV CSV")
    parser.add_argument("--lyrics_csv", type=str, required=True, help="Path to MERGE lyrics complete AV CSV")
    parser.add_argument("--audio_meta_csv", type=str, default=None, help="Path to MERGE audio complete metadata CSV")
    parser.add_argument("--lyrics_meta_csv", type=str, default=None, help="Path to MERGE lyrics complete metadata CSV")
    parser.add_argument("--out_dir", type=str, default="aligned_av_outputs", help="Output directory")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = AlignConfig(
        order_csv=str(args.order_csv),
        audio_csv=str(args.audio_csv),
        lyrics_csv=str(args.lyrics_csv),
        audio_meta_csv=str(args.audio_meta_csv) if args.audio_meta_csv else None,
        lyrics_meta_csv=str(args.lyrics_meta_csv) if args.lyrics_meta_csv else None,
        out_dir=str(args.out_dir),
    )
    audio_aligned, lyrics_aligned, paired_summary = align_audio_lyrics_av(cfg)
    audio_meta_aligned = None
    lyrics_meta_aligned = None
    if cfg.audio_meta_csv and cfg.lyrics_meta_csv:
        audio_meta_aligned, lyrics_meta_aligned = align_audio_lyrics_metadata(cfg)

    save_outputs(
        audio_aligned,
        lyrics_aligned,
        paired_summary,
        audio_meta_aligned,
        lyrics_meta_aligned,
        cfg.out_dir,
    )

    raw_order = len(_load_order_table(cfg.order_csv))
    print(f"[INFO] order pairs: {raw_order}")
    print(f"[INFO] kept aligned pairs: {len(audio_aligned)}")
    print(f"[INFO] dropped pairs: {raw_order - len(audio_aligned)}")


if __name__ == "__main__":
    main()

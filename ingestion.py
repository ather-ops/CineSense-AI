# =============================================================================
# CineSense AI — ingestion.py
# End-to-end pipeline: raw CSV → sentence chunking → embeddings → ChromaDB
# Author: ather-ops
# =============================================================================

# ── Step 1: Imports ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import nltk
from nltk.tokenize import sent_tokenize

from sentence_transformers import SentenceTransformer
import chromadb

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH    = "01-Data/netflix_titles.csv"
CHROMA_PATH  = "./chroma_data"
COLLECTION   = "netflix_titles"
EMBED_MODEL  = "all-MiniLM-L6-v2"
BATCH_SIZE   = 100
MAX_SENT     = 2
GREEN        = "#2ECC71"
SLATE        = "#2C3E50"
PURPLE       = "#EE22CC"


# ── Step 2: Load data ─────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        print(f"Dataset loaded — {len(df):,} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Error: {path} not found. Place the CSV in 01-Data/.")
        raise
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        raise


# ── Step 3: Clean missing values ──────────────────────────────────────────────
def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            fill_val = df[col].median() if "year" in col.lower() else df[col].mean()
            df[col] = df[col].fillna(fill_val)
        else:
            df[col] = df[col].fillna("unknown")
    return df


# ── Step 4: EDA visualizations ───────────────────────────────────────────────
def run_eda(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    # Content type distribution
    plt.figure(figsize=(8, 6))
    plt.pie(df["type"].value_counts(), labels=df["type"].value_counts().index, autopct="%1.1f%%")
    plt.title("Content Type: Movies vs TV Shows", fontweight="bold", color=SLATE)
    plt.savefig("04-Visuals/content_type.png", bbox_inches="tight")

    # Top 10 countries
    plt.figure(figsize=(10, 6))
    top_countries = df[df["country"] != "unknown"]["country"].value_counts().head(10)
    sns.barplot(x=top_countries.values, y=top_countries.index,
                hue=top_countries.index, palette="Greens", legend=False)
    plt.title("Top 10 Content-Producing Countries", fontweight="bold", color=SLATE)
    plt.xlabel("Number of Titles")
    plt.savefig("04-Visuals/top_countries.png", bbox_inches="tight")

    # Top 10 genres
    plt.figure(figsize=(10, 6))
    genre_counts = df["listed_in"].str.split(",").explode().value_counts().head(10)
    sns.barplot(x=genre_counts.values, y=genre_counts.index,
                hue=genre_counts.index, palette=f"dark:#{PURPLE[1:]}", legend=False)
    plt.title("Top 10 Most Popular Genres", fontweight="bold", color=SLATE)
    plt.xlabel("Count")
    plt.savefig("04-Visuals/top_genres.png", bbox_inches="tight")

    # Release growth over time
    plt.figure(figsize=(12, 6))
    yearly = df.groupby("release_year")["show_id"].count().reset_index()
    sns.lineplot(data=yearly, x="release_year", y="show_id",
                 marker="o", color=GREEN, linewidth=3)
    plt.fill_between(yearly["release_year"], yearly["show_id"], color=GREEN, alpha=0.15)
    plt.title("Content Release Growth Over the Years", fontweight="bold", color=SLATE)
    plt.xlabel("Year")
    plt.ylabel("Total Titles")
    plt.savefig("04-Visuals/release_growth.png", bbox_inches="tight")

    # Rating distribution
    plt.figure(figsize=(10, 6))
    rating_order = df["rating"].value_counts().index
    sns.countplot(data=df, x="rating", order=rating_order,
                  palette=f"dark:#{GREEN[1:]}", hue="rating", legend=False)
    plt.title("Audience Segment Distribution", fontweight="bold", color=SLATE)
    plt.xlabel("Rating Category")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("04-Visuals/rating_distribution.png", bbox_inches="tight")

    plt.show()
    print("EDA charts saved to 04-Visuals/")


# ── Step 5: Sentence-level chunking ──────────────────────────────────────────
def sentence_chunk(text: str, max_sentences: int = MAX_SENT) -> list[str]:
    """
    Split text into chunks of max_sentences complete sentences each.
    Uses NLTK sent_tokenize for proper sentence-boundary detection.
    """
    sentences = sent_tokenize(text)
    return [
        " ".join(sentences[i : i + max_sentences])
        for i in range(0, len(sentences), max_sentences)
    ]


def build_chunks(df: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """
    Build sentence-level chunks from combined per-row text fields.
    Returns parallel lists: chunk texts and their metadata dicts.
    """
    all_chunks:      list[str]  = []
    metadata_chunks: list[dict] = []

    for _, row in df.iterrows():
        combined = " ".join([
            str(row["title"])       if pd.notnull(row["title"])       else "",
            str(row["director"])    if pd.notnull(row["director"])    else "",
            str(row["cast"])        if pd.notnull(row["cast"])        else "",
            str(row["listed_in"])   if pd.notnull(row["listed_in"])   else "",
            str(row["description"]) if pd.notnull(row["description"]) else "",
        ]).strip()

        chunks = sentence_chunk(combined)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata_chunks.append({
                "show_id":      str(row["show_id"]),
                "title":        str(row["title"]),
                "type":         str(row["type"]),
                "country":      str(row["country"]),
                "release_year": int(row["release_year"]),
                "rating":       str(row["rating"]),
                "listed_in":    str(row["listed_in"]),
                "chunk_index":  idx,
                "total_chunks": len(chunks),
            })

    print(f"Chunking complete — {len(df):,} docs → {len(all_chunks):,} chunks")
    return all_chunks, metadata_chunks


# ── Step 6: Generate embeddings ───────────────────────────────────────────────
def generate_embeddings(chunks: list[str]):
    model = SentenceTransformer(EMBED_MODEL)
    print(f"Generating embeddings with {EMBED_MODEL}...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")
    return model, embeddings


# ── Step 7: Store in ChromaDB ─────────────────────────────────────────────────
def build_vector_store(chunks: list[str], metadata: list[dict], embeddings) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(name=COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"description": "Netflix movies and TV shows — CineSense AI"},
    )

    ids = [f"{m['show_id']}_chunk_{m['chunk_index']}" for m in metadata]

    for i in range(0, len(embeddings), BATCH_SIZE):
        end = i + BATCH_SIZE
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            metadatas=metadata[i:end],
            documents=chunks[i:end],
        )

    print(f"Inserted {len(chunks):,} chunks into ChromaDB at {CHROMA_PATH}")
    return collection


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  CineSense AI — Ingestion Pipeline")
    print("=" * 70)

    df                      = load_data(DATA_PATH)
    df                      = fill_missing(df)
    run_eda(df)
    chunks, metadata        = build_chunks(df)
    embed_model, embeddings = generate_embeddings(chunks)
    collection              = build_vector_store(chunks, metadata, embeddings)

    print("\n" + "=" * 70)
    print("  Ingestion complete. ChromaDB ready for rag_engine.py")
    print("=" * 70)

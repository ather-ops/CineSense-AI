print("=" * 80)
print("  CineSense AI — Day 5: Sentence-Level Chunking  |  by ather-ops")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Import libraries
# ─────────────────────────────────────────────────────────────────────────────
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

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Load data
# ─────────────────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv("netflix_titles.csv")
    print(f"\nDataset loaded — {len(df):,} rows, {df.shape[1]} columns")
    print(df.head())
except FileNotFoundError:
    print("Error: netflix_titles.csv not found. Place it in the working directory.")
    exit()
except Exception as e:
    print(f"Unexpected error loading data: {e}")
    exit()

print("\n" + "=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Basic EDA
# ─────────────────────────────────────────────────────────────────────────────
print("\nDataset info:")
print(df.info())
print("\nDescriptive statistics:")
print(df.describe())
print("\nMissing values per column:")
print(df.isnull().sum())
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print("\nColumns:", df.columns.tolist())

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Fill missing values
# ─────────────────────────────────────────────────────────────────────────────
def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            fill_val = df[col].median() if "year" in col.lower() else df[col].mean()
            df[col] = df[col].fillna(fill_val)
        else:
            df[col] = df[col].fillna("unknown")
    return df

df = fill_missing(df)
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: EDA visualizations
# ─────────────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
GREEN  = "#2ECC71"
PURPLE = "#EE22CC"
SLATE  = "#2C3E50"

# Graph 1 — Content type distribution
plt.figure(figsize=(8, 6))
plt.pie(df["type"].value_counts(), labels=df["type"].value_counts().index, autopct="%1.1f%%")
plt.title("Content Type: Movies vs TV Shows", fontweight="bold", color=SLATE)
plt.ylabel("Number of Titles")

# Graph 2 — Top 10 content-producing countries
plt.figure(figsize=(10, 6))
top_countries = df[df["country"] != "unknown"]["country"].value_counts().head(10)
sns.barplot(x=top_countries.values, y=top_countries.index,
            hue=top_countries.index, palette="Greens", legend=False)
plt.title("Top 10 Content-Producing Countries", fontweight="bold", color=SLATE)
plt.xlabel("Number of Titles")

# Graph 3 — Top 10 genres
plt.figure(figsize=(10, 6))
genre_counts = df["listed_in"].str.split(",").explode().value_counts().head(10)
sns.barplot(x=genre_counts.values, y=genre_counts.index,
            hue=genre_counts.index, palette="dark:#EE22CC", legend=False)
plt.title("Top 10 Most Popular Genres", fontweight="bold", color=SLATE)
plt.xlabel("Count")

# Graph 4 — Content release growth over time
plt.figure(figsize=(12, 6))
yearly = df.groupby("release_year")["show_id"].count().reset_index()
sns.lineplot(data=yearly, x="release_year", y="show_id",
             marker="o", color=GREEN, linewidth=3)
plt.fill_between(yearly["release_year"], yearly["show_id"], color=GREEN, alpha=0.15)
plt.title("Content Release Growth Over the Years", fontweight="bold", color=SLATE)
plt.xlabel("Year")
plt.ylabel("Total Titles")

# Graph 5 — Rating distribution
plt.figure(figsize=(10, 6))
rating_order = df["rating"].value_counts().index
sns.countplot(data=df, x="rating", order=rating_order,
              palette="dark:#2ECC71", hue="rating", legend=False)
plt.title("Audience Segment Distribution", fontweight="bold", color=SLATE)
plt.xlabel("Rating Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Sentence-level chunking  (Day 5 core upgrade)
# ─────────────────────────────────────────────────────────────────────────────
def sentence_chunk(text: str, max_sentences: int = 2) -> list[str]:
    """
    Split text into chunks of max_sentences sentences each.
    Uses NLTK sent_tokenize for proper sentence boundary detection.

    Args:
        text: The input string to chunk.
        max_sentences: Maximum number of sentences per chunk.

    Returns:
        List of text chunks.
    """
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i : i + max_sentences])
        chunks.append(chunk)
    return chunks


print("\n" + "=" * 60)
print("Step 6 — Building sentence-level chunks")
print("=" * 60)

all_chunks: list[str] = []
metadata_chunks: list[dict] = []

for _, row in df.iterrows():
    parts = [
        str(row["title"])       if pd.notnull(row["title"])       else "",
        str(row["director"])    if pd.notnull(row["director"])    else "",
        str(row["cast"])        if pd.notnull(row["cast"])        else "",
        str(row["listed_in"])   if pd.notnull(row["listed_in"])   else "",
        str(row["description"]) if pd.notnull(row["description"]) else "",
    ]
    combined = " ".join(parts).strip()

    chunks = sentence_chunk(combined, max_sentences=2)
    for chunk_idx, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        metadata_chunks.append({
            "show_id":      str(row["show_id"]),
            "title":        str(row["title"]),
            "type":         str(row["type"]),
            "country":      str(row["country"]),
            "release_year": int(row["release_year"]),
            "rating":       str(row["rating"]),
            "listed_in":    str(row["listed_in"]),
            "chunk_index":  chunk_idx,
            "total_chunks": len(chunks),
        })

print(f"Original documents : {len(df):,}")
print(f"Total chunks       : {len(all_chunks):,}")
avg_words = sum(len(c.split()) for c in all_chunks) / len(all_chunks)
print(f"Avg words per chunk: {avg_words:.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Generate embeddings
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 7 — Generating embeddings")
print("=" * 60)

model = SentenceTransformer("all-MiniLM-L6-v2")

try:
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    print(f"Embeddings generated — shape: {embeddings.shape}")
except Exception as e:
    print(f"Embedding error: {e}")
    exit()

# ─────────────────────────────────────────────────────────────────────────────
# Step 8: Initialise ChromaDB
# ─────────────────────────────────────────────────────────────────────────────
client = chromadb.Client()

try:
    client.delete_collection("netflix_titles")
except Exception:
    pass

collection = client.create_collection(name="netflix_titles")

# ─────────────────────────────────────────────────────────────────────────────
# Step 9: Batch insert into ChromaDB
# ─────────────────────────────────────────────────────────────────────────────
ids = [f"{meta['show_id']}_chunk_{meta['chunk_index']}" for meta in metadata_chunks]

BATCH_SIZE = 100
for i in range(0, len(embeddings), BATCH_SIZE):
    end = i + BATCH_SIZE
    collection.add(
        ids=ids[i:end],
        embeddings=embeddings[i:end].tolist(),
        metadatas=metadata_chunks[i:end],
        documents=all_chunks[i:end],
    )

print(f"\nInserted {len(all_chunks):,} chunks into ChromaDB successfully.")

# ─────────────────────────────────────────────────────────────────────────────
# Step 10: Advanced semantic search
# ─────────────────────────────────────────────────────────────────────────────
def advanced_netflix_search(
    collection,
    model,
    query_text: str,
    genre: str | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    rating: str | None = None,
    movie_type: str | None = None,
    top_k: int = 5,
):
    """
    Semantic search over the ChromaDB collection with optional metadata filters.

    Supports compound filters via ChromaDB's $and operator.
    Deduplicates results so each title appears only once in the output.
    Displays the matching chunk text for transparency.
    """
    query_emb = model.encode([query_text])[0]

    conditions = []
    if genre:
        conditions.append({"listed_in": {"$contains": genre}})
    if min_year:
        conditions.append({"release_year": {"$gte": min_year}})
    if max_year:
        conditions.append({"release_year": {"$lte": max_year}})
    if rating:
        conditions.append({"rating": {"$eq": rating}})
    if movie_type:
        conditions.append({"type": {"$eq": movie_type}})

    if len(conditions) > 1:
        where_filter = {"$and": conditions}
    elif len(conditions) == 1:
        where_filter = conditions[0]
    else:
        where_filter = None

    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        where=where_filter,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    print("\n" + "=" * 60)
    print(f"Query : {query_text}")
    print("=" * 60)

    if not results["metadatas"][0]:
        print("No results found. Try adjusting your filters.")
        return results

    seen_titles: set[str] = set()
    rank = 1
    for i, meta in enumerate(results["metadatas"][0]):
        if meta["title"] in seen_titles:
            continue
        seen_titles.add(meta["title"])
        chunk_preview = results["documents"][0][i][:150].replace("\n", " ")
        print(f"\n{rank}. {meta['title']}  ({meta['release_year']}) — {meta['rating']}")
        print(f"   Type   : {meta['type']}  |  Country: {meta['country']}")
        print(f"   Genre  : {meta['listed_in']}")
        print(f"   Chunk  : {chunk_preview}...")
        rank += 1

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Step 11: Run test queries
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 11 — Search Tests")
print("=" * 60)

advanced_netflix_search(collection, model, "Action Thriller", top_k=3)
advanced_netflix_search(collection, model, "Romantic Comedy", genre="Romantic", min_year=2020, top_k=3)
advanced_netflix_search(collection, model, "Documentary", rating="PG-13", movie_type="Movie", top_k=3)
advanced_netflix_search(collection, model, "Crime Drama", min_year=2019, max_year=2021, top_k=3)

# ─────────────────────────────────────────────────────────────────────────────
# Step 12: Save results to CSV
# ─────────────────────────────────────────────────────────────────────────────
def save_search_results(results, filename: str = "search_results.csv") -> None:
    if results["metadatas"][0]:
        pd.DataFrame(results["metadatas"][0]).to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")

save_search_results(
    advanced_netflix_search(collection, model, "Action Thriller", genre="Action", min_year=2020, top_k=5)
)

print("\n" + "=" * 80)
print("  Day 5 Complete — Sentence-level chunking pipeline fully operational")
print("=" * 80)

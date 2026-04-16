print("====================== CineSense AI next level gen AI tool by ather-ops ===================================")

# step 1: Import libraries
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# step 2: Load data with error handling
try:
    df = pd.read_csv("netflix_titles.csv")
    print("file loaded successfully")
    print("original data \n", df.head())
except FileNotFoundError:
    print("File not found check again")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()
print("=="*60)

# step 3: Basic EDA
print("Basic info:\n", df.info())
print("=="*60)
print("Basic Statistic:\n", df.describe())
print("=="*60)
print("Missing values :\n", df.isnull().sum())
print("=="*60)
print("Columns:\n", df.columns.tolist())
print("=="*60)
print("Duplicated values :\n", df.duplicated().sum())
print("=="*60)

# step 4: Filling missing values
def analysis(df):
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            if 'year' in col.lower():
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna("unknown")
    return df

analysis(df)
print("df with filled missing values:\n", df.head())
print("==" * 40)
print("Missing values AFTER analysis:\n", df.isnull().sum())
print("==" * 40)

# step 5: Initial visualisation
sns.set_theme(style="whitegrid")
primary_green = "#2ECC71"
secondary_green = "#EE22CC"
dark_slate = "#2C3E50"

# Graph 1: Distribution pie plot
plt.figure(figsize=(8,6))
plt.pie(df["type"].value_counts(), labels=df["type"].value_counts().index, autopct='%1.1f%%')
plt.title("Content Type: Movies vs TV Shows", fontweight="bold", color=dark_slate)
plt.ylabel("Number of Titles")

# Graph 2: Top countries bar plot
plt.figure(figsize=(10,6))
top_countries = df[df["country"] != "unknown"]["country"].value_counts().head(10)
sns.barplot(x=top_countries.values, y=top_countries.index, hue=top_countries.index, palette="Greens", legend=False)
plt.title('Top 10 Content-Producing Countries', fontweight='bold', color=dark_slate)
plt.xlabel('Number of Titles')

# Graph 3: Top genres bar plot 
plt.figure(figsize=(10,6))
genres = df["listed_in"].str.split(',').explode()
genre_counts = genres.value_counts().head(10)
sns.barplot(x=genre_counts.values, y=genre_counts.index, hue=genre_counts.index, palette='dark:#EE22CC', legend=False)
plt.title('Top 10 Most Popular Genres', fontweight='bold', color=dark_slate)
plt.xlabel('Count')

# Graph 4: Time Series line plot
plt.figure(figsize=(12, 6))
yearly_counts = df.groupby('release_year')['show_id'].count().reset_index()
sns.lineplot(data=yearly_counts, x='release_year', y='show_id', marker='o', color=primary_green, linewidth=3)
plt.fill_between(yearly_counts['release_year'], yearly_counts['show_id'], color=primary_green, alpha=0.15)
plt.title('Content Release Growth Over the Years', fontweight='bold', color=dark_slate)
plt.xlabel('Year')
plt.ylabel('Total Titles')

# Graph 5: Rating distribution 
plt.figure(figsize=(10, 6))
rating_order = df['rating'].value_counts().index
sns.countplot(data=df, x='rating', order=rating_order, palette='dark:#2ECC71', hue='rating', legend=False)
plt.title('Audience Segment Distribution', fontweight='bold', color=dark_slate)
plt.xlabel('Rating Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# step 6: Create combined text for embeddings
df["combined_txt"] = (df["title"] + " " + df["director"] + " " + df["cast"] + " " + df["listed_in"] + " " + df["description"])

# step 7: Model selection and embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
try:
    print("Generating embeddings. This may take a few minutes...")
    embeddings = model.encode(df["combined_txt"].tolist(), show_progress_bar=True)
    print("Embeddings generated successfully!")
except Exception as e:
    print(f"Error generating embeddings: {e}")
    exit()

# step 8: Initialise chromadb
client = chromadb.PersistentClient(path="./03-Core/chroma_db")

# step 9: Creating collections
try:
    client.delete_collection("netflix_titles")
except:
    pass

collection = client.create_collection(name="netflix_titles")

# step 10: Prepare Data and Insert
ids = df["show_id"].tolist()
metadatas = []
for idx, row in df.iterrows():
    try:
        added_year = int(str(row["date_added"]).split(",")[-1].strip())
    except:
        added_year = row["release_year"]
    metadatas.append({
        "title": row["title"],
        "type": row["type"],
        "country": row["country"],
        "release_year": row["release_year"],
        "added_year": added_year,
        "rating": row["rating"],
        "listed_in": row["listed_in"],
        "description": row["description"]
    })

documents = df["description"].fillna("no description").tolist()

# step 11: Batch insert
batch_size = 100
for i in range(0, len(embeddings), batch_size):
    end = i + batch_size
    collection.add(
        ids=ids[i:end],
        embeddings=embeddings[i:end].tolist(),
        metadatas=metadatas[i:end],
        documents=documents[i:end]
    )
print("Data inserted into ChromaDB successfully!")

# step 12: Advanced search function
def advanced_netflix_search(collection, model, query_text, genre=None, min_year=None, max_year=None, rating=None, movie_type=None, top_k=5):
    query_emb = model.encode([query_text])[0]
    conditions = []

    if genre:
        conditions.append({"listed_in": {"$contains": genre}})
    if min_year:
        conditions.append({"release_year": {"$gte": min_year}})
    if max_year:
        conditions.append({"release_year": {"$lte": max_year}})
    if rating:
        conditions.append({"rating": rating})
    if movie_type:
        conditions.append({"type": movie_type})

    if len(conditions) > 1:
        where_filter = {"$and": conditions}
    elif len(conditions) == 1:
        where_filter = conditions[0]
    else:
        where_filter = None

    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        where=where_filter,
        n_results=top_k
    )

    print("\n" + "="*60)
    print(f"Search Query: {query_text}")
    print("="*60)

    if results["metadatas"][0]:
        for i, meta in enumerate(results["metadatas"][0]):
            print(f"\n{i+1}. Title: {meta['title']}")
            print(f"   Year: {meta['release_year']} | Rating: {meta['rating']}")
            print(f"   Type: {meta['type']} | Country: {meta['country']}")
            print(f"   Genre: {meta['listed_in']}")
            print(f"   Description: {meta['description'][:150]}...")
    else:
        print("No results found. Try adjusting your filters.")

    return results

# step 13: Test different searches
print("\n" + "="*60)
print("TESTING SEARCH FUNCTIONALITY")
print("="*60)

# Test 1: Basic search
advanced_netflix_search(collection, model, "Action Thriller", top_k=3)

# Test 2: Filtered search
advanced_netflix_search(collection, model, "Romantic Comedy", genre="Romantic", min_year=2020, top_k=3)

# Test 3: Specific rating search
advanced_netflix_search(collection, model, "Documentary", rating="PG-13", movie_type="Movie", top_k=3)

# Test 4: Year range search
advanced_netflix_search(collection, model, "Crime Drama", min_year=2019, max_year=2021, top_k=3)

print("\n" + "="*60)
print("Day 3 Completed Successfully!")
print("="*60)

# Optional: Save results to CSV
def save_search_results(results, filename="search_results.csv"):
    if results["metadatas"][0]:
        df_results = pd.DataFrame(results["metadatas"][0])
        df_results.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

# Save last search results
save_search_results(advanced_netflix_search(collection, model, "Action Thriller", genre="Action", min_year=2020, top_k=5))

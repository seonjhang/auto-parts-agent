import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load a free, local embedding model
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Load data
data = pd.read_csv("auto_parts_catalog.csv")

# Combine part_name + description for richer embeddings
texts = (data["part_name"] + "." + data["description"]).tolist()

# Keep the rest of each row as metadat - this is for later, in order to filter/display results
metadatas = data.to_dict(orient="records")

# 3. Build the vector DB with all texts
db = FAISS.from_texts(texts, embeddings_model, metadatas=metadatas) 
description = data.iloc[:,-1]

# 4. Save
db.save_local("my_faiss")


# Testing with similary search
# results = db.similarity_search("brake pads for 2018 Honda Civic", k=3)
# for r in results:
#     print(r.page_content, "->", r.metadata["price_usd"])

display(Image(chain.get_graph().draw_mermaid_png()))
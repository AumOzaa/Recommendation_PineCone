from fastapi import FastAPI
from pinecone import Pinecone
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from fastapi import Body

app = FastAPI()
load_dotenv()

@app.post("/create_post_embedding" , status_code=201)
async def create_embeddings(payload: dict = Body(...)):
        print(payload.get("content"))
        print(payload.get("id"))
        id = payload.get("id")
        content = payload.get("content")

        pc = Pinecone(api_key=os.getenv("pinecone_api_key"))
        
        index_name = "social-media"

        # if index_name not in pc.list_indexes().names():
        #     pc.create_index(
        #     name=index_name,
        #     dimension=384, 
        #     metric="cosine",
        #     spec=ServerlessSpec(cloud="aws", region="us-east-1")
        # )


        index = pc.Index(index_name)
        model = SentenceTransformer('BAAI/bge-small-en-v1.5')

        embeddings = model.encode([content]).tolist() # Convert to list for Pinecone

        vectors_to_upsert = [
        {
            "id" : id,
            "values" : embeddings[0],
            "metadata" : {
            "text"  : content,
            }
        },
    ]

        index.upsert(vectors=vectors_to_upsert)

        return {"vector": embeddings[0]}

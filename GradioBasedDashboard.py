import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

import gradio as gr
import os

load_dotenv()

books= pd.read_csv('booksWithEmotions.csv')
books['large_thumbnail']= books['thumbnail']+ "&fife=w800"
books['large_thumbnail']= np.where(books['large_thumbnail'].isna(),"coverNotFound.jpg",books['large_thumbnail'])

rawDouments= TextLoader("tagged_description.txt", encoding='utf-8').load()
textSplitter= CharacterTextSplitter(separator='\n', chunk_size=1, chunk_overlap=0)
documents= textSplitter.split_documents(rawDouments)

embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
dbBooks= Chroma.from_documents(documents, embedding=embedding)

port = int(os.environ.get("PORT", 7860))

def retreiveSemanticRecommendation(
        query: str,
        category: str=None,
        tone: str=None,
        initialTopK: int=50,
        finalTopK: int=16
)->pd.DataFrame:
    
    recs= dbBooks.similarity_search(query, k=initialTopK)
    booksList= [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    booksRecs= books[books['isbn13'].isin(booksList)].head(initialTopK)

    if category != "All":
        booksRecs= booksRecs[booksRecs['simpleCategories']==category].head(finalTopK)
    else:
        booksRecs= booksRecs.head(finalTopK)

    if tone == "Happy":
        booksRecs.sort_values(by='joy', ascending=False, inplace=True)
    elif tone == "Surprising":
        booksRecs.sort_values(by='surprise', ascending=False, inplace=True)
    elif tone == "Angry":
        booksRecs.sort_values(by='anger', ascending=False, inplace=True)
    elif tone == ["Suspenseful", "fear"]:
        booksRecs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        booksRecs.sort_values(by="sadness", ascending=False, inplace=True)
    elif tone == "Disgusting":
        booksRecs.sort_values(by="disgust", ascending=False, inplace=True)
    
    return booksRecs

def recommedBooks(
        query: str,
        category: str,
        tone: str
):
    recommendations= retreiveSemanticRecommendation(query, category, tone)
    result= []

    for _,row in recommendations.iterrows():
        description= row['description']
        truncatedDescSplit= description.split()
        truncatedDescription= " ".join(truncatedDescSplit[:30]) + "..."

        authersSplit= row['authors'].split(";")
        if len(authersSplit) == 2:
            authersStr= authersSplit[0]+ " and "+ authersSplit[1]
        elif len(authersSplit)>2:
            authersStr= f"{', '.join(authersSplit[:-1])}, and {authersSplit[-1]}"
        else:
            authersStr= row["authors"]
        
        caption= f"{row['title']} by {authersStr}: {truncatedDescription}"
        result.append((row['large_thumbnail'],caption))

    return result

categories= ['All']+ sorted(books["simpleCategories"].unique())
tones= ['All']+ ['Happy', 'Surprising', 'Angry', 'Suspenseful', 'Sad']

with gr.Blocks() as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        userQuery= gr.Textbox(label="Please enter description of the Book:", placeholder='eg: A Story about forgiveness')
        categoryDropdown= gr.Dropdown(choices= categories, label='Select a Category', value= 'All')
        toneDropdown= gr.Dropdown(choices= tones, label='Select a Tone', value= 'All')
        submitButton= gr.Button('Find Recommendations')

    gr.Markdown('## Recommendations')
    output= gr.Gallery(label= "Recommended Books", columns= 8, rows= 2)

    submitButton.click(fn= recommedBooks,
                    inputs= [userQuery, categoryDropdown, toneDropdown],
                    outputs= output)
    
if __name__ == "__main__":
    dashboard.launch(theme=gr.themes.Soft(), share=True, server_port=port)
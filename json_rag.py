import json
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os

print("📚 JSON RAG with ChromaDB Starting...")

# 1. JSON dosyasını oku
print("📖 Loading book data...")
with open('book_data.json', 'r', encoding='utf-8') as f:
    book_data = json.load(f)

# 2. JSON'dan Document'ler oluştur (Section-based chunking)
documents = []
for chapter in book_data['chapters']:
    chapter_num = chapter['chapter_num']
    chapter_title = chapter['title']
    
    for section in chapter['sections']:
        # Her section bir chunk
        doc = Document(
            page_content=section['content'],
            metadata={
                'chapter_num': chapter_num,
                'chapter_title': chapter_title,
                'section_title': section['section_title'],
                'page': section['page'],
                'book_title': book_data['title']
            }
        )
        documents.append(doc)

print(f"✅ Created {len(documents)} chunks (section-based)")

# 3. OpenAI API Key
api_key = input("Enter OpenAI API Key: ")
os.environ["OPENAI_API_KEY"] = api_key

# 4. Embeddings oluştur
print("🔢 Creating embeddings...")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# 5. ChromaDB'ye kaydet
print("💾 Storing in ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("✅ ChromaDB ready!")

# 6. Test query
print("\n" + "="*50)
print("🔍 TEST QUERY")
print("="*50)

query = "What is the difference between Qdrant and Pinecone?"
print(f"Query: {query}\n")

# 7. Semantic search
results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results, 1):
    print(f"\n📄 Result {i}:")
    print(f"Chapter: {doc.metadata['chapter_num']} - {doc.metadata['chapter_title']}")
    print(f"Section: {doc.metadata['section_title']}")
    print(f"Page: {doc.metadata['page']}")
    print(f"Content: {doc.page_content[:150]}...")

# 8. Metadata filtering test
print("\n" + "="*50)
print("🔍 METADATA FILTERING TEST")
print("="*50)

query2 = "chunking"
print(f"Query: {query2}")
print(f"Filter: Chapter 3 only\n")

# ChromaDB'de metadata filtering
results_filtered = vectorstore.similarity_search(
    query2,
    k=2,
    filter={"chapter_num": 3}
)

for i, doc in enumerate(results_filtered, 1):
    print(f"\n📄 Result {i}:")
    print(f"Chapter: {doc.metadata['chapter_num']} - {doc.metadata['chapter_title']}")
    print(f"Section: {doc.metadata['section_title']}")
    print(f"Content: {doc.page_content[:100]}...")

print("\n✅ JSON RAG Demo Complete!")

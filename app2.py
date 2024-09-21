import ollama
import chromadb
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from crewai import Agent, Task, Crew
import sys
import os

# Initialize the LLaMA model
ollama_llama3 = Ollama(model='llama3')

# Initialize ChromaDB for embeddings
client = chromadb.Client()
collection = client.create_collection(name="docs")

def get_embedding(text):
    response = ollama.embeddings(
        model='nomic-embed-text',
        prompt=text
    )
    return response['embedding']

def load_and_index_pdf(pdf_path):
    # Load PDF and extract text
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    # Create and store embeddings
    for i, doc in enumerate(chunks):
        # Check available attributes
        if hasattr(doc, 'page_content'):
            chunk_text = doc.page_content
        else:
            # Fallback if 'page_content' is not available
            chunk_text = str(doc)
        embedding = get_embedding(chunk_text)
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[chunk_text]
        )

def query_collection(prompt):
    # Generate embedding for the query prompt
    response = ollama.embeddings(
        model='nomic-embed-text',
        prompt=prompt
    )
    query_embedding = response['embedding']
    
    # Retrieve the most relevant document
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )
    return results['documents'][0][0]

def info_pmb(topic):
    # Create Researcher agent
    researcher = Agent(
        role='Ahli Analis Informasi PMB Undiksha',
        goal=f"Temukan informasi {topic}",
        backstory="""Anda bekerja di sebuah lembaga pendidikan di Singaraja, Bali yang terkenal dengan kota pendidikan yaitu Universitas Pendidikan Ganesha (Undiksha). Keahlian Anda terletak pada mengidentifikasi informasi terkait Penerimaan Mahasiswa Baru (PMB) di Undiksha. Anda memiliki bakat untuk membedah data yang kompleks dan menyajikan wawasan yang dapat ditindaklanjuti. Tetapi jika tidak ada informasi yang sesuai, katakanlah bahwa informasi tidak ditemukan.""",
        verbose=True,
        allow_delegation=False,
        llm=ollama_llama3
    )

    # Create Writer agent
    writer = Agent(
        role='Ahli Penulis Informasi PMB Undiksha',
        goal=f"Berikan informasi PMB Undiksha {topic}",
        backstory="""Anda adalah seorang penulis yang berpengalaman dalam menulis seputar informasi tentang sesuatu hal. Anda memiliki kemampuan untuk mengubah informasi yang kompleks menjadi tulisan yang mudah dipahami. Anda dapat membuat informasi yang menarik dan aktual tentang informasi Penerimaan Mahasiswa Baru (PMB) di Undiksha. Tetapi jika tidak ada informasi yang sesuai, katakanlah bahwa informasi tidak ditemukan.""",
        verbose=True,
        allow_delegation=False,
        llm=ollama_llama3
    )

    # Create tasks for your agents
    task1 = Task(
        description=f"Lakukan analisis komprehensif tentang {topic}.",
        expected_output="Laporan analisis lengkap dalam poin-poin penting",
        agent=researcher
    )

    task2 = Task(
        description=f"""Dengan menggunakan wawasan yang diberikan, kembangkan 
        informasi yang menyoroti topik {topic}.
        Informasi Anda harus informatif namun mudah diakses dan mudah dipahami. 
        Buatlah dengan aktual dan bagus, hindari kata-kata yang rumit sehingga tidak terdengar seperti AI.""",
        expected_output="Informasi yang menyoroti topik PMB Undiksha sesuai dengan topik yang diberikan.",
        agent=writer
    )

    # Instantiate your crew with a sequential process
    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        verbose=2, # You can set it to 1 or 2 to different logging levels
    )

    # Get your crew to work!
    result = crew.kickoff()
    return result

def main():
    if len(sys.argv) < 2:
        print("Silakan tanyakan sesuatu tentang PMB di Undiksha.")
        print('Penggunaan: python app.py "Dimana Kampus Undiksha?"')
        sys.exit(1)

    # Get the first command-line argument after the script name
    topic = sys.argv[1]

    # Now you can use 'argument' in your script
    print("\n\n#### Topic ####")
    print(topic)

    if topic == '':
        print("Topic is empty.")
        sys.exit(1)

    # Load and index PDF (make sure to provide the correct path to your PDF)
    pdf_path = 'dataset.pdf'
    if os.path.exists(pdf_path):
        load_and_index_pdf(pdf_path)
    
    # Retrieve and generate response
    result = query_collection(topic)

    print("\n\n#### Result ####")
    print(result)

if __name__ == '__main__':
    main()

import chromadb
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew
import sys
from sentence_transformers import SentenceTransformer

# Initialize Ollama LLaMA3 model
ollama_llama3 = Ollama(model='llama3')

def info_pmb(topic):
    # Create Researcher agent
    researcher = Agent(
        role='Ahli Analis Informasi',
        goal=f"Temukan informasi {topic}",
        backstory="""Anda bekerja di sebuah lembaga pendidikan di Singaraja, Bali yang terkenal dengan kota pendidikan yaitu Universitas Pendidikan Ganesha (Undiksha). Keahlian Anda terletak pada mengidentifikasi informasi terkait Penerimaan Mahasiswa Baru (PMB) di Undiksha. Anda memiliki bakat untuk membedah data yang kompleks dan menyajikan wawasan yang dapat ditindaklanjuti.""",
        verbose=True,
        allow_delegation=False,
        llm=ollama_llama3
    )

    # Create Writer agent
    writer = Agent(
        role='Ahli Penulis',
        goal=f"Berikan informasi PMB Undiksha {topic}",
        backstory="""Anda adalah seorang penulis yang berpengalaman dalam menulis seputar informasi tentang sesuatu hal. Anda memiliki kemampuan untuk mengubah informasi yang kompleks menjadi tulisan yang mudah dipahami. Anda dapat membuat informasi yang menarik dan aktual tentang informasi Penerimaan Mahasiswa Baru (PMB) di Undiksha.""",
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
        verbose=2,  # You can set it to 1 or 2 for different logging levels
    )

    # Query ChromaDB
    client = chromadb.Client()
    collection = client.get_or_create_collection("pmb_undiksha")

    # Use SentenceTransformer to get embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(topic)

    # Query ChromaDB
    results = collection.query(query_embedding, n_results=5)

    # Extract relevant text from ChromaDB
    relevant_text = ""
    for result in results['matches']:
        relevant_text += result['metadata']['text']

    # Modify prompt for LLM
    new_prompt = f"Gunakan informasi berikut untuk menjawab pertanyaan: {relevant_text}\nPertanyaan: {topic}"

    # Set the new prompt for Writer agent
    writer.llm.prompt = new_prompt

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

    # Print the topic
    print("\n\n#### Topic ####")
    print(topic)

    if topic.strip() == '':
        print("Topic is empty.")
        sys.exit(1)

    # Get the information about PMB
    result = info_pmb(topic)

    # Print the result
    print("\n\n#### Result ####")
    print(result)

if __name__ == '__main__':
    main()

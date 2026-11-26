# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,TOOLS_pip_installation
# MAGIC %pip install -U langchain-community langchain-core langchain-text-splitters langchain-huggingface chromadb sentence-transformers transformers torch

# COMMAND ----------

# DBTITLE 1,import Libraries
# Create documents
from langchain_core.documents import Document

# Chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Hugging Face embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Chroma vector database
from langchain_community.vectorstores import Chroma

# Hugging Face language model
from transformers import pipeline

# COMMAND ----------

# DBTITLE 1,office knowledge database
office_documents = [
    Document(
        page_content="Office working hours are Monday to Friday, from 8 AM to 5 PM.",
        metadata={"source": "Office Hours"}
    ),

    Document(
        page_content="Employees must submit leave requests to their manager at least 3 days before taking leave.",
        metadata={"source": "Leave Policy"}
    ),

    Document(
        page_content="Employees who need a new laptop must submit an equipment request to the IT department.",
        metadata={"source": "Equipment Policy"}
    ),

    Document(
        page_content="For IT support, employees must email support@office.com and include a description of the problem.",
        metadata={"source": "IT Support"}
    ),

    Document(
        page_content="Meeting rooms can be booked through the office calendar. Employees should cancel bookings they no longer need.",
        metadata={"source": "Meeting Room Policy"}
    )
]

print("Number of office documents:", len(office_documents))

# COMMAND ----------

# DBTITLE 1,Chunk the documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)

chunks = text_splitter.split_documents(office_documents)

print("Number of chunks:", len(chunks))

# COMMAND ----------

# MAGIC %md
# MAGIC chunk_size=200: Each chunk can contain up to about 200 characters.
# MAGIC chunk_overlap=30: Adjacent chunks can share 30 characters.
# MAGIC split_documents(): Breaks the documents into smaller pieces.

# COMMAND ----------

# DBTITLE 1,looking at the chunks
for chunk in chunks:
    print(chunk.page_content)
    print("----------------")

# COMMAND ----------

# DBTITLE 1,embeddings using Hugging Face
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# COMMAND ----------

# DBTITLE 1,test the convertion
test_embedding = embedding_model.embed_query(
    "What time does the office open?"
)

print(test_embedding[:10])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC These numbers represent the meaning of the question. The embedding model will also convert our office document chunks into vectors so we can compare them with the question

# COMMAND ----------

# DBTITLE 1,Chroma vector database
#Now we store the chunks and their embeddings in Chroma.
vector_database = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model
)

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Chroma receives our office chunks.
# MAGIC 2. The embedding model converts each chunk into a vector.
# MAGIC 3. Chroma stores the vectors and the original text.
# MAGIC 4. We can now search for information based on meaning.

# COMMAND ----------

# DBTITLE 1,the retriever
#The retriever searches the vector database and returns the most relevant chunks.
retriever = vector_database.as_retriever(
    search_kwargs={"k": 2} #2most relevent chunks
)

# COMMAND ----------

# DBTITLE 1,MORE TESTING
question = "What time does the office open?"

relevant_documents = retriever.invoke(question)

for document in relevant_documents:
    print(document.page_content)
    print(document.metadata)

# COMMAND ----------

# MAGIC %md
# MAGIC The retriever should find the office hours information.
# MAGIC
# MAGIC The important point is that it searches by meaning, not only by exact words.

# COMMAND ----------

# DBTITLE 1,Hugging Face language model
#The retriever finds the information, but we still need an LLM to write the answer.
#Note: the "text2text-generation" pipeline task was removed in transformers v5,
#so we load the flan-t5 model directly instead.
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def llm(prompt, max_new_tokens=150):
    inputs = llm_tokenizer(prompt, return_tensors="pt")
    output_ids = llm_model.generate(**inputs, max_new_tokens=max_new_tokens)
    return llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)

# COMMAND ----------

#Build the RAG question-answering function.This part Now we connect everything together.
def ask_office_question(question):

    # Search for relevant office information
    relevant_documents = retriever.invoke(question)

    # Combine the retrieved information
    context = "\n".join(
        document.page_content
        for document in relevant_documents
    )

    # Create the prompt
    prompt = f"""
    Answer the question using only the information below.

    If the answer is not in the information, say:
    I don't know based on the office documents.

    Information:
    {context}

    Question:
    {question}

    Answer:
    """

    # Generate the answer
    answer = llm(prompt)

    return answer

# COMMAND ----------

# MAGIC %md
# MAGIC 1. The user asks a question.
# MAGIC 2. The retriever searches Chroma.
# MAGIC 3. The relevant text is placed into context.
# MAGIC 4. The prompt sends the context and question to the LLM.
# MAGIC 5. The LLM generates an answer.
# MAGIC This is the main RAG process: retrieve relevant information, then generate an answer using that information.

# COMMAND ----------

#Ask your office RAG system questions
answer = ask_office_question(
    "What time does the office open?"
)

print(answer)

# COMMAND ----------

# DBTITLE 1,another Question
print(
    ask_office_question(
        "How do I request a new laptop?"
    )
)

# COMMAND ----------

#Try asking something that is not in the database:
print(
    ask_office_question(
        "Who is the office manager?"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC
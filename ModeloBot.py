from langchain_groq import ChatGroq
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
import os
load_dotenv()

#set_debug(True)
#set_verbose(True)


class OrganizadorPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, llm_model: str = "llama3-8b-8192"): #Se usa el modelo llama3 porque maneja la mayor cantidad de tokens por minuto
        self.model = ChatGroq(model= llm_model, temperature=0.5)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000, chunk_overlap=800
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Eres un asistente de IA que ayuda al usuario a crear planes de estudio.
                    Un plan de estudio es la estructura de tiempo de lectura y análisis de un documento específico.
                    Objetivo: El plan debe ajustarse a las necesidades y objetivos del estudiante, maximizando el uso eficiente de su tiempo disponible.

                    Criterios de éxito:

                    El plan se ajusta a los tiempos estipulados que el estudiante tiene disponibles.
                    El plan de lectura se genera si el contenido es estrictamente literario.

                    El resultado debe ser una lista con el tema a estudiar, el tiempo asignado a estudio y el número de página donde se encuentra la información
                    """,
                ),
                (
                    "human",
                    "Estos son los documentos que necesito que analices: {context}\n",
                ),
                (
                    "ai",
                    "Genial! Ahora dime el tiempo estipulado para el plan de estudio"
                ),
                (
                    "human",
                    "Tiempo: {question}"
                )
            ]
        )

        self.vector_store = None
        self.retriever = None
        self.chain = None
        memory = MemorySaver()

    def ingest(self, pdf_file_path: str): #funcion para cargar los datos
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000,
            chunk_overlap=800 #10%
        )
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        # La vectorstore se crea en FAISS como base de datos vectorial, y el embedding es el de Cohere
        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=CohereEmbeddings(model="embed-multilingual-v3.0")
        )

    def ask(self, query: str): #funcion para hacer las preguntas
        if not self.vector_store:
            self.vector_store = FAISS.from_documents(
            embedding=CohereEmbeddings(model="embed-multilingual-v3.0")
        )
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.0},
        )

        self.retriever.invoke(query)

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        if not self.chain:
            return "sube un archivo PDF primero, por favor."

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None

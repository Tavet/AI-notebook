# Warning control
import warnings
warnings.filterwarnings('ignore')

from unstructured_client.models import shared
from unstructured_client import UnstructuredClient
from unstructured_client.models.errors import SDKError
from unstructured.staging.base import dict_to_elements
from unstructured.chunking.title import chunk_by_title

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from io import StringIO
from lxml import etree

import os
from Utils import Utils

utils = Utils()

DLAI_API_KEY = utils.get_uc_api_key()
os.environ["OPENAI_API_KEY"] = utils.get_openai_api_key()

s = UnstructuredClient(
    api_key_auth=DLAI_API_KEY
)


filename = "/Users/breynerrojas/Documents/Personal/tavet/AI-notebook/LLM/RAG-Bot/files/Breyner_CV.pdf"

with open(filename, "rb") as f:
    files=shared.Files(
        content=f.read(),
        file_name=filename,
    )

req = shared.PartitionParameters(
    files=files,
    strategy="hi_res",
    hi_res_model_name="yolox",
    pdf_infer_table_structure=True,
    skip_infer_table_types=[],
)

try:
    resp = s.general.partition(req)
    pdf_elements = dict_to_elements(resp.elements)
except SDKError as e:
    print(e)

pdf_elements[0].to_dict()

tables = [el for el in pdf_elements if el.category == "Table"]

table_html = tables[0].metadata.text_as_html

parser = etree.XMLParser(remove_blank_text=True)
file_obj = StringIO(table_html)
tree = etree.parse(file_obj, parser)

elements = chunk_by_title(pdf_elements)

documents = []
for element in elements:
    metadata = element.metadata.to_dict()
    del metadata["languages"]
    metadata["source"] = metadata["filename"]
    documents.append(Document(page_content=element.text, metadata=metadata))

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

template = """You are an AI assistant for answering questions about the provided CV.
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about the professional experience of the person in the CV, politely inform them that you are tuned to only answer questions about the CV.
Question: {question}
Context:
=========
{context}
========="""
prompt = PromptTemplate(template=template, input_variables=["question", "context"])

llm = OpenAI(temperature=0)

doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")
question_generator_chain = LLMChain(llm=llm, prompt=prompt)
qa_chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator_chain,
    combine_docs_chain=doc_chain,
)

print(qa_chain.invoke({
    "question": "What Breyner did as a Software Engineer at Dell Technologies?",
    "chat_history": []
})["answer"])

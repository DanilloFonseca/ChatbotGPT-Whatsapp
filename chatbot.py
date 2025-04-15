import time
import flask
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document


load_dotenv()
app = flask.Flask(__name__)


api_key = os.getenv('OPEN_API_KEY')
if not api_key:
    raise ValueError("A chave de API do OpenAI não foi encontrada. Verifique o teu arquivo .env.")
print("Chave de API carregada com sucesso!")


def create_embeddings_with_retry(documents, embeddings, retries=5, wait_time=10):
    for attempt in range(retries):
        try:
            print(f"Tentando criar embeddings... tentativa {attempt + 1}")
            return FAISS.from_documents(documents, embeddings)
        except Exception as e:
            print(f"[Erro tentativa {attempt + 1}] Tipo: {type(e).__name__}, Detalhes: {e}")
            time.sleep(wait_time)
    raise Exception("Falha ao criar embeddings após várias tentativas.")


def split_documents(documents, batch_size=5):
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]


def testar_embedding_simples(embedding):
    try:
        doc = Document(page_content="Olá, sou um teste do PET-CC!")
        FAISS.from_documents([doc], embedding)
        print("Teste de embedding simples executado com sucesso!")
    except Exception as e:
        print("Falha no teste de embedding:", e)
        raise


print("Carregando documentos do CSV...")
loaderCsv = CSVLoader(file_path='chatbotPetcc.csv')
documents = loaderCsv.load()
print(f"Total de documentos carregados: {len(documents)}")

embeddings = OpenAIEmbeddings(openai_api_key=api_key)
testar_embedding_simples(embeddings)

print("Criando embeddings com retry e FAISS em lotes...")
all_vector_stores = []
for idx, batch in enumerate(split_documents(documents, batch_size=5)):
    print(f"Processando lote {idx + 1}")
    store = create_embeddings_with_retry(batch, embeddings)
    all_vector_stores.append(store)

print("Mesclando todos os vetores em um único FAISS...")
vector_store = FAISS.merge_from(all_vector_stores)

llm = ChatOpenAI(openai_api_key=api_key)

template = (
    "Você é o mascote do Programa de Educação Tutorial da UFRN (Pet-CC/UFRN). "
    "Use o seguinte contexto para responder perguntas sobre o PET-CC.\n\n"
    "Contexto: {context}\n"
    "Pergunta: {pergunta}"
)
prompt = ChatPromptTemplate.from_template(template)

retriever = vector_store.as_retriever()
chain = (
    {"context": retriever, "pergunta": RunnablePassthrough()}
    | prompt
    | llm
)

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = flask.request.json
        if not data or 'message' not in data:
            return flask.jsonify({"error": "Corpo inválido: 'message' não encontrado."}), 400
        print(f"Mensagem recebida: {data['message']}")
        response = chain.invoke(data['message'])
        return flask.jsonify({"response": response.content})
    except Exception as e:
        print(f"Erro no webhook: {e}")
        return flask.jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)

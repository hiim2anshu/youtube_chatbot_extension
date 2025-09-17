from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)
CORS(app)

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "open_ai_api_token_key"  # replace with your actual key

@app.route('/process-video', methods=['POST'])
def process_video():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json(silent=True)
    video_id = data.get('video_id')
    prompt_text = data.get('prompt')

    if not video_id or not prompt_text:
        return jsonify({"error": "Video ID and prompt are required"}), 400
    


    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id, languages=["en"])
        raw_transcript = fetched.to_raw_data()
        transcript = " ".join(entry["text"] for entry in raw_transcript)
        # print(transcript)

    except TranscriptsDisabled:
        print("No captions available for this video.")
    except NoTranscriptFound:
        print("No transcript found in the requested language.")
    except VideoUnavailable:
        print("The video is unavailable.")
    except Exception as e:
        print("An unexpected error occurred:", str(e))


    # Split transcript into chunks
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])
    except Exception as e:
        print("Error splitting transcript:", e)
        return jsonify({"error": "Failed to split transcript"}), 500

    # Create embeddings and vector store
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    except Exception as e:
        print("Error creating FAISS vector store:", e)
        return jsonify({"error": "Failed to process transcript"}), 500

    # LLM and prompt template
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        prompt = PromptTemplate(
            template="""
            You are a helpful assistant.
            Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.

            {context}
            Question: {question}
            """,
            input_variables=['context', 'question']
        )

        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })

        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm | parser
        answer = main_chain.invoke(prompt_text)

    except Exception as e:
        print("Error generating answer with LLM:", e)
        return jsonify({"error": "Failed to generate answer"}), 500

    return jsonify({"summary": answer})


if __name__ == '__main__':
    app.run(port=5000, debug=True)


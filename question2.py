import os, openai, re, sys
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI, PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

#要約用に、abstractだけ持ってくる関数
#question1とは微妙に違うので注意
def abstract_extraction(pdf_file_path, docs):        
    #Abstractから1 Introductionまでを抜き出す
    pattern = r"Abstract(.*?)1 Introduction"
    match = re.search(pattern, docs[0].page_content, re.DOTALL | re.IGNORECASE)

    #もしマッチすれば、docsをそれだけにする
    if match:
        docs[0].page_content = match.group(1)
        docs = docs[0]
        return [docs]
    #マッチしない時は2ページ目のInstructionまでを持ってくる
    else :
        pattern_2ndpage = r"(.*?)1 Introduction"
        match_2ndpage = re.search(pattern_2ndpage, docs[1].page_content, re.DOTALL | re.IGNORECASE)
        if match_2ndpage:
            docs[1].page_content = match_2ndpage.group(1)
            docs = docs[:2]
            return docs
        #それでもマッチしない時は2ページ目までを持ってくる
        else : 
            docs = docs[:2]
            return docs

#PDFの内容をチャンク化してベクトル化する関数
def pdf_database(docs):
    #内容をチャンク化
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)

    #ベクトルデータベースに格納
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_documents(chunks, embeddings)

    return knowledge_base

    
#初めの要約部分の関数
def summarize_pdf(pdf_file_path, docs, llm):
    #abstractだけ持ってくる
    docs_abst = abstract_extraction(pdf_file_path, docs)
    prompt_template = """    
    Write a concise summary of this in one sentence. Start with 'This is a paper about'. Then, traslate into Japanese with ですます調.      

    {text}

    Japanese output(ですます調): """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    
    chain = load_summarize_chain(llm, chain_type="map_reduce", 
                                map_prompt=PROMPT, combine_prompt=PROMPT)
    custom_summary = chain({"input_documents": docs_abst}, return_only_outputs=True)["output_text"]

    #\nで始まる事があるので、その場合は削除
    if custom_summary.startswith("\n"):
        custom_summary = custom_summary.split('\n')[1]
    
    return custom_summary

#Q&Aの関数
def qanda(knowledge_base, llm, custom_summary):
    #プロンプトの設定
    prompt_template = """あなたはフレンドリーなAIアシスタントです。以下のcontextとchat_historyを使用して、最後のquestionに答えてください。\
    答えがわからない場合は、'すみません、その質問の答えは本論文に記載がため、わかりません。'と言ってください。\
    また、日本語で回答してください。 

    {context}
    
    {chat_history}
    質問: {question}    
    回答:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question", "chat_history"]
    )
    
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
    
    #first responseをmemoryに格納しておく
    #custom_summaryにしているのは、頭の"SYSTEM:"を省くため
    memory.chat_memory.add_ai_message(custom_summary)

    #input
    user_question = input('USER:')
    
    #終了の合図
    if user_question == "ありがとう！":
        print("SYSTEM:こちらこそ！また何か気になる論文があればいつでも聞いて下さいね。")
        return 
    elif user_question == "終了":
        return            
    else : 
        #終了しない時
        #英語で出力してしまう時の対策
        user_question = user_question + "日本語で回答して下さい。"
    
    while True:
        docs_qa = knowledge_base.similarity_search(user_question)
        chain = load_qa_chain(llm, chain_type="map_reduce", question_prompt=PROMPT, memory = memory)
        response = chain({"input_documents": docs_qa, "question": user_question}, return_only_outputs=True)["output_text"]

        #空白を詰める。
        print(''.join(f"SYSTEM : {response}".split()))
        
        user_question = input('USER:')
        
        #終了の合図
        if user_question == "ありがとう！":
            print("SYSTEM:こちらこそ！また何か気になる論文があればいつでも聞いて下さいね。")
            break
        elif user_question == "終了":
            break        
        else : 
            #終了しない時
            user_question = user_question + "日本語で回答して下さい。"
            
            
def main():
    url = sys.argv[1]    
    openai_key = sys.argv[2]
    os.environ["OPENAI_API_KEY"] = f"{openai_key}"
    
    #初期設定
    llm = OpenAI(temperature = 0)
    loader = PyPDFLoader(f"{url}")
    docs = loader.load_and_split()
        
    #knowledge_base
    knowledge_base = pdf_database(docs)
    
    #初めの要約
    custom_summary = summarize_pdf(f"{url}", docs, llm)    
    
    #first_responseとして格納しておく
    #空白を詰める。
    first_response = ''.join(f"SYSTEM : こんにちは！{custom_summary}何かお手伝い出来ることはありますか？".split()) 
    print(first_response)
    
    #Q&A
    qanda(knowledge_base, llm, custom_summary)
        
if __name__ == "__main__":
    main()
import * as dotenv from "dotenv";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Document } from "@langchain/core/documents";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { AIMessage, HumanMessage } from "@langchain/core/messages";


dotenv.config();

const chatModel = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0,
  configuration: { basePath: "https://api.chatanywhere.tech/v1"}
});

// demo1
// const res = await chatModel.invoke("what is LangSmith?");
// console.log(res);

// llm chain
// const prompt = ChatPromptTemplate.fromMessages([
//   ["system", "You are a world class technical documentation writer."],
//   ["user", "{input}"],
// ]);

// const outputParser = new StringOutputParser();
// const chain = prompt.pipe(chatModel).pipe(outputParser);
// const res = await chain.invoke({
//   input: 'what is LangSmith?'
// });
// console.log(res);

// retrieval chain

const loader = new CheerioWebBaseLoader(
  "https://docs.smith.langchain.com/user_guide"
);
const docs = await loader.load();
// 使用文本分割器处理获取到的内容，减少模型调用中传递的最大数据量的干扰
const splitter = new RecursiveCharacterTextSplitter();
const splitDocs = await splitter.splitDocuments(docs);

// 使用 OpenAI API 生成嵌入代码的类。
// 该类扩展了 Embeddings 类，并实现了 OpenAIEmbeddingsParams 和 AzureOpenAIInput。
const embeddings = new OpenAIEmbeddings({
  configuration: {
    baseURL: "https://api.chatanywhere.tech/v1"
  }
});
// 使用嵌入模型将文档提取到向量存储中
// 从 Document 实例数组创建 MemoryVectorStore 实例
// 它会将文档添加到存储中。
const vectorstore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);

// 该链将接受传入的问题，查找相关文档，然后将这些文档与原始问题一起传递给 LLM 并要求其回答原始问题。
// const prompt =
//   ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:

// <context>
// {context}
// </context>

// Question: {input}`); // 格式化一系列对话信息。

// // 创建一个链，将文档列表传递给模型。
// const documentChain = await createStuffDocumentsChain({
//   llm: chatModel,
//   prompt,
// });

// // 也可以通过直接传入文档来运行它
// // const res = await documentChain.invoke({
// //   input: "what is LangSmith?",
// //   context: [
// //     new Document({
// //       pageContent:
// //         "LangSmith is a platform for building production-grade LLM applications.",
// //     }),
// //   ],
// // });
// // console.log(result)

// const retriever = vectorstore.asRetriever();

// // 创建一个检索链，检索文件，然后将其传递出去。
// const retrievalChain = await createRetrievalChain({
//   combineDocsChain: documentChain,
//   retriever,
// });
// const result = await retrievalChain.invoke({
//   input: "what is LangSmith?",
// });

// console.log(result.answer);


// 会话检索
const retriever = vectorstore.asRetriever();

const historyAwarePrompt = ChatPromptTemplate.fromMessages([
  new MessagesPlaceholder("chat_history"),
  ["user", "{input}"],
  [
    "user",
    "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
  ],
]);

const historyAwareRetrieverChain = await createHistoryAwareRetriever({
  llm: chatModel,
  retriever,
  rephrasePrompt: historyAwarePrompt,
});

const chatHistory = [
  new HumanMessage("Can LangSmith help test my LLM applications?"),
  new AIMessage("Yes!"),
];

const res = await historyAwareRetrieverChain.invoke({
  chat_history: chatHistory,
  input: "Tell me how!",
});
console.log(res);


const historyAwareRetrievalPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "Answer the user's questions based on the below context:\n\n{context}",
  ],
  new MessagesPlaceholder("chat_history"),
  ["user", "{input}"],
]);

const historyAwareCombineDocsChain = await createStuffDocumentsChain({
  llm: chatModel,
  prompt: historyAwareRetrievalPrompt,
});

const conversationalRetrievalChain = await createRetrievalChain({
  retriever: historyAwareRetrieverChain,
  combineDocsChain: historyAwareCombineDocsChain,
});

const result2 = await conversationalRetrievalChain.invoke({
  chat_history: [
    new HumanMessage("Can LangSmith help test my LLM applications?"),
    new AIMessage("Yes!"),
  ],
  input: "tell me how",
});

console.log(result2.answer);
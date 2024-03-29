import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { createRetrieverTool } from "langchain/tools/retriever";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { AgentExecutor, createOpenAIFunctionsAgent } from "langchain/agents";
import { pull } from "langchain/hub";
import { ChatPromptTemplate } from "langchain/prompts";

const loader = new CheerioWebBaseLoader(
    "https://docs.smith.langchain.com/user_guide"
);
const docs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter();
const splitDocs = await splitter.splitDocuments(docs);

const embeddings = new OpenAIEmbeddings({
    configuration: {
        baseURL: "https://api.chatanywhere.tech/v1"
    }
});
const vectorstore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
);

const retriever = vectorstore.asRetriever();
const retrieverTool = await createRetrieverTool(retriever, {
    name: "langsmith_search",
    description:
        "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
});

const searchTool = new TavilySearchResults();
const tools = [retrieverTool, searchTool];

// Get the prompt to use - you can modify this!
// If you want to see the prompt in full, you can at:
// https://smith.langchain.com/hub/hwchase17/openai-functions-agent
const agentPrompt = await pull<ChatPromptTemplate>(
    "hwchase17/openai-functions-agent"
);

const agentModel = new ChatOpenAI({
    modelName: "gpt-3.5-turbo-1106",
    temperature: 0,
    configuration: {
        baseURL: "https://api.chatanywhere.tech/v1"
    }
});

const agent = await createOpenAIFunctionsAgent({
    llm: agentModel,
    tools,
    prompt: agentPrompt,
});

const agentExecutor = new AgentExecutor({
    agent,
    tools,
    verbose: true,
});
const agentResult = await agentExecutor.invoke({
    input: "how can LangSmith help with testing?",
});

console.log(agentResult.output);
import * as dotenv from "dotenv";
import { OpenAI } from "langchain";

dotenv.config();

const model = new OpenAI({
  modelName: "gpt-3.5-turbo",
  openAIApiKey: process.env.OPENAI_API_KEY,
  temperature: 0
}, { basePath: "https://api.chatanywhere.tech/v1"});

const res = await model.call(
  "What's a good idea for an application to build with GPT-3?"
);

console.log(res);

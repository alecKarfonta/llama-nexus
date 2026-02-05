- [ ] On the deploy page refactor the paramaeter section Reference the information on the argements here https://github.com/ggml-org/llama.cpp/tree/master/tools/server


- [ ]  On the chat page show the current active model


DEFAULT PROMPT
Write a detailed plan to implement this functionality Include references to files in the codebase that need to change to support this functionality. 

-[ ] Lets plan out an integration with graph rag: https://github.com/alecKarfonta/graphrag

We should be able leverage grpahrag for more advanced document processing, chunking, entity relationship extraction, etc

This will be a major refactor so lets make all of this new functionality work in tandem with the existing Document management system in llama-nexus. We just want to be able to leverage the more advanved features of graphrag when possible.


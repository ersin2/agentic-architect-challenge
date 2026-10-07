"""Part 3: an agent that answers questions about one document, with memory and a calculator tool.

The loop is written by hand on top of agentkit.llm (raw Gemini/OpenAI calls), so every
step can be explained: ask the model -> if it asks for a tool, run it and send the
result back -> repeat until it answers, within a step limit.
"""

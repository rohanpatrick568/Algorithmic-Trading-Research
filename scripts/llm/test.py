import lmstudio as lms

model = lms.llm("qwen3-4b-thinking-2507")
result = model.respond("What is the meaning of life?")

print(result)

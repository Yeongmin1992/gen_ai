


llm = ChatOpenAI(temperature=0.1, max_tokens=200, model="gpt-3.5-turbo")

bug_step1_chain = create_chain(
    llm=llm
    template_path=BUG_STEP1_PROMPT_TEMPLATE,
)
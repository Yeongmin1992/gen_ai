from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain, LLMRouterChain, ConversationChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import PromptTemplate
import os

load_dotenv()

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BUG_STEP1_PROMPT_TEMPLATE = os.path.join(
    CUR_DIR, "prompt_templates", "bug_say_sorry.txt"
)
BUG_STEP2_PROMPT_TEMPLATE = os.path.join(
    CUR_DIR, "prompt_templates", "bug_request_context.txt"
)
ENHANCE_STEP1_PROMPT_TEMPLATE = os.path.join(
    CUR_DIR, "prompt_templates", "enhancement_say_thanks.txt"
)

def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template

def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key = output_key,
        # 진행상황 확인 편하도록
        verbose=True
    )

llm = ChatOpenAI(
    temperature=0.1,
    max_tokens=200,
    model="gpt-3.5-turbo"
)

bug_step1_chain = create_chain(
    llm=llm,
    template_path=BUG_STEP1_PROMPT_TEMPLATE,
    output_key="text"
)
bug_step2_chain = create_chain(
    llm=llm,
    template_path=BUG_STEP2_PROMPT_TEMPLATE,
    output_key="text"
)
enhance_step1_chain = create_chain(
    llm=llm,
    template_path=ENHANCE_STEP1_PROMPT_TEMPLATE,
    output_key="text"
)

# bug_sequential_chain = SequentialChain(
#     chains=[bug_step1_chain, bug_step2_chain],
#     input_variables=["user_message"],
#     verbose=True
# )
# enhance_sequential_chain = SequentialChain(
#     chains=[enhance_step1_chain],
#     input_variables=["user_message"],
#     verbose=True
# )

# langchain에선 intent가 bug인지, enhancement인지 구분하기 위해 RouterChain을 사용함.
# LLMRouterChain을 사용하면 자동으로 LLM을 실행하여 Routing을 할 수 있다.
# Router의 경우엔 OutputParser와 함께 사용해야한다.

destinations = [
    "bug: Related to a bug, vulnerability, unexpected error with an existing feature",
    "documentation: Changes to documentation and examples, like .md, .rst, .ipynb files. Changes to the docs/ folder",
    "enhancement: A large net-new component, integration, or chain. Use sparingly. The largest features",
    "improvement: Medium size change to existing code to handle new use-cases",
    "nit: Small modifications/deletions, fixes, deps or improvements to existing code or docs",
    "question: A specific question about the codebase, product, project, or how to use a feature",
    "refactor: A large refactor of a feature(s) or restructuring of many files",
]
destinations = "\n".join(destinations)
router_prompt_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations)
router_prompt = PromptTemplate.from_template(
    template=router_prompt_template, output_parser=RouterOutputParser()
)
router_chain = LLMRouterChain.from_llm(llm=llm, prompt=router_prompt, verbose=True)

# MultiPromptChain은 SequentialChain 사용 불가능. MultiPromptChain은 제한이 많어 커스텀으로 진행 추천.
# multi_prompt_chain = MultiPromptChain(
#     router_chain=router_chain,
#     default_chains={
#         "bug": bug_sequential_chain,
#         "enhancement": enhance_sequential_chain
#     },
#     # 두개로 분리되지 않은 질문일 때는 ConversationChain을 사용
#     default_chain=ConversationChain(llm=llm, output_key="text")
    
# )

multi_prompt_chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains={
        "bug": bug_step1_chain,
        "enhancement": enhance_step1_chain,
    },
    default_chain=ConversationChain(llm=llm, output_key="text"),
)
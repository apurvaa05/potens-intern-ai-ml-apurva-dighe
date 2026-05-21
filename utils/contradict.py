from utils.llm import generate_answer

def check_contradiction(doc1, doc2, topic):

    context = f"""
Document 1:
{doc1}

Document 2:
{doc2}
"""

    question = f"""
Do these two documents contradict each other regarding:
{topic}?

Answer in this format only:

Contradiction: Yes/No

Reason:
<short explanation>
"""

    response = generate_answer(
        question,
        context
    )

    return response
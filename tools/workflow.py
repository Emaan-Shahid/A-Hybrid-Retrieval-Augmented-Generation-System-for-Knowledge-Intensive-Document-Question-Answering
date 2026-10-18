from .summary_tool import summarize_document
from .todo_tool import generate_todo
from .email_tool import generate_email


def run_workflow(context):

    summary = summarize_document(context)

    todos = generate_todo(summary)

    email = generate_email(summary)

    return {
        "summary": summary,
        "todos": todos,
        "email": email,
    }
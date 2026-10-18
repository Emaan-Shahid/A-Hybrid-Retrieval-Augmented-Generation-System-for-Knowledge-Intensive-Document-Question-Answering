"""
Tool Router for FlowMind AI Productivity Assistant
"""

def select_tool(query: str) -> str:
    """
    Decide which tool should handle the user's request.
    """

    query = query.lower()

    # -----------------------------
    # Workflow
    # -----------------------------
    workflow_keywords = [
        "workflow",
        "meeting",
        "minutes",
        "action items",
        "follow up",
        "follow-up"
    ]

    # -----------------------------
    # Summary
    # -----------------------------
    summary_keywords = [
        "summarize",
        "summary",
        "summarise",
        "key points",
        "brief"
    ]

    # -----------------------------
    # Email
    # -----------------------------
    email_keywords = [
        "email",
        "mail",
        "draft email",
        "compose email"
    ]

    # -----------------------------
    # Todo
    # -----------------------------
    todo_keywords = [
        "todo",
        "to-do",
        "task",
        "tasks",
        "checklist"
    ]

    if any(word in query for word in workflow_keywords):
        return "workflow"

    elif any(word in query for word in summary_keywords):
        return "summary"

    elif any(word in query for word in email_keywords):
        return "email"

    elif any(word in query for word in todo_keywords):
        return "todo"

    else:
        return "rag"
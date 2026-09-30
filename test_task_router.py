from app.analysis.task_router import TaskRouter


router = TaskRouter()


queries = [

    "What methodology did the researchers use?",

    "What datasets were used in the studies?",

    "Which models and algorithms were used?",

    "What are the limitations of the study?",

    "What future research directions did the authors suggest?",

    "Give me a literature review of these papers.",

    "What does this paper say about Virtual Try-On?"

]


print("\n")
print("=" * 80)
print("TASK ROUTER TEST")
print("=" * 80)


for query in queries:

    result = router.route(
        query
    )

    print("\n" + "-" * 80)

    print(
        f"Query: {query}"
    )

    print(
        f"Task: {result['task']}"
    )

    print(
        f"Scores: {result['scores']}"
    )
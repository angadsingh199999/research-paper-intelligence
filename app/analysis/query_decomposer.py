class QueryDecomposer:

    def __init__(self):

        self.subtasks = {

            "methodology": [
                "methodology",
                "method",
                "methods",
                "experimental design",
                "research design",
                "how was the study conducted"
            ],

            "datasets": [
                "dataset",
                "datasets",
                "data set",
                "data source",
                "data sources"
            ],

            "models": [
                "model",
                "models",
                "algorithm",
                "algorithms",
                "architecture",
                "classifier"
            ],

            "limitations": [
                "limitation",
                "limitations",
                "weakness",
                "weaknesses",
                "constraint",
                "constraints",
                "drawback",
                "drawbacks"
            ],

            "results": [
                "result",
                "results",
                "finding",
                "findings",
                "main finding",
                "main findings",
                "hypothesis",
                "hypotheses",
                "hypothesis testing",
                "what did the study find",
                "what were the findings",
                "what did the researchers find",
                "key findings"
            ],

            "future_work": [
                "future work",
                "future research",
                "further research",
                "research directions"
            ]
        }


    def decompose(self, query):

        query_lower = query.lower()

        detected_tasks = []


        # ----------------------------------------
        # Detect requested subtasks
        # ----------------------------------------

        for task, keywords in (
            self.subtasks.items()
        ):

            for keyword in keywords:

                if keyword in query_lower:

                    detected_tasks.append(
                        task
                    )

                    break


        # ----------------------------------------
        # If nothing detected
        # ----------------------------------------

        if not detected_tasks:

            return [
                {
                    "task":
                    "question_answering",

                    "query":
                    query
                }
            ]


        # ----------------------------------------
        # Create focused queries
        # ----------------------------------------

        decomposed_queries = []


        for task in detected_tasks:

            if task == "methodology":

                subquery = (
                    "What methodology "
                    "and experimental design "
                    "did the researchers use?"
                )


            elif task == "datasets":

                subquery = (
                    "What datasets, data sources, "
                    "or data were used?"
                )


            elif task == "models":

                subquery = (
                    "What models, algorithms, "
                    "or architectures were used?"
                )


            elif task == "limitations":

                subquery = (
                    "What limitations, weaknesses, "
                    "or constraints did the authors "
                    "identify?"
                )


            elif task == "future_work":

                subquery = (
                    "What future research or "
                    "research directions did the "
                    "authors suggest?"
                )

            elif task == "results":

                subquery = (
                    "What were the main findings, "
                    "hypothesis results, and statistical "
                    "results reported by the researchers?"
                )

            else:

                subquery = query


            decomposed_queries.append({

                "task": task,

                "query": subquery

            })


        return decomposed_queries
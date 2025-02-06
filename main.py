import os
import re
import subprocess
from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.syntax import Syntax

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI

from RAG import CodeEmbeddings


class Settings(BaseSettings):
    """
    Configuration settings for the application, including the Google API key.
    """

    GOOGLE_API_KEY: str

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


def run_cpp_code(cpp_code: str) -> str:
    """
    Executes C++ code within a Docker container.

    Args:
        cpp_code (str): The C++ code to execute.

    Returns:
        str: The output of the program or an error message.
    """
    # Command that builds the Docker image for running C++ code
    build_cmd = ["docker", "build", "-t", "cpp-runner", "./run_cpp"]
    try:
        subprocess.run(build_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        return f"Docker build failed: {e.stderr.decode().strip()}"

    # Command to run the Docker container and execute the C++ code
    run_cmd = ["docker", "run", "--rm", "-i", "cpp-runner", "-"]
    try:
        result = subprocess.run(
            run_cmd, input=cpp_code.encode(), capture_output=True, check=True
        )
        return result.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
        return f"Program execution failed: {e.stderr.decode().strip()}"


# Initializing settings from environment variables.
# Configuring the Language Model with specified parameters.
settings = Settings()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-thinking-exp-01-21",
    api_key=settings.GOOGLE_API_KEY,
    temperature=1.0,
)


class CppCodeInput(BaseModel):
    """
    Schema for inputting C++ code to be executed.
    """

    code: str = Field(description="C++ code to execute")


# Define the tool for running C++ code using the previously defined function
cpp_code_runner = Tool(
    name="cpp_code_runner",
    description="Executes C++ code in Docker container",
    func=run_cpp_code,
    args_schema=CppCodeInput,
)


SYSTEM_PROMPT = """You are an expert C++ quiz generator. Follow these strict rules:

1. Generate ONLY complex and tricky cpp code snippets
2. NEVER generate basic or simple code patterns
3. Each snippet MUST include at least TWO of:
    * Pointer manipulation
    * Function behavior quirks
    * Reference complexities
    * Memory management
    * Loop edge cases and unconventional patterns
    * Const correctness
    * sizeof() and datatype size edge cases
    * Order of operations (e.g., x++ + x++)
    * Conditional statement edge cases (if(x=0), switch fallthrough)
    * Operator overloading (IF OOP)
    * Complex templates (IF OOP)
    * Virtual functions (IF OOP)
    * Multiple inheritance (IF OOP)

   - Be complete and compilable, or intentionally broken
   - Broken code should illustrate a concept.
   - If a snippet has compilation errors, and it is intentional, keep it, otherwise fix it.
   - Include only iostream, string, fstream
   - Include 'using namespace std;'
   - Have deterministic output (even if undefined behavior is predictable)
   - Do not add helpful comments. Make the code confusing.

3. For each snippet, follow this format:
   ```cpp
   // Question #N
   [code here]
   // Expected Output: [output]
   // Explanation: [brief explanation]
   ```
"""

# Initialize message history with the system prompt
messages: List[Any] = [SystemMessage(content=SYSTEM_PROMPT)]


class QuizOutput(BaseModel):
    """
    Schema for storing the output of each quiz question.
    """

    code: str
    output: str
    success: bool
    explanation: str = ""


def log_cpp_snippet(code: str, output: str, success: bool) -> None:
    """
    Logs C++ snippets to a file with timestamps.

    Args:
        code (str): The C++ code snippet.
        output (str): Output from running the code.
        success (bool): Whether compilation/execution was successful.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"""
=== {timestamp} ===
Success: {success}

CODE:
{code}

OUTPUT:
{output}

{"=" * 50}
"""

    log_file = os.path.join(log_dir, "cpp_snippets.log")
    with open(log_file, "a") as f:
        f.write(log_entry)


def clean_code_snippet(code: str) -> str:
    """
    Cleans code snippet by removing markdown formatting and extra whitespace.

    Args:
        code (str): Raw code snippet potentially containing markdown.

    Returns:
        str: Cleaned code string.
    """
    # Remove markdown code block markers
    code = code.replace("```cpp", "").replace("```", "").strip()
    # Remove trailing whitespace from each line
    return "\n".join(line.rstrip() for line in code.splitlines())


def split_code_snippets(content: str) -> List[str]:
    """
    Splits multiple code snippets from LLM response.

    Args:
        content (str): The raw LLM response containing multiple snippets.

    Returns:
        List[str]: A list of individual code snippets.
    """
    snippets = []
    current_snippet = []

    for line in content.splitlines():
        if line.strip().startswith("// Question #"):
            if current_snippet:
                snippets.append("\n".join(current_snippet))
                current_snippet = []
        current_snippet.append(line)

    if current_snippet:
        snippets.append("\n".join(current_snippet))

    return snippets


def format_output(result: Dict[str, Any]) -> None:
    """
    Pretty prints quiz results with better formatting.

    Args:
        result (Dict[str, Any]): Result containing code snippets and outputs.
    """
    console = Console()

    for i, snippet in enumerate(result.get("snippets", []), 1):
        console.print(f"\n[bold cyan]Question #{i}[/bold cyan]")
        console.print("=" * 40)

        # Format code with syntax highlighting
        code = Syntax(snippet.get("code", ""), "cpp", theme="monokai")
        console.print("\n[bold]Code:[/bold]")
        console.print(code)

        # Format output
        console.print("\n[bold]Output:[/bold]")
        console.print(snippet.get("output", ""))
        console.print("=" * 40)


def parse_difficulties(query: str) -> dict:
    """
    Parse the query for difficulty instructions.
    Expected format examples: "3 hard", "2 medium", "5 easy"
    Returns a dictionary with keys 'hard', 'medium', 'easy' and integer counts.
    """
    pattern = r'(\d+)\s*(hard|medium|easy)'
    matches = re.findall(pattern, query.lower())
    difficulties = {}
    for count, level in matches:
        difficulties[level] = int(count)
    return difficulties


def run_chain(query: str) -> Dict[str, Any]:
    """
    Runs the chain to generate and test C++ code snippets.
    Before invoking the LLM, this function integrates reference code snippets
    retrieved via RAG for context. Also, if difficulty instructions are provided
    in the query, it appends a DIFFICULTY INSTRUCTIONS section that tells the LLM
    to generate the specified number of questions at each difficulty level.

    The reference JSON file is chosen based on the query's content:
      - If the query mentions "oop" or "object oriented", the OOP file is loaded.
      - If the query mentions "pf" or "programming fundamentals", the PF file is loaded.

    Args:
        query (str): The user query to generate code snippets.

    Returns:
        Dict[str, Any]: A dictionary containing the snippets and their results.
    """
    global messages

    # Determining the JSON file to load based on the query's content.
    if any(keyword in query.lower() for keyword in ["oop", "object oriented"]):
        json_path = "data/oop.json"
    elif any(keyword in query.lower() for keyword in ["pf", "programming fundamentals"]):
        json_path = "data/pf.json"
    else:
        # Default to PF if no clear indicator is present.
        json_path = "data/pf.json"

    ref_text = ""
    try:
        rag = CodeEmbeddings()
        # Loading the JSON file containing reference code snippets.
        rag.load_json(json_path)
        # Searching for references matching the user query.
        references = rag.search(query, k=3)
        ref_lines = []
        for i, ref in enumerate(references):
            if rag.file_type == "oop":
                ref_lines.append(f"Reference {i+1} (Question):\n{ref['question']}")
            else:
                ref_lines.append(
                    f"Reference {i+1} (Context: {ref['context']}):\n{ref['code']}"
                )
        ref_text = "\n\n".join(ref_lines)
    except Exception as e:
        print(f"RAG integration error: {e}")

    # Combining the original query with the reference snippets (if any).
    combined_query = query
    if ref_text:
        combined_query += (
            "\n\nUse the following reference code snippets for context when generating new questions:\n\n"
            f"{ref_text}"
        )
    # Add extra instruction if Programming Fundamentals was requested.
    if rag.file_type == "pf":
         combined_query += (
             "\n\nIMPORTANT: The generated C++ code snippet must strictly adhere to Programming Fundamentals principles. "
             "Avoid using Object-Oriented Programming constructs such as classes, inheritance, virtual functions, operator overloading, and templates. "
             "Instead, focus on pointer manipulation, manual memory management, loop edge cases, and conditional statement quirks."
         )

    # Parse and append difficulty instructions if provided
    difficulties = parse_difficulties(query)
    if difficulties:
        diff_instr = "\n\nDIFFICULTY INSTRUCTIONS:\n"
        diff_instr += "Generate questions with the following difficulty breakdown:\n"
        for level in ['hard', 'medium', 'easy']:
            if level in difficulties:
                diff_instr += f" - {difficulties[level]} {level} question{'s' if difficulties[level] > 1 else ''}\n"
        diff_instr += "\nBelow are examples to help you judge the difficulty levels:\n"
        diff_instr += "\nEasy questions examples:\n"
        diff_instr += "```cpp\n#include <iostream>\nusing namespace std;\n\nint main()\n{\n    int *a, *b, *c;\n    int x = 800, y = 300;\n    a = &x;\n    b = &y;\n    *a = (*b) - 200;\n    cout<<x<<\" \"<<*a;\n    return 0;\n}\n```\n"
        diff_instr += "```cpp\n#include <iostream>\nusing namespace std;\n\nint main() {\n    int a = 5;\n    if (a = 0) {\n        cout << \"Zero\" << endl;\n    } else {\n        cout << \"Non-zero\" << endl;\n    }\n    cout << a << endl;\n    return 0;\n}\n```\n"
        diff_instr += "\nMedium questions examples:\n"
        diff_instr += "```cpp\n#include <iostream>\nusing namespace std;\n\nvoid find(int a, int &b, int &c, int d)\n{\n    if (d < 1)\n        return;\n    cout << a << \",\" << b << \",\" << c << endl;\n    c = a + 2 * b;\n    int temp = b;\n    b = a;\n    a = 2 * temp;\n    d % 2 ? find(b, a, c, d - 1) : find(c, b, a, d - 1);\n}\n\nint main()\n{\n    int a = 1, b = 2, c = 3, d = 4;\n    find(a, b, c, d);\n    cout << a << \",\" << b << \",\" << c << endl;\n    return 0;\n}\n```\n"
        diff_instr += "```cpp\n#include <iostream>\nusing namespace std;\n\nint WHAT(int A[], int N) {\n    int ANS = 0;\n    int S = 0;\n    int E = N-1;\n    \n    for(S = 0, E = N-1; S < E; S++, E--) {\n        ANS += A[S] - A[E];\n    }\n    return ANS;\n}\n\nint main() {\n    int A[] = {1, 2, 3, 4, -5, 1, 3, 2, 1};\n    cout << WHAT(A, 7);\n    return 0;\n}\n```\n"
        diff_instr += "\nHard question example:\n"
        diff_instr += "```cpp\n#include <iostream>\nusing namespace std;\n\nconst int s = 3;\nint *listMystery(int list[][s])\n{\n    int i = 1, k = 0;\n    int *n = new int[s];\n    for (int i = 0; i < s; ++i)\n        n[i] = 0;\n    while (i < ::s)\n    {\n        int j = ::s - 1;\n        while (j >= i)\n        {\n            n[k++] = list[j][i] * list[i][j];\n            j = j - 1;\n        }\n        i = i + 1;\n    }\n    return n;\n}\nvoid displayMystery(int *arr)\n{\n    cout << \"[ \";\n    for (int i = 0; i < s; ++i)\n        cout << arr[i] << ((i != ::s - 1) ? \", \" : \" \");\n    cout << \"]\" << endl;\n}\nint main()\n{\n    int L[][::s] = {{8, 9, 4}, {2, 3, 4}, {7, 6, 1}};\n    int *ptr = listMystery(L);\n    displayMystery(ptr);\n    delete[] ptr;\n    return 0;\n}\n```\n"
        combined_query += diff_instr

    messages.append(HumanMessage(content=combined_query))

    try:
        # Invoking the LLM with the current message history.
        response = llm.invoke(messages)
        if not response or not response.content:
            raise ValueError("Empty response from LLM")

        # Cleaning and split the LLM response into individual code snippets.
        cleaned_content = clean_code_snippet(response.content)
        code_snippets = split_code_snippets(cleaned_content)
        results = []

        # Executing each code snippet and logging the results.
        for snippet in code_snippets:
            run_result = cpp_code_runner.invoke({"code": snippet})
            success = not run_result.startswith("Program execution failed") and not run_result.startswith("Docker build failed")
            log_cpp_snippet(snippet, run_result, success)
            results.append({"code": snippet, "output": run_result, "success": success})

        # Appending the AI's response to the message history.
        messages.append(AIMessage(content=response.content))
        return {"snippets": results}

    except Exception as e:
        print(f"\nError: {e}")
        return {"snippets": []}


query = "Generate 10 hard C++ code snippets for PF"
result = run_chain(query)
format_output(result)

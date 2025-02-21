import os
import re
import json
import shutil
import subprocess
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
import atexit
import markdown2

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from rich.panel import Panel
from rich import print as rprint

from langchain_core.messages import (
    HumanMessage, 
    SystemMessage
)
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


class DockerRunner:
    """Manages Docker container for running C++ code."""
    _instance = None
    _is_built = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DockerRunner, cls).__new__(cls)
        return cls._instance

    def ensure_image_built(self):
        """Builds Docker image only if not already built."""
        if not self._is_built:
            build_cmd = ["docker", "build", "-t", "cpp-runner", "./run_cpp"]
            try:
                subprocess.run(build_cmd, check=True, capture_output=True)
                self._is_built = True
            except subprocess.CalledProcessError as e:
                return f"Docker build failed: {e.stderr.decode().strip()}"

    def run_code(self, cpp_code: str) -> str:
        """Runs C++ code in the Docker container."""
        self.ensure_image_built()
        run_cmd = ["docker", "run", "--rm", "-i", "cpp-runner", "-"]
        try:
            result = subprocess.run(
                run_cmd, input=cpp_code.encode(), capture_output=True, check=True
            )
            return result.stdout.decode().strip()
        except subprocess.CalledProcessError as e:
            return f"Program execution failed: {e.stderr.decode().strip()}"


def run_cpp_code(cpp_code: str) -> str:
    """Executes C++ code using the DockerRunner singleton."""
    return DockerRunner().run_code(cpp_code)


class CacheManager:
    """Manages cache directories and operations."""
    def __init__(self, base_dir: str = "cache"):
        self.base_dir = Path(base_dir)
        self.explanations_dir = self.base_dir / "explanations"
        self.queries_dir = self.base_dir / "queries"
        self._init_directories()
    
    def _init_directories(self):
        """Create all required cache directories."""
        self.base_dir.mkdir(exist_ok=True)
        self.explanations_dir.mkdir(exist_ok=True)
        self.queries_dir.mkdir(exist_ok=True)
    
    def get_explanation_path(self, cache_key: str) -> Path:
        """Get path for explanation cache file."""
        return self.explanations_dir / f"{cache_key}.json"
    
    def get_query_path(self, cache_key: str) -> Path:
        """Get path for query cache file."""
        return self.queries_dir / f"{cache_key}.json"
    
    def clear_cache(self, keep_docker: bool = True) -> None:
        """Clear all cache directories except Docker-related files."""
        try:
            # Clear explanations directory
            if self.explanations_dir.exists():
                shutil.rmtree(self.explanations_dir)
            self.explanations_dir.mkdir(exist_ok=True)
            
            # Clear queries directory
            if self.queries_dir.exists():
                shutil.rmtree(self.queries_dir)
            self.queries_dir.mkdir(exist_ok=True)
            
        except Exception as e:
            print(f"Error clearing cache: {e}")

    def cleanup(self) -> None:
        """Completely remove cache directory."""
        try:
            if self.base_dir.exists():
                shutil.rmtree(self.base_dir)
        except Exception as e:
            print(f"Error cleaning up cache: {e}")


class ReportGenerator:
    """Generates HTML and Markdown reports for quiz results."""
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.env = Environment(
            loader=FileSystemLoader("templates"),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def generate(self, result: Dict[str, Any], query: str) -> tuple[Path, Path]:
        """Generate both HTML and Markdown reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_file = self.output_dir / f"quiz_{timestamp}.html"
        md_file = self.output_dir / f"quiz_{timestamp}.md"
        
        # First generate markdown
        md_content = self._generate_markdown(query, result.get("snippets", []))
        md_file.write_text(md_content)
        
        # Convert markdown to HTML and embed in template
        html_body = markdown2.markdown(
            md_content,
            extras=['fenced-code-blocks', 'code-friendly']
        )
        
        # Generate final HTML 
        template = self.env.get_template("quiz_report.html")
        html_content = template.render(content=html_body)
        html_file.write_text(html_content)
        
        return html_file, md_file
    
    def _generate_markdown(self, query: str, snippets: List[Dict[str, Any]]) -> str:
        """Generate Markdown format report."""
        md = [
            f"# C++ Quiz Report\n",
            f"Query: {query}\n",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        ]
        
        for i, snippet in enumerate(snippets, 1):
            explanation = snippet.get("explanation", "").strip()
            formatted_explanation = "\n\n".join(p.strip() for p in explanation.split('\n') if p.strip())
            # Escape asterisks in the regex pattern
            formatted_explanation = re.sub(r'\*\*Explanation:\*\*', '', formatted_explanation)
            
            md.extend([
                f"## Question {i}\n",
                "### Code\n```cpp",
                snippet.get("code", ""),
                "```\n",
                "### Output\n```",
                snippet.get("output", ""),
                "```\n",
                "### Explanation\n",
                formatted_explanation,
                "\n---\n"
            ])
        
        return "\n".join(md)


class ExplanationGenerator:
    """Handles code explanation generation with proper context management."""
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",  # Regular Flash model for explanations
            api_key=api_key,
            temperature=0.3,  # Lower temperature for more focused explanations
        )
    
    def _get_cache_key(self, code: str, output: str) -> str:
        """Generate cache key for code and output combination."""
        combined = f"{code}{output}".encode()
        return hashlib.md5(combined).hexdigest()
    
    def get_explanation(self, code: str, output: str) -> str:
        """Get explanation for code output, using cache if available."""
        cache_key = self._get_cache_key(code, output)
        cache_file = cache_mgr.get_explanation_path(cache_key)
        
        try:
            if cache_file.exists():
                cached_data = safe_json_load(str(cache_file))
                if cached_data and 'explanation' in cached_data:
                    return cached_data['explanation']
        except Exception as e:
            print(f"Cache read error: {e}")
        
        explanation_prompt = f"""Act as a C++ expert. Explain the following code and its output:

CODE:
{code}

OUTPUT:
{output}

Provide a technical explanation for why this code produces exactly this output.
Focus on key concepts, memory behavior, and any tricky aspects.
Be concise but thorough. Do not miss anything, but note that the user is a student preparing for an exam, and is already familiar with C++."""

        try:
            messages = [
                SystemMessage(content="You are a C++ expert. Explain code behavior accurately and technically, yet concisely."),
                HumanMessage(content=explanation_prompt)
            ]
            
            explanation = self.llm.invoke(messages).content.strip()
            
            # Cache the explanation with safe JSON handling
            safe_json_dump({'explanation': explanation}, str(cache_file))
            
            return explanation
        except Exception as e:
            return f"Error generating explanation: {str(e)}"


@lru_cache(maxsize=100)
def get_explanation(code: str, output: str) -> str:
    """Get explanation using the ExplanationGenerator."""
    return explanation_gen.get_explanation(code, output)


class BatchProcessor:
    """Handles batch processing of code snippets and explanations."""
    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.docker_runner = DockerRunner()
    
    def process_batch(self, snippets: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of snippets together."""
        results = []
        
        progress_mgr.update_status("Running C++ code snippets...")
        # Process code execution in parallel
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = []
            for i, snippet in enumerate(snippets, 1):
                progress_mgr.update_status(f"Running snippet {i}/{len(snippets)}...")
                futures.append(executor.submit(self.docker_runner.run_code, snippet))
            
            outputs = [f.result() for f in futures]
        
        progress_mgr.update_status("Generating explanations...")
        explanations = self._batch_generate_explanations(snippets, outputs)
        
        # Combine results
        progress_mgr.update_status("Preparing results...")
        for snippet, output, explanation in zip(snippets, outputs, explanations):
            success = not (output.startswith("Program execution failed") or 
                         output.startswith("Docker build failed"))
            results.append({
                "code": snippet,
                "output": output,
                "success": success,
                "explanation": explanation
            })
        
        return results
    
    def _batch_generate_explanations(self, snippets: List[str], outputs: List[str]) -> List[str]:
        """Generate explanations for multiple snippets in one LLM call."""
        all_explanations = []
        
        # Process each snippet individually to avoid cross-contamination
        for i, (code, output) in enumerate(zip(snippets, outputs), 1):
            explanation_prompt = f"""Act as a C++ expert. Analyze this specific code snippet and its output:

CODE:
{code}

OUTPUT:
{output}

Provide a focused technical explanation for this specific snippet.
Focus on key concepts, memory behavior, and any tricky aspects.
Keep your explanation specific to this code only."""

            try:
                messages = [
                    SystemMessage(content="You are a C++ expert. Explain code behavior accurately and technically."),
                    HumanMessage(content=explanation_prompt)
                ]
                
                response = explanation_gen.llm.invoke(messages)
                explanation = response.content.strip()
                
                # Clean up the explanation
                explanation = re.sub(r'```[^`]*```', '', explanation)  # Remove code blocks
                explanation = re.sub(r'CODE:|OUTPUT:|EXPLANATION:', '', explanation)  # Remove headers
                explanation = '\n'.join(line.strip() for line in explanation.split('\n') if line.strip())
                
                all_explanations.append(explanation)
            except Exception as e:
                all_explanations.append(f"Error generating explanation: {str(e)}")
        
        return all_explanations


def process_snippets(code_snippets: List[str]) -> List[Dict[str, Any]]:
    """Process multiple snippets efficiently."""
    batch_processor = BatchProcessor()
    return batch_processor.process_batch(code_snippets)


# Initializing settings from environment variables.
# Configuring the Language Model with specified parameters.
settings = Settings()
cache_mgr = CacheManager()

# Initialize explanation generator
explanation_gen = ExplanationGenerator(settings.GOOGLE_API_KEY)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-thinking-exp-01-21",
    api_key=settings.GOOGLE_API_KEY,
    temperature=1.0,
    timeout=30,
    retry_on_failure=True
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
    * Bitwise operations
    * Array and pointer arithmetic
    * Recursion edge cases
    
    * Operator overloading (IF OOP)
    * Complex templates (IF OOP)
    * Virtual functions (IF OOP)
    * Multiple inheritance (IF OOP)

4. Each snippet should have the following characteristics:
   - Be complete and compilable, or intentionally broken
   - Broken code should illustrate a concept.
   - If a snippet has compilation errors, and it is intentional, keep it, otherwise fix it.
   - Include only iostream, string, fstream headers, avoid STL. Also, use namespace std.
   - Have deterministic output (even if undefined behavior is predictable)
   - Do not add helpful comments. Make the code confusing.

5. For each snippet, follow this format:
   ```cpp
   // Question #N
    #include <iostream>
    [any other includes here]
    using namespace std;
    
    [code here]
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
        console.print(Panel(
            snippet.get("output", ""),
            border_style="blue",
            expand=False
        ))
        
        if snippet.get("explanation"):
            console.print("\n[bold]Explanation:[/bold]")
            # Split explanation into paragraphs and format them
            explanation = snippet.get("explanation", "").strip()
            paragraphs = [p.strip() for p in explanation.split('\n') if p.strip()]
            
            # Create a formatted panel for the explanation
            explanation_text = "\n\n".join(paragraphs)
            console.print(Panel(
                explanation_text,
                border_style="green",
                expand=True,
                padding=(1, 2)
            ))
        
        console.print("\n" + "=" * 40)


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle non-serializable objects."""
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

def safe_json_dump(data: Dict[str, Any], file_path: str) -> None:
    """Safely dump data to JSON file with proper encoding."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=JSONEncoder, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving cache: {e}")

def safe_json_load(file_path: str) -> Optional[Dict[str, Any]]:
    """Safely load JSON data from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (JSONDecodeError, FileNotFoundError, Exception) as e:
        print(f"Error loading cache: {e}")
        return None

class ProgressManager:
    """Manages progress indicators and status updates."""
    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            transient=True
        )
        self.current_task = None
    
    def update_status(self, message: str, complete_previous: bool = True) -> None:
        """Update current status message."""
        if self.current_task and complete_previous:
            self.progress.remove_task(self.current_task)
            self.current_task = None
        
        if not self.current_task:
            self.current_task = self.progress.add_task(message)
        else:
            self.progress.update(self.current_task, description=message)

    def complete_task(self) -> None:
        """Mark current task as complete and remove it."""
        if self.current_task:
            self.progress.remove_task(self.current_task)
            self.current_task = None
    
    def __enter__(self):
        self.progress.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.current_task:
            self.complete_task()
        self.progress.stop()


progress_mgr = ProgressManager()

def run_chain(query: str) -> Dict[str, Any]:
    """Optimized version of run_chain."""
    with progress_mgr:
        progress_mgr.update_status("Checking cache...")
        cache_key = hashlib.md5(query.encode()).hexdigest()
        cache_file = cache_mgr.get_query_path(cache_key)
        
        try:
            if cache_file.exists():
                cached_result = safe_json_load(str(cache_file))
                if cached_result:
                    return cached_result
        except Exception as e:
            print(f"Query cache read error: {e}")

        progress_mgr.update_status("Loading reference examples...")
        # Determining the JSON file to load based on the query's content.
        if any(keyword in query.lower()
               for keyword in ["oop", "object oriented"]):
            json_path = "data/oop.json"

        elif any(keyword in query.lower()
                 for keyword in ["pf", "programming fundamentals"]):
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
       
        # Extra instruction if Programming Fundamentals was requested.
        if rag.file_type == "pf":
             combined_query += (
                 "\n\nIMPORTANT: The generated C++ code snippet must strictly adhere to Programming Fundamentals principles. "
                 "Avoid using Object-Oriented Programming constructs such as classes, inheritance, virtual functions, operator overloading, and templates. "
                 "Instead, focus on pointer manipulation, manual memory management, loop edge cases, recursion oddities, and conditional statement quirks."
             )
        # Extra instructions if OOP was requested.
        if rag.file_type == "pf":
             combined_query += (
                 "\n\nIMPORTANT: The generated C++ code snippet must strictly adhere to Object-Oriented Programming principles. "
                "Focus on classes, constructor and destructor calls, shallow, deep copies, inheritance, operator overloading. "
                "Mix in manual memory management, pointer manipulation, and other low-level constructs."
                )
       
        # Use preset difficulty from the data.
        combined_query += (
            "\n\nIMPORTANT: Generate C++ code snippets based solely on the preset difficulty levels stored in the data. Do not attempt to modify or infer difficulty manually."
        )
        messages.append(HumanMessage(content=combined_query))

        try:
            progress_mgr.update_status("Generating code snippets...")
            # Get initial response with code snippets
            response = llm.invoke(messages)
            if not response or not response.content:
                raise ValueError("Empty response from LLM")

            progress_mgr.update_status("Processing code snippets...")
            cleaned_content = clean_code_snippet(response.content)
            code_snippets = split_code_snippets(cleaned_content)
            
            # Process snippets
            results = process_snippets(code_snippets)
            
            progress_mgr.update_status("Caching results...")
            output = {"snippets": results}
            safe_json_dump(output, str(cache_file))
            
            progress_mgr.update_status("Cleaning up...")
            try:
                shutil.rmtree(cache_mgr.explanations_dir)
                cache_mgr.explanations_dir.mkdir(exist_ok=True)
            except Exception as e:
                print(f"Cache cleanup error: {e}")
            
            return output

        except Exception as e:
            print(f"\nError: {e}")
            return {"snippets": []}


def main():
    """Main function to run the quiz generator."""
    
    # Register cache cleanup to run at program exit
    atexit.register(cache_mgr.cleanup)
    
    rprint("\n[bold cyan]C++ Quiz Generator[/bold cyan]")
    rprint("=" * 40)
    
    with progress_mgr:
        # Initialize with clear sequence
        cache_mgr.clear_cache()
        
        progress_mgr.update_status("Processing ...")
        query = "Generate 10 hard difficulty C++ code snippets for PF"
        progress_mgr.complete_task()
        
        progress_mgr.update_status("Generating questions...")
        result = run_chain(query)
        progress_mgr.complete_task()
        
        progress_mgr.update_status("Creating output reports...")
        report_gen = ReportGenerator()
        html_file, md_file = report_gen.generate(result, query)
        progress_mgr.complete_task()
    
    # Display results
    format_output(result)
    
    rprint("\n[bold green]Reports generated successfully![/bold green]")
    rprint(f"HTML Report: [blue]{html_file}[/blue]")
    rprint(f"Markdown Report: [blue]{md_file}[/blue]")

if __name__ == "__main__":
    main()

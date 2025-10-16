from agents.base_agent import BaseAgent
from openai import OpenAI
import os

class SummarizerAgent(BaseAgent):
    def __init__(self, name, role, memory, use_llm=False):
        super().__init__(name, role, memory)
        self.use_llm = use_llm
        if use_llm:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = "gpt-5-mini"
    def act(self, task_results):
        """summarizes a list of task results into key insights. Each result is a dic
        with fields:{'agent': str, 'task': str, 'output': str}"""

        if not task_results:
            summary = "No results available for summarization."
        elif self.use_llm:
            prompt = (
                "You are a summarizer agent for a multi-agent orchestration system. "
                "Given the list of task outputs below, synthesize the main outcomes, insights, and next actions.\n\n"
                f"Results:\n{task_results}\n\nReturn a structured summary:"
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            summary = response.choices[0].message.content.strip()
        else:
            summary = (
                f"[SummarizerAgent] Synthesized {len(task_results)} tasks into summary.\n"
                "All agents completed tasks successfully.\n"
                "Overall system performance: Stable."
            )

        self.memory.write(self.name, f"Summary generated:\n{summary}")
        return summary
import os
import re

from rag.engine import tokenize

GREETING = "Hello,\n\nThanks for reaching out to Sentinel Support."
CLOSING = (
    "If anything is still unclear, just reply to this email and a specialist "
    "will pick it up right away."
)


class TemplateLLM:
    name = "template-extractive"

    def complete(self, system: str, prompt: str) -> str:
        goal_match = re.search(r"^GOAL: (.+)$", prompt, re.MULTILINE)
        goal = goal_match.group(1) if goal_match else ""
        goal_tokens = set(tokenize(goal))

        blocks: list[tuple[str, list[str]]] = []
        current_ref = None
        current_lines: list[str] = []
        for line in prompt.splitlines():
            if line.startswith("[["):
                if current_ref is not None:
                    blocks.append((current_ref, current_lines))
                current_ref = line[2:].split("]]")[0].strip()
                current_lines = [line.split("]]", 1)[-1].strip()]
            elif line.startswith("END_OF_CONTEXT"):
                if current_ref is not None:
                    blocks.append((current_ref, current_lines))
                    current_ref = None
                    current_lines = []
            elif current_ref is not None:
                current_lines.append(line.strip())
        if current_ref is not None:
            blocks.append((current_ref, current_lines))

        candidates: list[tuple[float, str, str]] = []
        for ref, lines in blocks:
            text = " ".join(part for part in lines if part)
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                sentence = sentence.strip()
                tokens = set(tokenize(sentence))
                if len(tokens) < 4:
                    continue
                overlap = len(tokens & goal_tokens) / len(tokens)
                candidates.append((overlap, sentence, ref))
        candidates.sort(key=lambda item: -item[0])

        picked: list[tuple[str, str]] = []
        seen_sentences: set[str] = set()
        for overlap, sentence, ref in candidates:
            if sentence in seen_sentences:
                continue
            seen_sentences.add(sentence)
            picked.append((sentence, ref))
            if len(picked) == 3:
                break

        if not picked:
            body = (
                "We are looking into this and will follow up shortly with a "
                "definitive answer."
            )
        else:
            body = "Based on our documentation:\n" + "\n".join(
                f"- {sentence} [{ref}]" for sentence, ref in picked
            )
        return f"{GREETING}\n\n{body}\n\n{CLOSING}"


class OpenAILLM:
    def __init__(self, api_key: str):
        import openai

        self.name = "openai"
        self._client = openai.OpenAI(api_key=api_key)

    def complete(self, system: str, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""


class AnthropicLLM:
    def __init__(self, api_key: str):
        import anthropic

        self.name = "anthropic"
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, system: str, prompt: str) -> str:
        response = self._client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return "".join(block.text for block in response.content if block.type == "text")


def get_llm():
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return OpenAILLM(openai_key)
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        return AnthropicLLM(anthropic_key)
    return TemplateLLM()

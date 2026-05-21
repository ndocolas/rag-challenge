import ReactMarkdown from "react-markdown";
import type { ChatMessage } from "../../types/chat";
import { CitationChips } from "./CitationChips";
import { ThinkingIndicator } from "./ThinkingIndicator";

export function AssistantMessage({ message }: { message: ChatMessage }) {
  const { content, citations, currentTool, status } = message;

  return (
    <div className="flex justify-start px-4">
      <div className="max-w-[80%]">
        {currentTool && <ThinkingIndicator label={currentTool} />}
        {content && (
          <div className="prose prose-invert prose-sm max-w-none text-zinc-200 leading-relaxed">
            <ReactMarkdown>{content}</ReactMarkdown>
          </div>
        )}
        {!content && status === "streaming" && !currentTool && (
          <span className="inline-block w-2 h-4 bg-zinc-400 animate-pulse rounded-sm" />
        )}
        {status === "done" && citations && citations.length > 0 && (
          <CitationChips citations={citations} />
        )}
      </div>
    </div>
  );
}

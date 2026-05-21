import { useEffect, useRef } from "react";
import type { ChatMessage } from "../../types/chat";
import { AssistantMessage } from "./AssistantMessage";
import { UserMessage } from "./UserMessage";

export function MessageList({ messages }: { messages: ChatMessage[] }) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto py-6 space-y-4">
      {messages.map((m) =>
        m.role === "user" ? (
          <UserMessage key={m.id} content={m.content} />
        ) : (
          <AssistantMessage key={m.id} message={m} />
        ),
      )}
      <div ref={bottomRef} />
    </div>
  );
}

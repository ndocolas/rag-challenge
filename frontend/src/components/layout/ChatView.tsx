import { Menu } from "lucide-react";
import { useEffect, useRef } from "react";
import { useAppContext } from "../../context/AppContext";
import { useChatStream } from "../../hooks/useChatStream";
import { ChatInput } from "../chat/ChatInput";
import { EmptyState } from "../chat/EmptyState";
import { MessageList } from "../chat/MessageList";

export function ChatView() {
  const {
    activeSessionId,
    setActiveSessionId,
    toggleSidebar,
    sessions,
    refreshSessions,
    addOptimisticSession,
  } = useAppContext();
  const { messages, isStreaming, send, stop, loadHistory, reset } = useChatStream();
  const skipHistoryRef = useRef(false);

  useEffect(() => {
    if (!activeSessionId) {
      reset();
      return;
    }
    if (skipHistoryRef.current) {
      skipHistoryRef.current = false;
      return;
    }
    void (async () => {
      reset();
      await loadHistory(activeSessionId);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSessionId]);

  async function handleSend(text: string) {
    if (!activeSessionId) return;
    if (!sessions.some((s) => s.session_id === activeSessionId)) {
      addOptimisticSession({
        session_id: activeSessionId,
        title: text.slice(0, 60),
        created_at: new Date().toISOString(),
      });
    }
    await send(activeSessionId, text);
    void refreshSessions();
  }

  async function handleSuggestion(text: string) {
    const newId = crypto.randomUUID();
    skipHistoryRef.current = true;
    setActiveSessionId(newId);
    addOptimisticSession({
      session_id: newId,
      title: text.slice(0, 60),
      created_at: new Date().toISOString(),
    });
    await send(newId, text);
    void refreshSessions();
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-2 px-4 py-3 border-b border-zinc-800 shrink-0">
        <button
          onClick={toggleSidebar}
          className="text-zinc-500 hover:text-zinc-200 transition-colors p-1 rounded"
          aria-label="Toggle sidebar"
        >
          <Menu size={18} />
        </button>
      </div>

      {!activeSessionId ? (
        <EmptyState onSuggestion={handleSuggestion} />
      ) : (
        <>
          <MessageList messages={messages} />
          <ChatInput isStreaming={isStreaming} onSend={handleSend} onStop={stop} />
        </>
      )}
    </div>
  );
}

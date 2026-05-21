import { ArrowUp, Square } from "lucide-react";
import { type KeyboardEvent, useCallback, useRef } from "react";

interface ChatInputProps {
  isStreaming: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
}

export function ChatInput({ isStreaming, onSend, onStop }: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleInput = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
  }, []);

  const handleSend = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    const text = el.value.trim();
    if (!text || isStreaming) return;
    el.value = "";
    el.style.height = "auto";
    onSend(text);
  }, [isStreaming, onSend]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  return (
    <div className="p-4 border-t border-zinc-800 bg-zinc-950">
      <div className="flex items-end gap-2 max-w-3xl mx-auto bg-zinc-900 border border-zinc-700 rounded-xl px-3 py-2 focus-within:border-zinc-500 transition-colors">
        <textarea
          ref={textareaRef}
          onInput={handleInput}
          onKeyDown={handleKeyDown}
          placeholder="Mensagem..."
          rows={1}
          maxLength={4000}
          className="flex-1 resize-none bg-transparent text-sm text-zinc-100 placeholder:text-zinc-500 outline-none leading-relaxed min-h-[24px] max-h-[140px] py-0.5"
        />
        <button
          onClick={isStreaming ? onStop : handleSend}
          className={`shrink-0 flex items-center justify-center w-7 h-7 rounded-lg transition-colors ${
            isStreaming
              ? "bg-zinc-600 hover:bg-zinc-500 text-zinc-200"
              : "bg-zinc-200 hover:bg-white text-zinc-900 disabled:opacity-40"
          }`}
        >
          {isStreaming ? <Square size={12} /> : <ArrowUp size={14} />}
        </button>
      </div>
      <p className="text-center text-[10px] text-zinc-600 mt-2">
        As respostas são geradas por IA e podem conter erros. Consulte sempre um
        profissional de saúde.
      </p>
    </div>
  );
}

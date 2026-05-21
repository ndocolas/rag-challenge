import { useCallback, useRef, useState } from "react";
import { toast } from "sonner";
import { fetchHistory, httpErrorMessage, streamChat } from "../lib/api";
import type { ChatMessage } from "../types/chat";

const TOOL_LABELS: Record<string, string> = {
  buscar_filiais: "Buscando filiais...",
  detalhes_filial: "Consultando filial...",
  listar_cidades_atendidas: "Buscando cidades...",
  search_bulas: "Consultando bula...",
  buscar_bulas: "Consultando bula...",
};

function toolLabel(name: string): string {
  return TOOL_LABELS[name] ?? "Pesquisando...";
}

function applyFrame(
  messages: ChatMessage[],
  assistantId: string,
  updater: (m: ChatMessage) => ChatMessage,
): ChatMessage[] {
  return messages.map((m) => (m.id === assistantId ? updater(m) : m));
}

export function useChatStream() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const send = useCallback(async (sessionId: string, text: string): Promise<void> => {
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      status: "done",
    };
    const assistantId = crypto.randomUUID();
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      status: "streaming",
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      for await (const frame of streamChat(sessionId, text, controller.signal)) {
        if (frame.event === "token") {
          setMessages((prev) =>
            applyFrame(prev, assistantId, (m) => ({
              ...m,
              content: m.content + frame.data,
              currentTool: undefined,
            })),
          );
        } else if (frame.event === "tool_call") {
          const { name } = JSON.parse(frame.data) as { name: string };
          setMessages((prev) =>
            applyFrame(prev, assistantId, (m) => ({
              ...m,
              currentTool: toolLabel(name),
            })),
          );
        } else if (frame.event === "tool_result") {
          // label stays visible until first token clears it
        } else if (frame.event === "sources") {
          const { citations } = JSON.parse(frame.data) as { citations: ChatMessage["citations"] };
          setMessages((prev) =>
            applyFrame(prev, assistantId, (m) => ({ ...m, citations })),
          );
        } else if (frame.event === "done") {
          setMessages((prev) =>
            applyFrame(prev, assistantId, (m) => ({
              ...m,
              status: "done",
              currentTool: undefined,
            })),
          );
        } else if (frame.event === "error") {
          const parsed = JSON.parse(frame.data) as { message?: string };
          const msg = parsed.message ?? "Ocorreu um erro.";
          setMessages((prev) =>
            applyFrame(prev, assistantId, (m) => ({
              ...m,
              status: "error",
              content:
                m.content +
                `\n\n_Ocorreu um erro. A resposta pode estar incompleta._`,
              currentTool: undefined,
            })),
          );
          void msg;
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === "AbortError") {
        setMessages((prev) =>
          applyFrame(prev, assistantId, (m) => ({
            ...m,
            status: "done",
            content: m.content
              ? m.content + "\n\n_Resposta interrompida._"
              : "_Resposta interrompida._",
            currentTool: undefined,
          })),
        );
      } else {
        const errObj = err as { status?: number };
        const msg =
          errObj.status !== undefined
            ? httpErrorMessage(errObj.status)
            : "Erro de conexão. Tente novamente.";
        toast.error(msg);
        setMessages((prev) => prev.filter((m) => m.id !== assistantId));
      }
    } finally {
      setIsStreaming(false);
      abortRef.current = null;
    }
  }, []);

  const stop = useCallback((): void => {
    abortRef.current?.abort();
  }, []);

  const loadHistory = useCallback(async (sessionId: string): Promise<boolean> => {
    try {
      const history = await fetchHistory(sessionId);
      setMessages(history);
      return true;
    } catch (err: unknown) {
      const errObj = err as { status?: number };
      if (errObj.status === 404) return false;
      toast.error("Erro ao carregar histórico.");
      return false;
    }
  }, []);

  const reset = useCallback((): void => {
    setMessages([]);
  }, []);

  return { messages, isStreaming, send, stop, loadHistory, reset };
}

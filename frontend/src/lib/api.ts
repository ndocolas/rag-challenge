import type { ChatMessage, Session, SSEFrame } from "../types/chat";

const API_URL =
  (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";

// --- SSE parser ---

export function parseSSEFrame(raw: string): SSEFrame | null {
  const lines = raw.split("\n");
  let event = "message";
  const dataLines: string[] = [];
  for (const line of lines) {
    if (line.startsWith("event: ")) event = line.slice(7).trim();
    else if (line.startsWith("data: ")) dataLines.push(line.slice(6));
  }
  if (!dataLines.length) return null;
  return { event: event as SSEFrame["event"], data: dataLines.join("\n") };
}

// --- Error helpers ---

export function httpErrorMessage(status: number): string {
  if (status === 429) return "Muitas mensagens. Aguarde um momento.";
  if (status === 409) return "Aguarde a resposta anterior terminar.";
  if (status === 422) return "Mensagem inválida. Verifique o conteúdo.";
  if (status === 413) return "Mensagem muito longa.";
  return "Erro de conexão. Tente novamente.";
}

// --- Sessions API ---

export async function fetchSessions(): Promise<Session[]> {
  try {
    const res = await fetch(`${API_URL}/sessions`);
    if (!res.ok) return [];
    return res.json() as Promise<Session[]>;
  } catch {
    return [];
  }
}

export async function fetchHistory(sessionId: string): Promise<ChatMessage[]> {
  const res = await fetch(
    `${API_URL}/sessions/${encodeURIComponent(sessionId)}/history`,
  );
  if (res.status === 404) throw Object.assign(new Error("not_found"), { status: 404 });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json() as Promise<ChatMessage[]>;
}

// --- SSE streaming ---

export async function* streamChat(
  sessionId: string,
  message: string,
  signal: AbortSignal,
): AsyncGenerator<SSEFrame> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
    signal,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const msg =
      (body as { error?: { message?: string } }).error?.message ??
      httpErrorMessage(res.status);
    throw Object.assign(new Error(msg), { status: res.status, body });
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: false });
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      if (!part.trim()) continue;
      const frame = parseSSEFrame(part);
      if (frame) yield frame;
    }
  }
}

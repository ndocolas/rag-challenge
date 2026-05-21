export type MessageRole = "user" | "assistant";
export type MessageStatus = "done" | "streaming" | "error";
export type SSEEventType =
  | "trace_id"
  | "token"
  | "tool_call"
  | "tool_result"
  | "sources"
  | "done"
  | "error";

export interface Citation {
  bula_id: string;
  med_name: string;
  med_variant?: string;
  section_canonical?: string;
  section_label: string;
  source_page?: number;
  snippet: string;
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  citations?: Citation[];
  currentTool?: string;
  status: MessageStatus;
}

export interface Session {
  session_id: string;
  title: string;
  created_at: string;
}

export interface SSEFrame {
  event: SSEEventType;
  data: string;
}

export interface ApiErrorEnvelope {
  error: {
    code: string;
    message: string;
    status_code: number;
    trace_id?: string;
  };
}

import { formatDistanceToNow } from "date-fns";
import { ptBR } from "date-fns/locale";
import { PenSquare } from "lucide-react";
import { useAppContext } from "../../context/AppContext";
import type { Session } from "../../types/chat";

function SessionItem({
  session,
  active,
  onClick,
}: {
  session: Session;
  active: boolean;
  onClick: () => void;
}) {
  const relativeTime = formatDistanceToNow(new Date(session.created_at), {
    addSuffix: true,
    locale: ptBR,
  });

  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-2 rounded-md text-sm transition-colors ${
        active
          ? "bg-zinc-700 text-zinc-100"
          : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
      }`}
    >
      <p className="truncate font-medium text-zinc-200">
        {session.title || "Nova conversa"}
      </p>
      <p className="text-xs text-zinc-500 mt-0.5">{relativeTime}</p>
    </button>
  );
}

export function Sidebar() {
  const { activeSessionId, setActiveSessionId, sessions } = useAppContext();

  function handleNewChat() {
    setActiveSessionId(crypto.randomUUID());
  }

  return (
    <div className="h-full bg-zinc-900 flex flex-col w-64 border-r border-zinc-800">
      <div className="p-3 border-b border-zinc-800">
        <button
          onClick={handleNewChat}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-md text-sm text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100 transition-colors"
        >
          <PenSquare size={14} />
          Nova conversa
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-0.5">
        {sessions.map((s: Session) => (
          <SessionItem
            key={s.session_id}
            session={s}
            active={s.session_id === activeSessionId}
            onClick={() => setActiveSessionId(s.session_id)}
          />
        ))}
      </div>
    </div>
  );
}

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { fetchSessions } from "../lib/api";
import type { Session } from "../types/chat";

interface AppContextValue {
  activeSessionId: string | null;
  setActiveSessionId: (id: string | null) => void;
  sidebarOpen: boolean;
  toggleSidebar: () => void;
  sessions: Session[];
  refreshSessions: () => Promise<void>;
  addOptimisticSession: (session: Session) => void;
}

const AppContext = createContext<AppContextValue | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sessions, setSessions] = useState<Session[]>([]);

  const toggleSidebar = useCallback(() => setSidebarOpen((v) => !v), []);

  const refreshSessions = useCallback(async () => {
    const data = await fetchSessions();
    setSessions(data);
  }, []);

  const addOptimisticSession = useCallback((session: Session) => {
    setSessions((prev) =>
      prev.some((s) => s.session_id === session.session_id)
        ? prev
        : [session, ...prev],
    );
  }, []);

  useEffect(() => {
    void refreshSessions();
  }, [refreshSessions]);

  return (
    <AppContext.Provider
      value={{
        activeSessionId,
        setActiveSessionId,
        sidebarOpen,
        toggleSidebar,
        sessions,
        refreshSessions,
        addOptimisticSession,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext(): AppContextValue {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useAppContext must be used within AppProvider");
  return ctx;
}

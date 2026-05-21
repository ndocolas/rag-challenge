import { useCallback, useEffect, useState } from "react";
import { fetchSessions } from "../lib/api";
import type { Session } from "../types/chat";

export function useSessionList() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async (): Promise<void> => {
    const data = await fetchSessions();
    setSessions(data);
    setLoading(false);
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { sessions, loading, refresh };
}

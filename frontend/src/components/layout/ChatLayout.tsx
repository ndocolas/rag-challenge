import { Toaster } from "sonner";
import { useAppContext } from "../../context/AppContext";
import { ChatView } from "./ChatView";
import { Sidebar } from "./Sidebar";

export function ChatLayout() {
  const { sidebarOpen } = useAppContext();

  return (
    <div className="flex h-screen bg-zinc-950 overflow-hidden">
      <div
        className={`transition-all duration-200 ease-in-out shrink-0 ${
          sidebarOpen ? "w-64" : "w-0"
        } overflow-hidden`}
      >
        <Sidebar />
      </div>
      <main className="flex-1 flex flex-col min-w-0">
        <ChatView />
      </main>
      <Toaster theme="dark" position="top-right" />
    </div>
  );
}

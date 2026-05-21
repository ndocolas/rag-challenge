import { AppProvider } from "./context/AppContext";
import { ChatLayout } from "./components/layout/ChatLayout";

export default function App() {
  return (
    <AppProvider>
      <ChatLayout />
    </AppProvider>
  );
}

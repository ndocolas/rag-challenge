# Task 12 — Frontend polish final

## Objetivo

UX pronta para apresentação: estados de erro/loading bonitos, estado vazio,
responsivo mobile, microcopy revisado, animações sutis, atalhos teclado, persistência
e reset de sessão, README do frontend, screenshots no README raiz.

## Pré-requisitos

- Task 11 (UI integrada funcional).

## Subtarefas

### 1. ErrorBoundary global

`src/components/ErrorBoundary.tsx`:

```tsx
import { Component, ReactNode } from "react";

interface Props { children: ReactNode; }
interface State { hasError: boolean; message: string; }

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, message: "" };

  static getDerivedStateFromError(e: Error): State {
    return { hasError: true, message: e.message };
  }

  componentDidCatch(error: Error, info: any) {
    console.error("ErrorBoundary:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex h-screen items-center justify-center bg-red-50">
          <div className="rounded-lg border border-red-200 bg-white p-6 shadow">
            <h2 className="mb-2 font-semibold text-red-800">Algo deu errado</h2>
            <p className="mb-4 text-sm text-red-600">{this.state.message}</p>
            <button
              onClick={() => location.reload()}
              className="rounded bg-red-600 px-3 py-1 text-sm text-white"
            >
              Recarregar
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
```

Envolver em `main.tsx`:
```tsx
import { ErrorBoundary } from "@/components/ErrorBoundary";
<ErrorBoundary><App /></ErrorBoundary>
```

### 2. Toast pra erros transient

shadcn já adicionou `toast`. Em `useChatStream`, no caso `ERROR`:
```tsx
import { toast } from "sonner"; // ou @/components/ui/use-toast conforme versão
// ...
case "ERROR":
  toast.error("Erro no chat", { description: action.message });
  return { ...state, isStreaming: false, error: action.message };
```

Adicionar `<Toaster />` em `App.tsx`.

### 3. Loading skeleton + "pensando..."

Renderizar dentro da bolha assistant quando `isStreaming && content === ""`:

```tsx
// em MessageBubble.tsx, antes de StreamingText:
{turn.isStreaming && !turn.content && !turn.toolCalls?.length && (
  <div className="flex gap-1">
    <span className="h-2 w-2 animate-bounce rounded-full bg-gray-400 [animation-delay:0ms]" />
    <span className="h-2 w-2 animate-bounce rounded-full bg-gray-400 [animation-delay:120ms]" />
    <span className="h-2 w-2 animate-bounce rounded-full bg-gray-400 [animation-delay:240ms]" />
  </div>
)}
```

### 4. Estado vazio (welcome)

`src/components/EmptyState.tsx`:

```tsx
const SUGGESTIONS = [
  "Quais lojas em Curitiba têm Panvel Clinic e atendem 24h?",
  "Quais as contraindicações da Ritalina?",
  "Posologia do pantoprazol em adultos",
  "Vocês têm filial em Foz do Iguaçu?",
];

export function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="mx-auto flex max-w-xl flex-col items-center gap-6 py-12 text-center">
      <div className="text-4xl">💊</div>
      <div>
        <h1 className="text-xl font-semibold">Assistente Panvel</h1>
        <p className="mt-1 text-sm text-gray-600">
          Tire dúvidas sobre medicamentos e nossas lojas no Paraná.
        </p>
      </div>
      <div className="flex flex-col gap-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => onPick(s)}
            className="rounded-lg border bg-white px-4 py-2 text-left text-sm hover:bg-gray-50"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
```

Renderizar em `MessageList` quando `turns.length === 0`.

### 5. Responsivo mobile — SourcesPanel vira drawer

Usar `Sheet` do shadcn (`npx shadcn@latest add sheet`):

```tsx
// no ChatWindow.tsx, em mobile mostrar botão "Fontes (N)"
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";

// quando md:hidden:
<Sheet>
  <SheetTrigger asChild>
    <Button variant="outline" className="md:hidden">
      Fontes ({citations.length})
    </Button>
  </SheetTrigger>
  <SheetContent side="right">
    <SourcesPanelContent citations={citations} />
  </SheetContent>
</Sheet>
```

Refatorar `SourcesPanel` em 2: `SourcesPanelContent` (puro conteúdo) + `SourcesPanel`
(wrapper desktop).

### 6. Atalhos e UX de input

- Enter envia / Shift+Enter quebra linha (já feito Task 11)
- Auto-resize do textarea (até max-height)
- Counter de caracteres se >3500/4000
- Disable button durante streaming + spinner inline

### 7. Botão "limpar conversa"

No header ou canto superior:

```tsx
<Button
  variant="ghost"
  size="sm"
  onClick={() => {
    if (confirm("Limpar conversa?")) {
      localStorage.removeItem(SESSION_KEY);
      location.reload();
    }
  }}
>
  Limpar
</Button>
```

### 8. Animações sutis

Tailwind animations + `transition-*`:
- Fade-in de mensagem nova: `animate-in fade-in slide-in-from-bottom-2 duration-300`
  (com `tailwindcss-animate` plugin já incluído em shadcn)
- Slide do sheet (nativo do Radix)
- Cursor pulsante (já feito)

### 9. Microcopy

Revisar todos os textos:
- Placeholder input: "Pergunte sobre medicamentos ou filiais..."
- Erro genérico: "Não consegui processar agora. Tente novamente em alguns segundos."
- Erro de rede: "Sem conexão com o servidor."
- Sem fontes: "Esta resposta não usou consulta a bulas."
- Disclaimer médico (renderizar pequeno após resposta de bula):
  "ℹ️ Esta informação não substitui orientação médica."

### 10. `frontend/README.md`

```markdown
# Panvel Frontend

UI React + Vite + TS + Tailwind + shadcn/ui para o assistente Panvel.

## Dev

```bash
npm install
cp .env.example .env  # ajuste VITE_API_URL
npm run dev
```

Abra http://localhost:5173

## Build

```bash
npm run build
npm run preview
```

## Docker

```bash
docker build -t panvel-frontend .
docker run -p 5173:80 -e VITE_API_URL=http://localhost:8000 panvel-frontend
```

## Arquitetura

- `src/lib/api.ts` — cliente SSE (fetch + ReadableStream parser)
- `src/hooks/useChatStream.ts` — useReducer com eventos
- `src/components/` — UI (ChatWindow, MessageBubble, ToolCallBadge, SourcesPanel...)
- `src/types/chat.ts` — schemas dos eventos SSE

Eventos consumidos: `trace_id`, `token`, `tool_call`, `tool_result`, `sources`,
`done`, `error`.
```

### 11. Screenshots no README raiz

Tirar 3 screenshots:
1. Estado vazio com sugestões
2. Turno com tool calls (filiais) + resposta
3. Turno RAG com SourcesPanel populado

Salvar em `docs/screenshots/`. Incluir no README raiz:

```markdown
## Demo

### Estado inicial
![empty](docs/screenshots/empty.png)

### Consulta a filiais (tool calling)
![tools](docs/screenshots/tools.png)

### Pergunta farmacológica (RAG)
![rag](docs/screenshots/rag.png)
```

## Verificação

Checklist da demo (5 min):

- [ ] Abre frontend → vê estado vazio com 4 sugestões
- [ ] Clica sugestão → vira mensagem do user, assistant começa a streamar com "..."
- [ ] Tool call badge aparece em amber → vira verde
- [ ] Resposta cita filial corretamente
- [ ] Pergunta farmacológica popula SourcesPanel
- [ ] Click em source expande snippet completo
- [ ] Mobile: SourcesPanel vira drawer ao clicar "Fontes (4)"
- [ ] Provoca erro (Qdrant offline) → toast vermelho explicativo
- [ ] "Limpar" reseta sessão
- [ ] Refresh: nova sessão (localStorage limpo se clicado limpar)
- [ ] Refresh: mesma sessão (se não clicado limpar)
- [ ] Shift+Enter quebra linha; Enter envia
- [ ] Resposta de bula tem disclaimer médico

## Gotchas

- shadcn `sonner` vs `use-toast`: dependendo da versão, escolher um. Sonner é mais novo.
- Animations exigem `tailwindcss-animate` (já vem com shadcn init).
- Auto-resize textarea: usar lib (`react-textarea-autosize`) ou ref + scrollHeight.
- Mobile drawer com side="right": OK no Radix Sheet.
- Toast position: top-right default; ajustar conforme preferência.
- Screenshot: usar dark mode? MVP fica em light.
- Acessibilidade: `aria-live="polite"` na lista de mensagens para leitores de tela.

const SUGGESTED = [
  "Quais são as contraindicações do Ritalina?",
  "Qual a posologia do Advil para adultos?",
  "Pode tomar Rivotril com álcool?",
];

export function EmptyState({ onSuggestion }: { onSuggestion: (text: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-6 px-4 text-center">
      <div>
        <h1 className="text-2xl font-semibold text-zinc-100">Assistente Panvel</h1>
        <p className="text-zinc-500 mt-1 text-sm">Como posso ajudar?</p>
      </div>
      <div className="flex flex-col gap-2 w-full max-w-sm">
        {SUGGESTED.map((q) => (
          <button
            key={q}
            onClick={() => onSuggestion(q)}
            className="text-left px-4 py-2.5 rounded-lg border border-zinc-800 text-sm text-zinc-400 hover:border-zinc-600 hover:text-zinc-200 transition-colors"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

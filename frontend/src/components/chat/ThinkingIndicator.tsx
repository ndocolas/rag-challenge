export function ThinkingIndicator({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 text-xs text-zinc-500 mb-1">
      <span className="flex gap-0.5">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="w-1 h-1 rounded-full bg-zinc-500 animate-bounce"
            style={{ animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </span>
      {label}
    </div>
  );
}

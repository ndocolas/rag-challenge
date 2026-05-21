export function UserMessage({ content }: { content: string }) {
  return (
    <div className="flex justify-end px-4">
      <div className="max-w-[70%] bg-zinc-800 text-zinc-100 rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap break-words">
        {content}
      </div>
    </div>
  );
}

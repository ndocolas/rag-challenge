import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover";
import type { Citation } from "../../types/chat";

function CitationPopover({
  citation,
  index,
}: {
  citation: Citation;
  index: number;
}) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button className="inline-flex items-center justify-center w-5 h-5 text-[10px] font-medium rounded bg-zinc-700 text-zinc-300 hover:bg-zinc-600 hover:text-zinc-100 transition-colors cursor-pointer">
          {index + 1}
        </button>
      </PopoverTrigger>
      <PopoverContent
        side="top"
        className="w-72 bg-zinc-800 border-zinc-700 text-zinc-200 p-3 text-xs"
      >
        <p className="font-semibold text-zinc-100 truncate">
          {citation.med_name}
          {citation.med_variant ? ` — ${citation.med_variant}` : ""}
        </p>
        <p className="text-zinc-400 mt-0.5 mb-2">{citation.section_label}</p>
        <p className="text-zinc-300 leading-relaxed line-clamp-5">
          {citation.snippet.length > 200
            ? citation.snippet.slice(0, 200) + "…"
            : citation.snippet}
        </p>
        {citation.source_page && (
          <p className="text-zinc-500 mt-1">Página {citation.source_page}</p>
        )}
      </PopoverContent>
    </Popover>
  );
}

export function CitationChips({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;
  return (
    <div className="flex flex-wrap gap-1 mt-2">
      {citations.map((c, i) => (
        <CitationPopover key={`${c.bula_id}-${i}`} citation={c} index={i} />
      ))}
    </div>
  );
}

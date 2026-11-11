import { Card } from "../../components/ui/Card";
import { EmptyState } from "../../components/ui/states";
import { AttentionDemo } from "./demos/AttentionDemo";
import { CausalMaskDemo } from "./demos/CausalMaskDemo";
import { TemperatureDemo } from "./demos/TemperatureDemo";
import { TokenizerDemo } from "./demos/TokenizerDemo";
import type { DemoKey, LearnModule } from "./modules";

function Demo({ demo, modelId }: { demo: DemoKey; modelId: string | null }) {
  switch (demo) {
    case "tokenizer": return <TokenizerDemo modelId={modelId} />;
    case "causal-mask": return <CausalMaskDemo />;
    case "attention": return <AttentionDemo modelId={modelId} />;
    case "temperature": return <TemperatureDemo modelId={modelId} />;
  }
}

export function ModuleView({ module, modelId }: { module: LearnModule; modelId: string | null }) {
  if (!module.written) {
    return (
      <EmptyState title="Not written yet">
        This module has not been written yet. {module.summary}
      </EmptyState>
    );
  }
  return (
    <div className="space-y-4">
      <Card title={module.title}>
        <p className="mb-3 text-sm text-slate-300">{module.summary}</p>
        {module.sections.map((s) => (
          <section key={s.heading} className="mb-3">
            <h3 className="mb-1 text-sm font-semibold text-slate-100">{s.heading}</h3>
            {s.paragraphs.map((p, i) => (
              <p key={i} className="mb-2 max-w-3xl text-sm leading-relaxed text-slate-400">{p}</p>
            ))}
          </section>
        ))}
        {module.equations.map((e) => (
          <div key={e.label} className="mb-2">
            <div className="mb-1 text-xs text-slate-500">{e.label}</div>
            <pre className="overflow-x-auto rounded border border-navy-600 bg-navy-900 p-2 font-mono text-xs text-cyan">
              {e.text}
            </pre>
          </div>
        ))}
      </Card>
      {module.demo && (
        <Card title="Try it">
          <Demo demo={module.demo} modelId={modelId} />
        </Card>
      )}
    </div>
  );
}
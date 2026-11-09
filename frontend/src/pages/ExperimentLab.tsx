import { useEffect, useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

import { Card } from "../components/ui/Card";
import { EmptyState, ErrorState } from "../components/ui/states";
import {
  useExperimentCompare, useExperimentDefs, useExperimentRun, useExperimentRuns,
  useRunExperiments, useStopJob, useTrainingJobs,
} from "../hooks/useApi";
import { experimentCsvUrl } from "../services/api";
import { DASH, formatDuration, formatInt, formatLoss } from "../utils/format";
import { mergeCurves, spread } from "../utils/experiments";

const FIELD = "rounded border border-navy-600 bg-navy-900 px-2 py-1.5 text-sm font-mono";
const COLORS = ["#8b5cf6", "#22d3ee", "#f59e0b", "#34d399", "#f87171", "#60a5fa"];
const TIP = { background: "#111833", border: "1px solid #26325c" };

export function ExperimentLab() {
  const defs = useExperimentDefs();
  const jobs = useTrainingJobs();
  const activeJob = jobs.data?.jobs.find(
    (j) => j.kind === "experiment" && (j.status === "QUEUED" || j.status === "RUNNING"),
  );
  const live = activeJob !== undefined;
  const runs = useExperimentRuns(live);
  const launch = useRunExperiments();
  const stop = useStopJob();

  const [picked, setPicked] = useState<string[]>([]);
  const [maxIters, setMaxIters] = useState(1000);
  const [seed, setSeed] = useState(1337);
  const [force, setForce] = useState(false);
  const [batchId, setBatchId] = useState<string | null>(null);
  const [expKey, setExpKey] = useState("");
  const [cmp, setCmp] = useState<string[]>([]);

  useEffect(() => {
    if (batchId === null && runs.data && runs.data.runs.length > 0) {
      setBatchId(runs.data.runs[0]?.batch_id ?? null);
    }
  }, [runs.data, batchId]);
  useEffect(() => {
    if (expKey === "" && defs.data && defs.data.experiments.length > 0) {
      setExpKey(defs.data.experiments[0]?.key ?? "");
    }
  }, [defs.data, expKey]);

  const detail = useExperimentRun(batchId, live);
  const compare = useExperimentCompare(cmp);
  const experiment = defs.data?.experiments.find((e) => e.key === expKey);
  const names = useMemo(() => experiment?.variants.map((v) => v.name) ?? [], [experiment]);
  const rows = detail.data?.rows ?? [];
  const shown = rows.filter((r) => names.includes(r.run));
  const curveData = useMemo(
    () => (detail.data ? mergeCurves(detail.data.curves, names, "val_loss") : []),
    [detail.data, names],
  );
  const toggle = (list: string[], v: string, max: number) =>
    list.includes(v) ? list.filter((x) => x !== v) : list.length < max ? [...list, v] : list;

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold text-slate-50">Experiment Lab</h1>
      <p className="max-w-3xl text-sm text-slate-400">
        Runs the experiments defined in <code className="font-mono">experiments/definitions.py</code>{" "}
        through the training worker. Every GPT variant changes exactly one setting from the same
        base model. Finished variants are skipped, so rerunning a batch only fills gaps.
      </p>

      <Card title="Run experiments">
        {defs.error && <ErrorState error={defs.error} />}
        <div className="flex flex-wrap gap-3 text-sm">
          {defs.data?.experiments.map((e) => (
            <label key={e.key} className="flex items-center gap-1.5 text-slate-300">
              <input type="checkbox" checked={picked.includes(e.key)}
                onChange={() => setPicked(toggle(picked, e.key, 99))} />
              {e.title}
            </label>
          ))}
        </div>
        <p className="mt-1 text-xs text-slate-500">Nothing ticked runs all experiments.</p>
        <div className="mt-3 flex flex-wrap items-end gap-3 text-sm">
          <label className="text-slate-400">Iterations per run
            <input type="number" min={1} className={`${FIELD} mt-1 block w-32`} value={maxIters}
              onChange={(e) => setMaxIters(Number(e.target.value))} />
          </label>
          <label className="text-slate-400">Seed
            <input type="number" min={0} className={`${FIELD} mt-1 block w-32`} value={seed}
              onChange={(e) => setSeed(Number(e.target.value))} />
          </label>
          <label className="flex items-center gap-1.5 text-slate-400">
            <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
            Rerun finished variants
          </label>
          <button
            className="rounded bg-violet px-3 py-1.5 font-medium text-white hover:bg-violet-soft disabled:opacity-50"
            disabled={launch.isPending || live || maxIters < 1}
            onClick={() => launch.mutate(
              { experiments: picked.length ? picked : null, max_iters: maxIters, seed, force },
              { onSuccess: () => setBatchId(`it${maxIters}_s${seed}`) },
            )}>
            {launch.isPending ? "Queuing..." : "Run"}
          </button>
        </div>
        {launch.error && <div className="mt-2"><ErrorState error={launch.error} /></div>}
        {activeJob && (
          <div className="mt-3 rounded border border-navy-600 p-3 text-sm" role="status">
            <div className="flex items-center justify-between">
              <span>
                {activeJob.run_name}: {activeJob.status}
                {typeof activeJob.detail?.["current"] === "string" ? `, running ${activeJob.detail["current"]}` : ""}
              </span>
              <button className="rounded border border-navy-600 px-2 py-1 text-xs hover:bg-navy-700 disabled:opacity-50"
                disabled={activeJob.stop_requested || stop.isPending}
                onClick={() => stop.mutate(activeJob.job_id)}>Stop</button>
            </div>
            <div className="mt-2 h-2 overflow-hidden rounded bg-navy-700">
              <div className="h-full bg-violet" style={{ width: `${(activeJob.progress ?? 0) * 100}%` }} />
            </div>
            <p className="mt-1 text-xs text-slate-500">Progress counts finished variants.</p>
          </div>
        )}
      </Card>

      <Card title="Batches">
        {runs.error && <ErrorState error={runs.error} />}
        {runs.data && runs.data.runs.length === 0 && (
          <EmptyState title="No experiment results yet">Run an experiment above.</EmptyState>
        )}
        {runs.data && runs.data.runs.length > 0 && (
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase text-slate-500">
              <tr><th className="py-1">Compare</th><th>Batch</th><th>Iterations</th><th>Seed</th><th>Variants</th><th /></tr>
            </thead>
            <tbody className="divide-y divide-navy-600">
              {runs.data.runs.map((r) => (
                <tr key={r.batch_id} className={r.batch_id === batchId ? "bg-violet/10" : undefined}>
                  <td className="py-1.5">
                    <input type="checkbox" aria-label={`Compare ${r.batch_id}`}
                      checked={cmp.includes(r.batch_id)}
                      onChange={() => setCmp(toggle(cmp, r.batch_id, 4))} />
                  </td>
                  <td className="font-mono">{r.batch_id}</td>
                  <td>{formatInt(r.max_iters)}</td>
                  <td>{r.seed}</td>
                  <td>{r.variants_finished} / {r.variants_total}</td>
                  <td className="text-right">
                    <button className="rounded border border-navy-600 px-2 py-1 text-xs hover:bg-navy-700"
                      onClick={() => setBatchId(r.batch_id)}>View</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      {detail.error && <ErrorState error={detail.error} />}
      {detail.data && (
        <>
          <Card
            title={`Results: ${detail.data.batch_id}`}
            action={
              <a className="rounded border border-navy-600 px-2 py-1 text-xs hover:bg-navy-700"
                href={experimentCsvUrl(detail.data.batch_id)}>Download CSV</a>
            }>
            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs" aria-label="Experiment results">
                <thead className="text-slate-500">
                  <tr>
                    {["Variant", "Parameters", "Train loss", "Val loss", "Best val", "Best iter", "Val-train", "Time"].map((h) => (
                      <th key={h} className="pr-3 pb-1">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.run} title={r.label}>
                      <td className="pr-3">{r.run}</td>
                      <td className="pr-3">{formatInt(r.parameters)}</td>
                      <td className="pr-3">{formatLoss(r.final_train_loss)}</td>
                      <td className="pr-3">{formatLoss(r.final_val_loss)}</td>
                      <td className="pr-3">{formatLoss(r.best_val_loss)}</td>
                      <td className="pr-3">{formatInt(r.best_iter)}</td>
                      <td className="pr-3">{formatLoss(r.val_minus_train)}</td>
                      <td>{formatDuration(r.training_time_s)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-xs text-slate-500">
              Numbers come from each run&apos;s summary.json. Training times are wall-clock and were
              noisy in earlier batches (up to 4x between identical models), so do not read cost
              from them. A single seed cannot separate small differences from noise.
            </p>
          </Card>

          <Card title="Charts">
            <label className="text-sm text-slate-400">Experiment
              <select className={`${FIELD} ml-2`} value={expKey} onChange={(e) => setExpKey(e.target.value)}>
                {defs.data?.experiments.map((e) => <option key={e.key} value={e.key}>{e.title}</option>)}
              </select>
            </label>
            {shown.length === 0 ? (
              <p className="mt-3 text-sm text-slate-500">No finished runs of this experiment in this batch.</p>
            ) : (
              <div className="mt-3 grid gap-4 lg:grid-cols-2">
                <div className="h-64" data-testid="curves">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={curveData}>
                      <CartesianGrid stroke="#26325c" strokeDasharray="3 3" />
                      <XAxis dataKey="iter" stroke="#94a3b8" />
                      <YAxis stroke="#94a3b8" domain={["auto", "auto"]} />
                      <Tooltip contentStyle={TIP} />
                      <Legend />
                      {shown.map((r, i) => (
                        <Line key={r.run} dataKey={r.run} stroke={COLORS[i % COLORS.length]}
                          dot={false} isAnimationActive={false} />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={shown.map((r) => ({ name: r.run, best: r.best_val_loss }))}>
                      <CartesianGrid stroke="#26325c" strokeDasharray="3 3" />
                      <XAxis dataKey="name" stroke="#94a3b8" />
                      <YAxis stroke="#94a3b8" domain={["auto", "auto"]} />
                      <Tooltip contentStyle={TIP} cursor={{ fill: "rgba(139,92,246,0.15)" }} />
                      <Bar dataKey="best" name="best validation loss" fill="#8b5cf6" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
            <p className="mt-2 text-xs text-slate-500">
              Left: validation loss over training. Right: best validation loss (lower is better);
              the bar axis does not start at zero.
            </p>
          </Card>
        </>
      )}

      <Card title="Compare batches (for example, the same setup with different seeds)">
        {cmp.length < 2 && (
          <p className="text-sm text-slate-500">Tick two to four batches in the table above.</p>
        )}
        {compare.error && <ErrorState error={compare.error} />}
        {compare.data && (
          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-xs" aria-label="Batch comparison">
              <thead className="text-slate-500">
                <tr><th className="pr-3">Variant</th>
                  {compare.data.run_ids.map((id) => <th key={id} className="pr-3">{id}</th>)}
                  <th>Spread</th></tr>
              </thead>
              <tbody>
                {compare.data.variants.map((v) => (
                  <tr key={v.variant}>
                    <td className="pr-3">{v.variant}</td>
                    {compare.data.run_ids.map((id) => (
                      <td key={id} className="pr-3">{formatLoss(v.best_val_loss[id])}</td>
                    ))}
                    <td>{(() => { const s = spread(Object.values(v.best_val_loss)); return s === null ? DASH : s.toFixed(4); })()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-2 text-xs text-slate-500">
              Best validation loss per variant. Spread is the largest minus the smallest value
              across the selected batches. Differences between variants that are smaller than the
              spread of one variant across seeds should not be trusted.
            </p>
          </div>
        )}
      </Card>
    </div>
  );
}
import { useEffect, useState } from "react";

import { Card } from "../components/ui/Card";
import { MetricCard } from "../components/ui/MetricCard";
import { EmptyState, ErrorState } from "../components/ui/states";
import { LaunchForm } from "../features/training/LaunchForm";
import { LossChart } from "../features/training/LossChart";
import { useJobLogs, useResumeJob, useStopJob, useTrainingJob, useTrainingJobs } from "../hooks/useApi";
import { useTrainingSocket } from "../hooks/useTrainingSocket";
import type { JobStatus } from "../types/api";
import { DASH, formatDuration, formatLoss } from "../utils/format";

const ACTIVE: JobStatus[] = ["QUEUED", "RUNNING"];

export function TrainingLab() {
  const jobs = useTrainingJobs();
  const [selected, setSelected] = useState<string | null>(null);
  const trainingJobs = (jobs.data?.jobs ?? []).filter((x) => x.kind === "training");

  useEffect(() => {  // default to the newest job
    if (selected === null && jobs.data && jobs.data.jobs.length > 0) {
      setSelected(jobs.data.jobs[0]?.job_id ?? null);
    }
  }, [jobs.data, selected]);

  const job = useTrainingJob(selected);
  const socket = useTrainingSocket(selected);
  const active = job.data ? ACTIVE.includes(job.data.status) : false;
  const logs = useJobLogs(selected, active);
  const stop = useStopJob();
  const resume = useResumeJob();

  const j = job.data;
  const best = socket.rows.length > 0 ? Math.min(...socket.rows.map((r) => r.val_loss)) : null;
  const last = socket.rows[socket.rows.length - 1];
  const maxIters = j?.config.training.max_iters;
  const canResume = j !== undefined && (j.status === "STOPPED" || j.status === "FAILED" || j.status === "COMPLETED");

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold text-slate-50">Training Lab</h1>
      <p className="max-w-3xl text-sm text-slate-400">
        Training runs in a separate worker process, one job at a time. Charts show real
        evaluation results, recorded every eval interval.
      </p>
      <LaunchForm onCreated={setSelected} />

      <Card title="Runs">
        {jobs.error && <ErrorState error={jobs.error} />}
        {trainingJobs.length === 0 && <EmptyState title="No training runs yet" />}
        {trainingJobs.length > 0 && (
          <ul className="divide-y divide-navy-600 text-sm">
            {trainingJobs.map((x) => (
              <li key={x.job_id}>
                <button onClick={() => setSelected(x.job_id)}
                  className={`flex w-full items-center justify-between gap-3 px-2 py-2 text-left ${x.job_id === selected ? "bg-violet/10" : "hover:bg-navy-700"}`}>
                  <span className="font-mono">{x.run_name}</span>
                  <span className="text-xs text-slate-400">{x.model_type} | {x.status}
                    {x.progress !== null ? ` | ${Math.round(x.progress * 100)}%` : ""}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </Card>

      {job.error && <ErrorState error={job.error} />}
      {j && (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
            <MetricCard label="Status" value={j.status} hint={j.stop_requested ? "stop requested" : undefined} />
            <MetricCard label="Iteration" value={last ? `${last.iter} / ${maxIters ?? DASH}` : DASH} />
            <MetricCard label="Train loss" value={formatLoss(last?.train_loss)} />
            <MetricCard label="Val loss (best)" value={`${formatLoss(last?.val_loss)} (${formatLoss(best)})`} />
            <MetricCard label="Duration" value={formatDuration(last?.elapsed_s)}
              hint={`device setting: ${j.device}`} />
          </div>
          <div className="h-2 w-full overflow-hidden rounded bg-navy-700" aria-label="progress"
            role="progressbar" aria-valuenow={Math.round((j.progress ?? 0) * 100)} aria-valuemin={0} aria-valuemax={100}>
            <div className="h-full bg-violet" style={{ width: `${(j.progress ?? 0) * 100}%` }} />
          </div>

          {j.error && <ErrorState error={new Error(j.error)} />}

          <Card
            title={`${j.run_name} (${j.model_type})`}
            action={
              <div className="flex items-center gap-2 text-xs">
                <span className="text-slate-500">live: {socket.state}</span>
                <button className="rounded border border-navy-600 px-2 py-1 hover:bg-navy-700 disabled:opacity-50"
                  disabled={!active || j.stop_requested || stop.isPending}
                  onClick={() => stop.mutate(j.job_id)}>Stop</button>
                <button className="rounded border border-navy-600 px-2 py-1 hover:bg-navy-700 disabled:opacity-50"
                  disabled={!canResume || resume.isPending}
                  onClick={() => resume.mutate({ id: j.job_id, maxIters: null },
                    { onSuccess: (n) => setSelected(n.job_id) })}>Resume</button>
              </div>
            }>
            {socket.rows.length === 0
              ? <EmptyState title="No evaluations yet">The first result appears when the worker finishes its first evaluation.</EmptyState>
              : <LossChart rows={socket.rows} />}
            {(stop.error || resume.error) && <div className="mt-2"><ErrorState error={stop.error ?? resume.error} /></div>}
            <p className="mt-2 text-xs text-slate-500">
              Resume continues from the last checkpoint with the same max iterations
              {j.status === "COMPLETED" ? ". This run already completed, so raise max iterations via the API to extend it" : ""}.
            </p>
          </Card>

          <Card title="Worker log">
            <pre data-testid="log" className="max-h-64 overflow-auto rounded border border-navy-600 bg-navy-900 p-2 font-mono text-xs text-slate-300">
              {logs.data?.lines.join("\n") || "(no output yet)"}
            </pre>
          </Card>
        </>
      )}
    </div>
  );
}
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../services/api";
import type { TrainingJob } from "../types/api";

export const useHealth = () =>
  useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 5000 });

export const useSystemInfo = () => useQuery({ queryKey: ["system"], queryFn: api.systemInfo });

export const useModels = () => useQuery({ queryKey: ["models"], queryFn: api.listModels });

export const useModelDetail = (id: string | null) =>
  useQuery({
    queryKey: ["model", id],
    queryFn: () => api.modelDetail(id as string),
    enabled: id !== null,
  });

export const useModelConfig = (id: string | null) =>
  useQuery({
    queryKey: ["model", id, "config"],
    queryFn: () => api.modelConfig(id as string),
    enabled: id !== null,
  });

export const useModelParameters = (id: string | null) =>
  useQuery({
    queryKey: ["model", id, "parameters"],
    queryFn: () => api.modelParameters(id as string),
    enabled: id !== null,
  });

function useLifecycleMutation(fn: (id: string) => Promise<unknown>) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: fn,
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["models"] });
      void qc.invalidateQueries({ queryKey: ["system"] });
      void qc.invalidateQueries({ queryKey: ["model"] });
    },
  });
}

export const useLoadModel = () => useLifecycleMutation(api.loadModel);
export const useUnloadModel = () => useLifecycleMutation(api.unloadModel);
export const useCompare = () => useMutation({ mutationFn: api.compare });

export const useEncode = (modelId: string | null, text: string) =>
  useQuery({
    queryKey: ["encode", modelId, text],
    queryFn: () => api.encode(modelId as string, text),
    enabled: modelId !== null && text.length > 0,
    retry: false,
    placeholderData: keepPreviousData,
  });

export const useVocabulary = (modelId: string | null) =>
  useQuery({
    queryKey: ["vocabulary", modelId],
    queryFn: () => api.vocabulary(modelId as string),
    enabled: modelId !== null,
    retry: false,
  });

export const useDecode = () =>
  useMutation({
    mutationFn: ({ modelId, ids }: { modelId: string; ids: number[] }) =>
      api.decode(modelId, ids),
  });

export const useAttention = (modelId: string | null, prompt: string, layer: number) =>
  useQuery({
    queryKey: ["attention", modelId, prompt, layer],
    queryFn: () =>
      api.attention({ model_id: modelId as string, prompt, layer, max_tokens: 64 }),
    enabled: modelId !== null && prompt.length > 0,
    retry: false,
    placeholderData: keepPreviousData,
  });

export const useArchitecture = (modelId: string | null) =>
  useQuery({
    queryKey: ["architecture", modelId],
    queryFn: () => api.architecture(modelId as string),
    enabled: modelId !== null,
    retry: false,
  });

export const useForwardPass = () =>
  useMutation({
    mutationFn: ({ modelId, prompt }: { modelId: string; prompt: string }) =>
      api.forwardPass(modelId, prompt),
  });

const ACTIVE = new Set(["QUEUED", "RUNNING"]);

export const useTrainingJobs = () =>
  useQuery({
    queryKey: ["jobs"],
    queryFn: api.trainingJobs,
    refetchInterval: (q) => (q.state.data?.jobs.some((j) => ACTIVE.has(j.status)) ? 2000 : 10000),
  });

export const useTrainingJob = (id: string | null) =>
  useQuery({
    queryKey: ["job", id],
    queryFn: () => api.trainingJob(id as string),
    enabled: id !== null,
    refetchInterval: (q) => (q.state.data && ACTIVE.has(q.state.data.status) ? 2000 : false),
  });

export const useJobLogs = (id: string | null, active: boolean) =>
  useQuery({
    queryKey: ["job-logs", id],
    queryFn: () => api.jobLogs(id as string),
    enabled: id !== null,
    refetchInterval: active ? 2000 : false,
  });

function useJobMutation<V>(fn: (v: V) => Promise<TrainingJob>) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: fn,
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["jobs"] });
      void qc.invalidateQueries({ queryKey: ["job"] });
      void qc.invalidateQueries({ queryKey: ["models"] });
      void qc.invalidateQueries({ queryKey: ["exp-runs"] });
      void qc.invalidateQueries({ queryKey: ["exp-run"] });
    },
  });
}

export const useCreateJob = () => useJobMutation(api.createJob);
export const useStopJob = () => useJobMutation((id: string) => api.stopJob(id));
export const useResumeJob = () =>useJobMutation(({ id, maxIters }: { id: string; maxIters: number | null }) =>api.resumeJob(id, maxIters));

export const useExperimentDefs = () =>
  useQuery({ queryKey: ["exp-defs"], queryFn: api.experiments, staleTime: Infinity });

export const useExperimentRuns = (live: boolean) =>
  useQuery({ queryKey: ["exp-runs"], queryFn: api.experimentRuns, refetchInterval: live ? 3000 : false });

export const useExperimentRun = (id: string | null, live: boolean) =>
  useQuery({
    queryKey: ["exp-run", id],
    queryFn: () => api.experimentRun(id as string),
    enabled: id !== null,
    refetchInterval: live ? 3000 : false,
  });

export const useExperimentCompare = (ids: string[]) =>
  useQuery({
    queryKey: ["exp-compare", ids],
    queryFn: () => api.experimentCompare(ids),
    enabled: ids.length >= 2 && ids.length <= 4,
    retry: false,
  });

export const useRunExperiments = () => useJobMutation(api.runExperiments);

export const useDistribution = (
  modelId: string | null, prompt: string, temperature: number, topK: number | null,
) =>
  useQuery({
    queryKey: ["distribution", modelId, prompt, temperature, topK],
    queryFn: () =>
      api.distribution({ model_id: modelId as string, prompt, temperature, top_k: topK, top_n: 12 }),
    enabled: modelId !== null,
    retry: false,
    placeholderData: keepPreviousData,
  });
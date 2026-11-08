import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../services/api";

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
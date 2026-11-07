import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

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
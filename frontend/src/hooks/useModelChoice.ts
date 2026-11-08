import { useEffect, useState } from "react";

import { useModels } from "./useApi";

/** Model selection that defaults to a preferred checkpoint once the list loads. */
export function useModelChoice(preferred = "gpt_small:best") {
  const models = useModels();
  const [modelId, setModelId] = useState<string | null>(null);

  useEffect(() => {
    const ids = models.data?.models.map((m) => m.model_id) ?? [];
    if (modelId === null && ids.length > 0) {
      setModelId(ids.includes(preferred) ? preferred : (ids[0] ?? null));
    }
  }, [models.data, modelId, preferred]);

  return { models, modelId, setModelId };
}
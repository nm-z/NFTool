import { expect, Page, APIRequestContext } from "@playwright/test";

export const DEFAULT_API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
export const DEFAULT_API_KEY =
  process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";

export async function waitForConnected(page: Page, timeout = 20000) {
  await expect(page.getByText("CONNECTED", { exact: true })).toBeVisible({
    timeout,
  });
}

export async function openLibraryDataTab(page: Page) {
  await page.getByTestId("nav-library").click();
  await page.getByRole("tab", { name: "Data" }).click();
}

export async function openTrainWorkspace(page: Page) {
  await page.getByRole("button", { name: "Train" }).click();
}

export async function openInferenceTab(page: Page) {
  await page.getByRole("tab", { name: "Inference Playground" }).click();
}

export async function setTrainingDefaults(page: Page) {
  await page.getByRole("tab", { name: "Model" }).click();
  await page.getByTestId("input-trial-budget").fill("1");
  await page.getByTestId("input-max-epochs").fill("1");
  await page.getByTestId("input-early-stop-patience").fill("1");
  await page.getByTestId("input-batch-size").fill("2");

  await page.getByRole("tab", { name: "Performance" }).click();
  await page.getByRole("button", { name: "CPU" }).click();
  await page.getByRole("tab", { name: "Model" }).click();
}

export async function selectDatasets(
  page: Page,
  predictorPath: string,
  targetPath: string,
) {
  await page.getByTestId("select-predictors").selectOption(predictorPath);
  await page.getByTestId("select-targets").selectOption(targetPath);
}

export async function fetchRuns(request: APIRequestContext) {
  const res = await request.get(`${DEFAULT_API_URL}/api/v1/training/runs`, {
    headers: DEFAULT_API_KEY ? { "X-API-Key": DEFAULT_API_KEY } : undefined,
  });
  if (!res.ok()) {
    throw new Error(`Failed to fetch runs: ${res.status()}`);
  }
  return res.json();
}

export async function waitForNewRun(
  request: APIRequestContext,
  existingRunIds: Set<string>,
  timeoutMs = 60000,
) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const runs = await fetchRuns(request);
    const runList = Array.isArray(runs) ? runs : [];
    for (const run of runList) {
      const id = String(run.run_id ?? run.id ?? "");
      if (id && !existingRunIds.has(id)) {
        return { id, status: String(run.status || "") };
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error("Timed out waiting for a new training run.");
}

export async function waitForRunStatus(
  request: APIRequestContext,
  runId: string,
  desiredStatus: string,
  timeoutMs = 180000,
) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const runs = await fetchRuns(request);
    const run = Array.isArray(runs)
      ? runs.find((item) => String(item.run_id ?? item.id ?? "") === runId)
      : undefined;
    const status = String(run?.status || "");
    if (status === desiredStatus) {
      return;
    }
    if (status === "failed" || status === "aborted") {
      throw new Error(`Run ${runId} ended in ${status}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
  throw new Error(`Timed out waiting for run ${runId} to reach ${desiredStatus}`);
}

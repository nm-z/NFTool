import { test, expect } from "@playwright/test";
import {
  openLibraryDataTab,
  openInferenceTab,
  selectDatasets,
  setTrainingDefaults,
  waitForConnected,
  waitForNewRun,
  waitForRunStatus,
  fetchRuns,
} from "./utils";

const PREDICTOR_PATH = "data/sample_predictors.csv";
const TARGET_PATH = "data/sample_targets.csv";

test.describe("Core user workflows", () => {
  test("Dataset preview renders summary and table", async ({ page }) => {
    await page.goto("/");
    await page.waitForLoadState("networkidle");
    await waitForConnected(page);

    await openLibraryDataTab(page);

    const pathInput = page.getByTestId("input-dataset-path");
    await pathInput.fill(PREDICTOR_PATH);
    await page.getByTestId("btn-preview-dataset").click();

    await expect(page.getByText("Total Samples")).toBeVisible();
    await expect(page.locator("table")).toContainText("feature_1");
  });

  test("Training run updates metrics and supports inference evaluation", async ({
    page,
    request,
  }) => {
    test.setTimeout(300000);
    await page.goto("/");
    await page.waitForLoadState("networkidle");
    await waitForConnected(page);

    await setTrainingDefaults(page);
    await selectDatasets(page, PREDICTOR_PATH, TARGET_PATH);

    const runsBefore = await fetchRuns(request);
    const existingRunIds = new Set(
      (Array.isArray(runsBefore) ? runsBefore : []).map((run) =>
        String(run.run_id ?? run.id ?? ""),
      ),
    );

    await page.getByTestId("btn-run").click();

    const { id: runId } = await waitForNewRun(request, existingRunIds);

    await expect
      .poll(async () => page.getByTestId("metrics-count").textContent())
      .not.toContain("0 points");

    await waitForRunStatus(request, runId, "completed");

    await openInferenceTab(page);

    const runSelect = page.getByTestId("select-completed-model");
    await expect(runSelect).toBeVisible();
    await expect(runSelect.locator(`option[value="${runId}"]`)).toBeVisible({
      timeout: 20000,
    });
    await runSelect.selectOption(runId);

    await page.getByTestId("btn-execute-inference").click();

    await expect(page.getByText("Accuracy Summary")).toBeVisible({
      timeout: 30000,
    });
    await expect(page.getByText("MAE:")).toBeVisible();
  });
});

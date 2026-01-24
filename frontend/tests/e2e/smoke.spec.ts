import { test, expect } from "@playwright/test";

test.describe("Workflow tests", () => {
  test("CNN Training and Log Capture", async ({ page }) => {
    test.setTimeout(60000);
    await page.goto("/");
    await page.waitForLoadState("networkidle");
    
    // Wait for backend connection
    await expect(page.getByText("CONNECTED", { exact: true })).toBeVisible({
      timeout: 15000,
    });

    // 1. Click CNN in Inspector
    // We target the button explicitly
    const cnnBtn = page
      .locator("aside.border-l")
      .locator("button", { hasText: "CNN" });
    await cnnBtn.click();
    console.log("Selected CNN model.");

    // 2. Click Run button in Header
    const runBtn = page.getByTestId("btn-run");
    await runBtn.click();
    console.log("Clicked Run button.");

    // 3. Ensure we're in the log view (not tree view)
    const logsViewBtn = page.getByTestId("btn-toggle-logs");
    await expect(logsViewBtn).toBeVisible({ timeout: 5000 });
    await logsViewBtn.click();

    // 4. Wait for logs to populate
    // We look for the "Process Stream" container.
    // Logs are formatted as [{time}] or [{time} (Epoch N)]
    const logContainer = page.locator(".flex-1.overflow-y-auto.p-3.font-mono");
    // Wait for either timestamp bracket or epoch indicator
    await expect(logContainer).toContainText(/\[|\bEpoch\b/, { timeout: 30000 });

    // 5. Capture and Output Logs
    // Wait a bit to get some logs
    await page.waitForTimeout(3000);
    const logs = await logContainer.innerText();

    console.log("\n=== CAPTURED PROCESS STREAM LOGS ===");
    console.log(logs);
    console.log("====================================\n");

    expect(logs.length).toBeGreaterThan(0);
  });
});
